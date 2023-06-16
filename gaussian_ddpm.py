import math
from random import random
from functools import partial
from collections import namedtuple
import pywt

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchvision import transforms as T

from einops import rearrange, reduce

from PIL import Image
import numpy as np
from tqdm.auto import tqdm

from utils.helper import maybe_unnormalize_to_zero_to_one, maybe_normalize_to_neg_one_to_one, default, extract, identity

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)


def cosine_beta_schedule(timesteps, s = 0.008, shift=False, d=64):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2

    if shift:
        # shift schedule
        # As proposed in "simple diffusion: End-to-end diffusion for high resolution images"
        alphas_cumprod = torch.sigmoid( torch.log( (64/d)**2 * (alphas_cumprod/(1-alphas_cumprod)) ) )

    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        loss_weight = "p2",
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1. ,

        ae_model = None,  # None if train beyond latent space

        clip_denoised = True, 
        clip_max =  1, 
        clip_min = -1, 

        normalization= True, # whether normalize to [-1, 1] at training or denormalize at sampling.
        channels = 3,
        shift=False # use shifted beta schedule or not
    ):
        super().__init__()
        # assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        # assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        # self.channels = self.model.in_chans
        self.channels = channels
        self.self_condition = False

        self.normalization = normalization

        self.image_size = image_size
        self.objective = objective

        self.clip_denoised = clip_denoised
        self.clip_max = clip_max
        self.clip_min = clip_min

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps, shift=shift, d=image_size)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas

        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('betas', betas)
        register_buffer('alphas', alphas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
 
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting: perception prioritized weighting
        if loss_weight == "p2":
            register_buffer('loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)
        elif loss_weight == "min":
            register_buffer('loss_weight', ( ( 5/(alphas_cumprod/(1-alphas_cumprod)) ).clamp(max=1) )** -p2_loss_weight_gamma )
        else:
            raise NotImplementedError(f"loss weight:{loss_weight} is not supported")

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False):
        model_output = self.model(x, t, y=x_self_cond)
        maybe_clip = partial(torch.clamp, min = self.clip_min, max = self.clip_max) if clip_x_start else identity

        if self.objective == 'pred_noise':

            if self.model.name == "mask_dwt":
                pred_noise = model_output
                x_start = self.predict_start_from_noise(x, t, pred_noise)
                # exclusively clip approximation component to [-1, 1]
                s = 1
                x_start = torch.cat((x_start[:, :3].clamp(min = -s, max = 3*s), x_start[:, 3:].clamp(min = -s, max = s)), dim=1)
            else:
                pred_noise = model_output
                x_start = self.predict_start_from_noise(x, t, pred_noise)
                x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)
 
    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond, clip_x_start=False)

        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(self.clip_min, self.clip_max)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, 
                                                                          x_self_cond = x_self_cond,
                                                                          clip_denoised = self.clip_denoised)

        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise

        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, guidance_weight=0, guidance_idx=None, use_corrector=False):
        # NOTE: ddpm sampling does not support cfg guidance yet
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps, disable= (str(device) != "cuda:0") ):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)

            if use_corrector:
                batched_times = torch.full((img.shape[0],), t, device = img.device, dtype = torch.long)
                img = self.corrector(img, batched_times)

        if self.model.name == "mask_dwt":
            approx = maybe_unnormalize_to_zero_to_one(img[:, :3], self.normalization)
            detail = img[:, 3:]
            img = torch.cat([approx, detail], dim=1)
        else:
            img = maybe_unnormalize_to_zero_to_one(img, self.normalization)

        return img

    @torch.no_grad()
    def ddim_sample(self, shape, guidance_weight=-1, guidance_idx=None, use_corrector=False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        
        maybe_clip = partial(torch.clamp, min = self.clip_min, max = self.clip_max) if self.clip_denoised else identity

        x_start = None
        img = torch.randn(shape, device = device)
        corrected_img = img.clone()

        def s_fn(img, x_start, pred_noise, time, time_next):

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            return img
    
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', disable= (str(device) != "cuda:0") ):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = self.clip_denoised)

            if guidance_weight >=0 :
                # if use guidance
                cond_pred_noise, cond_x_start, *_ = self.model_predictions(img, time_cond, guidance_idx, clip_x_start = False)
                pred_noise = (1+guidance_weight)*cond_pred_noise - guidance_weight*pred_noise

                x_start = self.predict_start_from_noise(img, time_cond, pred_noise)
                x_start = maybe_clip(x_start)

            if time_next < 0:
                # x0
                img = x_start   
                if use_corrector:
                    pred_noise, x_start, *_ = self.model_predictions(corrected_img, time_cond, self_cond, clip_x_start = self.clip_denoised)
                    corrected_img = self.corrector(x_start, time)

                continue

            img = s_fn(img, x_start, pred_noise, time, time_next)
            if use_corrector:
                pred_noise, x_start, *_ = self.model_predictions(corrected_img, time_cond, self_cond, clip_x_start = self.clip_denoised)
                corrected_img = s_fn(corrected_img, x_start, pred_noise, time, time_next)
                corrected_img = self.corrector(corrected_img, time)

        if self.model.name == "mask_dwt":
            approx = maybe_unnormalize_to_zero_to_one(img[:, :3], self.normalization)
            detail = img[:, 3:]
            img = torch.cat([approx, detail], dim=1)

            if use_corrector:
                approx = maybe_unnormalize_to_zero_to_one(corrected_img[:, :3], self.normalization)
                detail = corrected_img[:, 3:]
                corrected_img = torch.cat([approx, detail], dim=1)
                return img, corrected_img

        else:
            img = maybe_unnormalize_to_zero_to_one(img, self.normalization)

            if use_corrector:
                corrected_img = maybe_unnormalize_to_zero_to_one(corrected_img, self.normalization)
                return img, corrected_img

        return img

    @torch.no_grad()
    def sample(self, batch_size = 16, guidance_weight=0, guidance_idx=None, use_corrector=False):
        image_size, channels = self.image_size, self.channels
        # print(image_size, channels)
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        samples = sample_fn((batch_size, channels, image_size, image_size), guidance_weight=guidance_weight, guidance_idx=guidance_idx, use_corrector=use_corrector)

        if self.model.name == "mask_dwt":
            # samples = torch.cat(samples.split(4, dim=1), dim=0)
            if use_corrector:
                img, corrected_img = samples
                img = img.cpu()
                corrected_img = corrected_img.cpu()
                
                idwt_img = np.stack([ pywt.idwt2((img[i][:3], img[i][3:].split(3, dim=0)), "haar") for i in range(img.shape[0]) ], axis=0)
                idwt_img = torch.from_numpy(idwt_img)

                idwt_corrected_img = np.stack([ pywt.idwt2((corrected_img[i][:3], corrected_img[i][3:].split(3, dim=0)), "haar") for i in range(corrected_img.shape[0]) ], axis=0)
                idwt_corrected_img = torch.from_numpy(idwt_corrected_img)

                return idwt_img, idwt_corrected_img

            else:
                samples = samples.cpu()
                idwt_samples = np.stack([ pywt.idwt2((samples[i][:3], samples[i][3:].split(3, dim=0)), "haar") for i in range(samples.shape[0]) ], axis=0)
                samples = torch.from_numpy(idwt_samples)

        return samples

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def corrector(self, x_t, t):
        n_steps = 2
        target_snr = (self.alphas_cumprod / (1-self.alphas_cumprod))[-1]
        batch, *_ = x_t.shape
        time_cond = torch.full((batch,), t, device=x_t.device, dtype=torch.long)

        for i in range(n_steps):

            grad = self.model(x_t, time_cond)
            noise = torch.randn_like(x_t)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()

            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * self.alphas[time_cond]

            x_mean = x_t + step_size[:, None, None, None] * grad
            x_t = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x_t


    def loss_fn(self, noise, x_start, t, model_out, mask=None):

        if self.loss_type == 'l1':
            ls_fn = F.l1_loss
        elif self.loss_type == 'l2':
            ls_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # prepare targets

        if self.model.name == "mask_dwt":
            extras = self.model.extras
            approx_target = target[:, :3]
            detail_target = target[:, 3:]

            approx_target = rearrange(approx_target, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.model.patch_size, p2=self.model.patch_size)          
            detail_target = rearrange(detail_target, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.model.detail_patch_size, p2=self.model.detail_patch_size)          

            # if mask.sum() != 0:
            B, N, C = approx_target.shape
            approx_target = approx_target[~mask[:, extras:extras+N]].reshape(B, -1, C)
            B, N, C = detail_target.shape
            detail_target = detail_target[~mask[:, -N:]].reshape(B, -1, C)

        elif getattr(self.model, "final_layer", None) :
            target = target
        else:
            extras = self.model.extras
            target = rearrange(target, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.model.patch_size, p2=self.model.patch_size)          
            B, N, C = target.shape
            target = target[~mask[:, extras:]].reshape(B, -1, C)

        if self.model.name == "mask_dwt":
            approx, detail = model_out
            approx_loss = ls_fn(approx, approx_target, reduction = 'none')
            approx_loss = reduce(approx_loss, 'b ... -> b (...)', 'mean')
            approx_loss = approx_loss * extract(self.loss_weight, t, approx_loss.shape)
            approx_loss = approx_loss.mean()

            detail_loss = ls_fn(detail, detail_target, reduction = 'none')
            detail_loss = reduce(detail_loss, 'b ... -> b (...)', 'mean')
            detail_loss = detail_loss * extract(self.loss_weight, t, detail_loss.shape)
            detail_loss = detail_loss.mean()
            return {
                # tensor
                "loss": approx_loss+detail_loss,
                "approx loss": approx_loss,
                "detail loss": detail_loss,
            }
        else:
            loss = ls_fn(model_out, target, reduction = 'none')
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            loss = loss * extract(self.loss_weight, t, loss.shape)

        return {
            # tensor
            "loss": loss.mean(),
        }


    def p_losses(self, x_start, t, mask=None, noise=None, label=None, **kwargs):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        # print(t.shape)
        model_out = self.model(x, t, mask=mask, y=x_self_cond if self.self_condition else label, **kwargs)

        return noise, model_out

    def forward(self, batch, *args, **kwargs):
        img, mask = batch[0], batch[1]
        mask = mask.flatten(1).to(torch.bool)
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        if self.model.name == "mask_dwt":
            # only normalize low-frequency component to [-1,1]
            approx = maybe_normalize_to_neg_one_to_one(img[:, :3], self.normalization)
            detail = img[:, 3:]
            x_start = torch.cat((approx, detail), dim=1)
        else:
            x_start = maybe_normalize_to_neg_one_to_one(img, self.normalization)

        _params = {
            "label": batch[2] if len(batch) > 2 else None
        }

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        noise, model_out = self.p_losses(x_start, t, mask, **_params)

        loss = self.loss_fn(noise, x_start, t, model_out, mask=mask)

        return loss

