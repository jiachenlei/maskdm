# Official implementation: "Masked Diffusion Models are Fast Distribution Learners"

> Abstract:  
Diffusion models have emerged as the de-facto generative model for image synthesis, yet they entail significant training overhead, hindering the technique’s broader
adoption in the research community. We observe that these models are commonly
trained to learn all fine-grained visual information from scratch, thus motivating
our investigation on its necessity. In this work, we show that it suffices to set up pretraining stage to initialize a diffusion model by encouraging it to learn some primer
distribution of the unknown real image distribution. Then the pre-trained model can
be fine-tuned for specific generation tasks efficiently. To approximate the primer
distribution, our approach centers on masking a high proportion (e.g., up to 90%)
of an input image and employing masked denoising score matching to denoise
visible areas. Utilizing the learned primer distribution in subsequent fine-tuning,
we efficiently train a ViT-based diffusion model on CelebA-HQ 256 × 256 in the
raw pixel space, achieving superior training acceleration compared to denoising
diffusion probabilistic model (DDPM) counterpart and a new FID score record of
6.73 for ViT-based diffusion models. Moreover, our masked pre-training technique
can be universally applied to various diffusion models that directly generate images in the pixel space, aiding in the learning of pre-trained models with superior
generalizability. For instance, a diffusion model pre-trained on VGGFace2 attains
a 46% quality improvement through fine-tuning on only 10% data from a different
dataset. Our code will be made publicly available.

For a *more intuitive* introduction of our method, you could refer to our [paper]() for more details of our method.  


### Schedule
As our current project is still a work in progress, we plan to gradually present more analysis and details on our method in the near future.
- [x] Submit Appendix of our paper
- [x] Analysis on high-resolution images (e.g. 256x256, 512x512) images **without using DWT** in raw pixel space. we observed some interesting phenomenons that are different from current results on CelebA 64x64
- [x] Experiments on natural Datasets other than human face: e.g., CIFAR10, LSUN, ImageNet.  
- [x] Experiments on applying our method to score-based models (e.g., beyond the DDPM framework): NCSM, etc
- [] Experiments that analyze masked score matching in latent space
- More...


---

## Checkpoints (ToDo)

<!-- |  Name  |  Model   | Dataset | Desciption | Link |
| ------ | -------- | ------ | --- | --- |
| pretrain_celebahq_m90block4_20.pt | MaskDM-B | CelebA-HQ  256x256 | pre-trained with 4x4 block-wise masking at a 90% mask rate for 200k steps | [HuggingFace](https://huggingface.co/jiachenlei/maskdm/blob/main/pretrain/pretrain_celebahq_m90block4_20.pt) |
| pretrain_vggface2_m90block4_20.pt| MaskDM-B | Vggface2  256x256 | pre-trained with 4x4 block-wise masking at a 90% mask rate for 200k steps | [HuggingFace](https://huggingface.co/jiachenlei/maskdm/blob/main/pretrain/pretrain_vggface2_m90block4_20.pt) |
| pretrain_celebahq_maskdm_dwt_20.pt| MaskDM-L | CelebA-HQ  256x256| Used DWT. pre-trained with 4x4 block-wise masking at a 70% mask rate for 200k steps | [HuggingFace](https://huggingface.co/jiachenlei/maskdm/blob/main/celebahq_dwt/pretrain_celebahq_maskdm_dwt_20.pt) |
| celebahq_maskdm_dwt_55.pt | MaskDM-L | CelebA-HQ 256x256 | Used DWT. pre-trained with 4x4 block-wise masking at a 70% mask rate for 200k steps and fine-tuned for 550k steps | [HuggingFace](https://huggingface.co/jiachenlei/maskdm/blob/main/celebahq_dwt/celebahq_maskdm_dwt_55.pt) | -->

---

## Documentation
- For the usage of `Acclerate` or `Wandb` library, please refer to the official documentation.

### Environment
```python
accelerate
timm
torch
PyWavelets
```
We also released a [docker image](https://hub.docker.com/layers/jiachenlei/maskdm/xformer/images/sha256-38a8e8e6cb06dc8938bf79a564ab1e3c999cc6a5b307c6f579c41fd456c79476?context=repo).


### Training
#### train on CelebA-HQ
```python
# python command
# using mask or not is specified in the config file 
# you can specify more command-line arguments as detailed in our code
# add the flag --debug prevents from logging online to Wandb

accelerate launch --num_processes 2 --gpu_ids 0,1 --mixed_precision fp16 main.py --name celebahq_base0 --config configs/celebahq_base.yml --training_steps 200000 --pretrained_model_ckpt /path/to/pretrained weights --debug

```
#### pre-train on CelebA
```python
accelerate launch --num_processes 2 --gpu_ids 0,1 --mixed_precision fp16 main.py --name celeba_small0 --config configs/ablation/pretrain/block2/celeba_10p.yml --training_steps 200000 --debug
```

### Sampling
To sample from a model, either trained by MSM or DSM, use the following command:

```bash

accelerate launch eval.py --name temp --config /path/to/config/file.yml --bs 64 --num_samples 10000 --ckpt /path/to/ckpt1.pt /path/to/ckpt2.pt /path/to/ckpt3.pt --output /path/to/save/samples/for/ckpt1.pt /path/to/save/samples/for/ckpt2.pt /path/to/save/samples/for/ckpt3.pt
```

### Evaluation
To compute FID score on generated images and the reference images which, under most circumstances, is the training set, use the following command:

```python

python -m tools.pytorch_fid --device cuda:0 /path/to/image/folder1 /path/to/image/folder2

# notice that the structure of the folder provided in the path should look like:
# - /path/to/image/folder1
#     - image file1
#     - image file2
#     ...

```

## Acknowledgement
This repository is heavily based on [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) by lucidrains. We also referred to repositories:  
- [U-ViT](https://github.com/baofff/U-ViT) by baofff,  
- [pytorch-fid](https://github.com/mseitzer/pytorch-fid) by mseitzer,  
- [stable-diffusion](https://github.com/CompVis/stable-diffusion/tree/main) by CompVis,  
- [MAE-pytorch](https://github.com/pengzhiliang/MAE-pytorch) by pengzhiliang,  
- [dpm-solver](https://github.com/LuChengTHU/dpm-solver/tree/main) by LuChengTHU,  
  
Thanks for open sourcing.

