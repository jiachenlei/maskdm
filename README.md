# Official implementation: "Masked Diffusion Models are Fast Learners"

For an *intuitive* introduction of our method, you could refer to our [project website](https://sites.google.com/view/maskdm/main). Or you could refer to our [paper](https://arxiv.org/abs/2306.11363) for more details. 

## Preface
### Invitation for coorporations on topics related to our work
The idea presented in our current work has the potential to be further expanded into various domains of diffusion models. As such, we are eager to explore research topics that are related to the findings of our paper. We sincerely welcome opportunities to collaborate with researchers/organisations that share a similar research interest. Please reach out to me at jiachenlei@zju.edu.cn to discuss potential collaborations.


### First Author Seeking PHD positions, 2024 fall admission
Obtained my M.S. degree from Zhejiang University this year, I am devoted to pursuing a **CS PhD**, specializing in areas such as **generative modeling, 3D reconstruction, representation learning**, and related fields. Driven by a strong passion for combining theory with the deep learning to address real-world challenges, I am seeking available phd position that aligns with my research interests. Please contact me via jiachenlei@zju.edu.cn to discuss potential opportunities for phd position.

### Schedule
As our current project is still a work in progress, we plan to gradually present more analysis and details on our method in the near future.
- [ ] Analysis on high-resolution images (e.g. 256x256, 512x512) images **without using DWT** in raw pixel space. we observed some interesting phenomena that are different from current results on CelebA 64x64
- [ ] Experiments on natural Datasets other than human face: e.g., CIFAR10, LSUN, ImageNet.  
- [ ] Experiments on applying our method to score-based models (e.g., beyond the DDPM framework): NCSM, etc
- [ ] Experiments that analyze masked score matching in latent space
- [ ] More...

## Errata
- In unconditional image synthesis experiment, the reported time expense of baseline model is inaccurate and is **expeceted to be larger**

We trained the baseline model for 940k steps which took 120 hours on 8 V100s, ~1.3 hour per 10k steps. However, we reported 32 V100-days as the time expense of baseline model in our v1.0 paper, which is smaller than the actual value. The correct time expense is ~**38 V100-day**. As for our method, the reported values are double-checked and found to be precise.

## FAQ

For your convinience, we present frequently asked quetions here.  
> Is the DWT neccessary for your method to work in pixel space?  

No. Due to limited computation resources, we currently can't afford training diffusion models directly on 256x256 images, considering it takes longer time for models to reach a satisfying FID score. **We plan to conduct more experiments without DWT**.  

---

## Checkpoints (Updating)

|  Name  |  Model   | Dataset | Desciption | Link |
| ------ | -------- | ------ | --- | --- |
| pretrain_celebahq_m90block4_20.pt | MaskDM-B | CelebA-HQ  256x256 | pre-trained with 4x4 block-wise masking at a 90% mask rate for 200k steps | [HuggingFace](https://huggingface.co/jiachenlei/maskdm/blob/main/pretrain/pretrain_celebahq_m90block4_20.pt) |
| pretrain_vggface2_m90block4_20.pt| MaskDM-B | Vggface2  256x256 | pre-trained with 4x4 block-wise masking at a 90% mask rate for 200k steps | [HuggingFace](https://huggingface.co/jiachenlei/maskdm/blob/main/pretrain/pretrain_vggface2_m90block4_20.pt) |
| pretrain_celebahq_maskdm_dwt_20.pt| MaskDM-L | CelebA-HQ  256x256| Used DWT. pre-trained with 4x4 block-wise masking at a 70% mask rate for 200k steps | [HuggingFace](https://huggingface.co/jiachenlei/maskdm/blob/main/celebahq_dwt/pretrain_celebahq_maskdm_dwt_20.pt) |
| celebahq_maskdm_dwt_55.pt | MaskDM-L | CelebA-HQ 256x256 | Used DWT. pre-trained with 4x4 block-wise masking at a 70% mask rate for 200k steps and fine-tuned for 550k steps | [HuggingFace](https://huggingface.co/jiachenlei/maskdm/blob/main/celebahq_dwt/celebahq_maskdm_dwt_55.pt) |

---


## Environment

```
accelerate
timm
torch/torchvision
einops
wandb
ema-pytorch
PyWavelets
```
We also released a [docker image](). After download the docker image, You could run the container by the following command:

```bash
docker run -it --shm-size 50g --gpus all -v /path/to/data:/path/to/data -v /path/to/code:/path/to/code -w /path/to/code jiachenlei/maskdm:latest
```


## Documentation
- For your convinience, we also provide a bash script, `run.sh`, which supports masked pre-training, denoising fine-tuning and sampling.  
- For the usage of `Acclerate` or `Wandb` library, please refer to the official documentation.


### Training
```python
# python command
# using mask or not is specified in the config file 
# you can specify more command-line arguments as detailed in our code

accelerate launch main.py --name temp # name of experiment
    --config /path/to/config/file.yml # path to config file
    --training_steps 200000 # total training steps
    --debug # no wandb logging. By removing this line, you could record the log online.
```

### Sampling
To sample from a model, either trained by MSM or DSM, use the following command:

```bash
# python command
# you can specify more command-line arguments as detailed in our code

accelerate launch eval.py --name temp # name of experiment
    --config /path/to/config/file.yml # path to config file
    --bs 64 # sampling batch size
    --num_samples 10000 # number of samples to generate
    --ckpt /path/to/ckpt1.pt /path/to/ckpt2.pt /path/to/ckpt3.pt # ckpt path, accept multiple ckpts seperated by space
    --output /path/to/save/samples/for/ckpt1.pt /path/to/save/samples/for/ckpt2.pt /path/to/save/samples/for/ckpt3.pt # output path, accept multiple paths seperated by space
    --sampler ddim # currently we only support DDIM sampling
```

### Using Bash Script for Training and Sampling
Pay attention to the default experiment settings in our bash script. You can modify the settings as needed.

```bash
# bash command
# using 4 gpus: 0,1,2,3, the experiment name is `temp` and config file: `/path/to/config/file.yml`

# run masked pre-training
bash run.sh 0,1,2,3 mask temp,/path/to/config/file.yml

# run denoising fine-tuning
# you could choose to specify pretrained weights or not
bash run.sh 0,1,2,3 mask temp,/path/to/config/file.yml,/path/to/checkpoint.pt
# train ddpm from scratch without loading pre-trained weights:(Don't Forget the Comma at the end)
bash run.sh 0,1,2,3 mask temp,/path/to/config/file.yml,

# run sampling
bash run.sh 0,1,2,3 mask temp,/path/to/config/file.yml

# or automatically run masked pre-training, denoising fine-tuning, sampling in a sequence by:

bash run.sh 0,1,2,3 mask,ddpm,test temp,/path/to/config/file.yml temp,/path/to/config/file.yml,/path/to/checkpoint.pt temp,/path/to/config/file.yml

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
  
Thanks for your contributions.

