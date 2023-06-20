# Official implementation: "Masked Diffusion Models are Fast Learners"

For a *more intuitive* introduction of our method, you could refer to our [project website](). Or you could refer to our [paper]() for more details of our method. 

**What did we do in summary?**:  
Diffusion models have emerged as the de-facto technique for image generation, yet
they entail significant computational overhead, hindering the technique’s broader
application in the research community. We propose a prior-based denoising training
framework, the first to incorporate the pre-train and fine-tune paradigm into the
diffusion model training process, which substantially improves training efficiency
and shows potential in facilitating various downstream tasks. Our approach centers
on masking a high proportion (e.g., up to 90%) of the input image and employing
masked score matching to denoise the visible areas, thereby guiding the diffusion
model to learn more salient features from training data as prior knowledge. By
utilizing this masked learning process in a pre-training stage, we efficiently train the
ViT-based diffusion model on CelebA-HQ 256×256 in the pixel space, achieving a
4x acceleration and enhancing the quality of generated images compared to DDPM.
Moreover, our masked pre-training technique is universally applicable to various
diffusion models that directly generate images in the pixel space and facilitates
learning pre-trained models with excellent generalizability: a diffusion model
pre-trained on VGGFace2 attains a 46% quality improvement through fine-tuning
with merely 10% local data.  
 

<!-- ## Preface
### Invitation for coorporations on topics related to our work
The idea presented in our current work has the potential to be further expanded into various domains of diffusion models. As such, We are eager to engage in future collaborations and explore research topics that are related to the findings of our paper. We sincerely welcome opportunities to collaborate with researchers/organisations that share a similar research interest. Please reach out to me at jiachenlei@zju.edu.cn to discuss potential collaborations.


### Seeking available PHD positions in universities of the USA, 2024 fall admission, or possbile intern positions
As the first author of the paper, I am devoted to pursuing a **CS PhD**, specializing in areas such as **generative modeling, 3D reconstruction, representation learning**, and related fields. I am driven by a strong passion for pushing the boundaries of knowledge in these domains and combining current theory and the deep learning to address real-world challenges. I am humbly seeking available phd positions that align with my research interests and intern opportunities for collaboration and growth. Please contact me via jiachenlei@zju.edu.cn to discuss potential opportunities for phd or intern positions. I am excited about the prospect of joining a dynamic research community and making significant contributions to it. -->


### Schedule
As our current project is still a work in progress, we plan to gradually present more analysis and details on our method in the near future.
- [ ] Submit Appendix of our paper
- [ ] Analysis on high-resolution images (e.g. 256x256, 512x512) images **without using DWT** in raw pixel space. we observed some interesting phenomenons that are different from current results on CelebA 64x64
- [ ] Experiments on natural Datasets other than human face: e.g., CIFAR10, LSUN, ImageNet.  
- [ ] Experiments on applying our method to score-based models (e.g., beyond the DDPM framework): NCSM, etc
- [ ] Experiments that analyze masked score matching in latent space
- [ ] More...


## FAQ
For a *more intuitive* introduction of our method, you could refer to our [project website](). Or you could refer to our [paper]() for more details of our method.  

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

## Documentation
- For your convinience, we also provide a bash script, `run.sh`, which supports masked pre-training, denoising fine-tuning and sampling.  
- For the usage of `Acclerate` or `Wandb` library, please refer to the official documentation.

### Environment
```python
accelerate
timm
torch
PyWavelets
```
We also released a [docker image]().


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

