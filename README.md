# Official code repository for the paper ""


## Preface
### Invitation for coorporations on topics related to our work
The idea presented in our current work has the potential to be further expanded into various domains of generative modeling. As such, We are eager to engage in future collaborations and explore research topics that are related to the findings of our paper. We welcome opportunities to collaborate with researchers/organisations that share a similar research interest in diffusion models. Please reach out to me at jiachenlei@zju.edu.cn to discuss potential collaborations.

### Seeking available PHD positions in universities of the USA, 2024 fall admission, or possbile intern positions
As the first author of the paper, I am devoted to pursuing a CS **PhD**, specializing in areas such as **generative modeling, 3D reconstruction, representation learning**, and related fields. I am driven by a strong passion for pushing the boundaries of knowledge in these domains and leveraging the power of AI to address real-world challenges. I am humbly seeking available phd positions that align with my research interests and intern opportunities for collaboration and growth. Please contact me via jiachenlei@zju.edu.cn to discuss potential opportunities for phd or intern positions. I am excited about the prospect of joining a dynamic research community and making significant contributions to it.


<!-- I am particularly interested in exploring doctoral programs at prestigious institutions such as MIT, Caltech, CMU, Stanford, Berkeley, and other leading research universities that have a strong focus on deep learning and cutting-edge AI research. -->

### Schedule
As our current project is still a work in progress, we plan to gradually present more experiments and analyses on our method in the near future.
- [ ] Appendix: present more details of current project. Temporarily, you could refer to configuration files in this repo for details of our experiments.
- [ ] Analysis on high-resolution images: How should we utilize masked score matching efficiently on high-resolution (e.g. 256x256) images without using DWT?  
- [ ] Experiments on natural Datasets other than human face: for example, LSUN.  
- [ ] Experiments on more diffusion variants (e.g., beyond the DDPM framework): NCSM, etc
- [ ] Experiments that implement masked score matching in latent space.
- [ ] * Video data
- [ ] * Adapting current method to CNN-based variants.
- [ ] More...


## FAQ
For your convinience, we will present frequently asked quetions here.


## Documentation

### (First-stage training) masked score matching (MSM)
```python

```

### (Second-stage training) denoising score matching (DSM, objective used in DDPM)
```python

```

### Sampling
To sample from a model, either trained by MSM or DSM, use the following command:

```python

```

### Evaluation
To compute FID score on generated images and the reference images which, under most circumstances, is the training set, use the following command:

```python

```


## Acknowledgement
This repository is heavily based on [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) by lucidrains. We also referred to repositories:  
- [U-ViT](https://github.com/baofff/U-ViT) by baofff,  
- [pytorch-fid](https://github.com/mseitzer/pytorch-fid) by mseitzer,  
- [stable-diffusion](https://github.com/CompVis/stable-diffusion/tree/main) by CompVis,  
- [MAE-pytorch](https://github.com/pengzhiliang/MAE-pytorch) by pengzhiliang,  
- [dpm-solver](https://github.com/LuChengTHU/dpm-solver/tree/main) by LuChengTHU,  
  
Thanks for your contributions.

