# Official code repository for the paper ""


## Preface
### PHD application
As the first author of the paper, I am devoted to pursuing a **PhD** in deep learning, specializing in areas such as **generative modeling, 3D reconstruction, and related fields**. I am driven by a strong passion for pushing the boundaries of knowledge in these domains and leveraging the power of artificial intelligence to address real-world challenges. I am humbly seeking available positions that align with my research interests and offer opportunities for collaboration and growth. Please contact me via jiachenlei@zju.edu.cn to discuss potential opportunities for research collaboration or to provide information about available positions. I am excited about the prospect of joining a dynamic research community and making significant contributions to it.

<!-- I am particularly interested in exploring doctoral programs at prestigious institutions such as MIT, Caltech, CMU, Stanford, Berkeley, and other leading research universities that have a strong focus on deep learning and cutting-edge AI research. -->
### Coorporation on topics related to our work
Besides, the idea presented in our current work has the potential to be further expanded into various domains of generative modeling. As such, I am eager to engage in future collaborations and explore research topics that are related to the findings of our paper. I welcome opportunities to collaborate with researchers who share a similar research interest in diffusion models, and related areas. Together, we can explore novel applications, develop innovative methodologies, and advance the field by collectively pushing the boundaries of knowledge. I invite interested individuals and institutions to reach out to me at jiachenlei@zju.edu.cn to discuss potential collaborations.


## Training

### first-stage: masked score matching
```python

```

### second-stage: denoising score matching (common diffusion training)
```python

```

## Sampling
To sample from a model, either trained by masked score matching or denoising score matching, use the following command:

```python

```

## Evaluation
To compute FID score on generated images and the reference images which is the model training set under most circumstances, use the following command:

```python

```

## FAQ
For your convinience, we present some frequently asked quetions here.

## Acknowledgement
This repository is heavily based on [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) by lucidrains. We also referred to repositories:  
(1) [U-ViT](https://github.com/baofff/U-ViT) by baofff,  
(2) [pytorch-fid](https://github.com/mseitzer/pytorch-fid) by mseitzer,  
(3) [stable-diffusion](https://github.com/CompVis/stable-diffusion/tree/main) by CompVis,  
(4) [MAE-pytorch](https://github.com/pengzhiliang/MAE-pytorch) by pengzhiliang,  
(5) [dpm-solver](https://github.com/LuChengTHU/dpm-solver/tree/main) by LuChengTHU,  
Thanks for your contributions.

