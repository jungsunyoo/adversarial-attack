# Adversarial attack on Medical Image Segmentation

## 1. Acquiring signed gradients to create noise
### Based on: [fast gradient signed method (FGSM)](https://arxiv.org/abs/1412.6572)


### Run 
``` AdvImg.py```
### to generate noise (perturbations) 

### Sample outcomes: 

#### 1) Original Image: 

![originalImage](./Figure_0_original_img.png)

#### 2) Signed gradient acquired from AdvImg.py: 

![signedGrad](./Figure_1_advImg_signed_grad.png)

#### 3) Adversarial Image with epsilon = 0.5 (advImg = origImg + epsilon * signedGrad)

![epsilon0.5](./Figure_2_advImg_epsilon0.5.png)

#### 4) Adversarial Image with epsilon = 0.01

![epsilon0.001](./Figure_3_advImg_epsilon0.01.png)


#### If epsilon is small, harder to fool model but the perturbation is hard to notice. Vice versa for bigger epsilon.
