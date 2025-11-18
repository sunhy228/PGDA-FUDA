# PGDA-FUDA
Pytorch implementation of PGDA (Prototype-Guided Diffusion Alignment Method) 

# Datasets
The used datasets can be downloaded at [https://github.com/jindongwang/transferlearning/tree/master/data] (under the folder data)
Split files are provided in data/splits, supported datasets are Office, Office-Home, VisDA-2017, and DomainNet

# Requirements
torch == 2.6.0 torchaudio ==2.6.0 torchvision == 0.21.0 python == 3.9.23 scikit-learn == 1.6.1 numpy == 1.26.4

# Configuration
The hyper-parameters are defined in pgda/config/config.yml.

# Training
In order to obtain the class prototype and diffusion model, you may run: 
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=gpu train.py \
--batchsize 64 \
--interval 1 \
--dataset office \
--cdim 128 

# Sampling
In order to sample and classify images from the pretrained model, you may run:
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=gpu sample.py --ddim True  --fid True

# Thanks
The settings involved in our paper follow PCS [https://github.com/zhengzangw/PCS-FUDA].
Classifier-free diffusion model's implementation is mainly based on CFG [https://github.com/coderpiaobozhe/classifier-free-diffusion-guidance-Pytorch]
Thanks for their repositories, which facilitate our code.