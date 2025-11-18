import os
import sys
import torch
import argparse
from math import ceil

from unet import Unet
from diffusion import GaussianDiffusion
from utils.utils_dm import get_named_beta_schedule
from embedding import ProtoEmbedding

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, init_process_group, destroy_process_group, all_gather, get_world_size
from torchvision import transforms
from torchvision.utils import save_image

@torch.no_grad()
def sample(params:argparse.Namespace):
    """
    Function to generate samples using a trained model.
    """
    assert params.genbatch % (torch.cuda.device_count() * params.clsnum) == 0 , 'please re-set your genbatch!!!'
    # initialize settings
    init_process_group(backend="nccl")
    # get local rank for each process
    local_rank = 0
    # set device
    device = torch.device("cuda", local_rank)
    # load models
    hyperparams_model = {}
    with open(os.path.join(params.path, 'params.txt'), 'r') as f:
        for line in f.readlines():
            key, val = line.split(':')
            hyperparams_model[key] = val[:-1]
    
    net = Unet(
                in_ch = int(hyperparams_model["inch"]),
                mod_ch = int(hyperparams_model["modch"]),
                out_ch = int(hyperparams_model["outch"]),
                ch_mul = [ int(i)  for i in hyperparams_model["chmul"][2:-1].split(",")],
                num_res_blocks = int(hyperparams_model["numres"]),
                cdim = int(hyperparams_model["cdim"]),
                use_conv= True if hyperparams_model["useconv"] == ' True' else False,
                droprate = float(hyperparams_model["droprate"]),
                dtype=torch.float32
            ).to(device)
    print("model is created")
    
    checkpoint = torch.load(os.path.join(params.path, f'checkpoints/ckpt_{params.epoch}_checkpoint.pt'), map_location='cpu')
    net.load_state_dict(checkpoint['net'])

    Proto = ProtoEmbedding(params)
    _, cemb_target = Proto.get_embedding()
    _, target_loader = Proto.get_loader()

    # settings for diffusion model
    betas = get_named_beta_schedule(num_diffusion_timesteps = int(hyperparams_model["T"]))
    diffusion = GaussianDiffusion(
                    dtype = torch.float32,
                    model = net,
                    betas = betas,
                    w = float(hyperparams_model["w"]),
                    v =  float(hyperparams_model["v"]),
                    device = device
                )
    # DDP settings
    diffusion.model = DDP(
                            diffusion.model,
                            device_ids = [local_rank],
                            output_device = local_rank
                        )
    cemblayer = DDP(
                    cemblayer,
                    device_ids = [local_rank],
                    output_device = local_rank
                )
    # eval mode
    #print(hyperparams_model["dataset"].replace(' ','') == 'stl10')
    diffusion.model.eval()

    cnt = torch.cuda.device_count()
    if params.fid:
        numloop = ceil(params.genum  / params.genbatch)
    else:
        numloop = 1
    each_device_batch = params.genbatch // cnt
    # label settings
    if params.label == 'range':
        lab = torch.ones(params.clsnum, each_device_batch // params.clsnum).type(torch.long) \
            * torch.arange(start = 0, end = params.clsnum).reshape(-1,1)
        lab = lab.reshape(-1, 1).squeeze()
        lab = lab.to(device)
    else:
        lab = torch.randint(low = 0, high = params.clsnum, size = (each_device_batch,), device=device)

    genshape = (each_device_batch, 3, int(hyperparams_model["image_size"]), int(hyperparams_model["image_size"]))

    for _ in range(numloop):
        if params.ddim:
            if params.con_sam:
                generated = diffusion.ddim_sample_con(genshape, params, hyperparams_model["select"][1:], cemb = cemb_target)
            else:
                generated = diffusion.ddim_sample(genshape, params, target_loader, hyperparams_model["select"][1:], cemb = cemb_target)
        else:
            generated = diffusion.sample(genshape, cemb = cemb_target)

    '''
    save_path = os.path.join(params.path, f'samples_{params.sample_init}/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(params.genbatch):
        save_image(samples{i}, os.path.join(save_path, f'samples{i}.png'),normalize = True)
    '''

def main():
    # several hyperparameters for models
    parser = argparse.ArgumentParser(description='test for diffusion model')

    parser.add_argument('--genbatch', type=int, default=80, help='batch size for sampling process')
    parser.add_argument('--path', type=str, default='./results/regular/amazon_dslr/2024-11-20_15-23-47', help='path for loading models')
    parser.add_argument('--epoch', type=int, default=10000, help='epochs for loading models')
    parser.add_argument('--label', type=str, default='range', help='labels of generated images')
    parser.add_argument('--clsnum', type=int, default=10, help='num of label classes')
    parser.add_argument('--seed', type=int, default=111, help='random seed for training')
    parser.add_argument('--shape', type=int, default=64, help='shape of sampled images')
    parser.add_argument('--protodir', type=str, default='./result/sumlog/', help='proto summary addresses')
    parser.add_argument('--fid', type=lambda x:(str(x).lower() in ['true','1', 'yes']), default=False, help='generate samples used for quantative evaluation')
    parser.add_argument('--genum', type=int, default=1000, help='num of generated samples')
    parser.add_argument('--select', type=str, default='linear', help='selection stragies for DDIM')
    parser.add_argument('--name', default='office', type=str, help='datasets for sampling')
    parser.add_argument('--ddim', type=lambda x:(str(x).lower() in ['true','1', 'yes']), default='true', help='whether to use ddim') 
    parser.add_argument('--con_sam', type=lambda x:(str(x).lower() in ['true','1', 'yes']), default='true', help='whether to sample with confidence strategy') 
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--num_steps', type=int, default=50, help='sampling steps for DDIM')
    parser.add_argument('--sample_init', type=str, default='image', help='init sampling mode for DDIM')
    parser.add_argument('--dataset', default='dslr', type=str, help='dataset for sampling')
    parser.add_argument('--numworkers', type=int, default=0, help='num workers for sampling process')
    parser.add_argument('--gamma', type=float, default=0.8, help='confidence threahold for sampling exict')
    parser.add_argument('--eta', type=float, default=1.0, help='random noise or image for sampling (ddim or ddpm)')
    parser.add_argument('--time_step', type=list, default=[200, 400, 600, 800], help='time step of image')
    args = parser.parse_args()
    sample(args)

if __name__ == '__main__':
    main()
