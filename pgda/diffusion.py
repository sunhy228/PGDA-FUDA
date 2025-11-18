import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.distributed import get_rank
from embedding import ProtoEmbedding
import torchvision
from dataloaders import load_cifar10_sample, load_sample

class GaussianDiffusion(nn.Module):
    def __init__(self, dtype:torch.dtype, model, betas:np.ndarray, w:float, v:float, device:torch.device):
        super().__init__()
        self.dtype = dtype
        self.model = model.to(device)
        self.model.dtype = self.dtype
        self.betas = torch.tensor(betas,dtype=self.dtype,device=device)
        self.w = w
        self.v = v
        self.T = len(betas)
        self.device = device
        self.alphas = 1 - self.betas    # α
        self.alphas_cump = self.alphas.cumprod(dim=0) # Σα
        self.alphas_cump_sqrt = torch.sqrt(self.alphas_cump.to(device)) # sqrt(Σα)
        self.alphas_cump_1m_sqrt = torch.sqrt(1 - self.alphas_cump.to(device)) # sqrt(1-Σα)
        self.log_alphas = torch.log(self.alphas) # log(α)
        
        self.log_alphas_bar = torch.cumsum(self.log_alphas, dim = 0)
        self.alphas_bar = torch.exp(self.log_alphas_bar)    
        # self.alphas_bar = torch.cumprod(self.alphas, dim = 0)
        
        self.log_alphas_bar_prev = F.pad(self.log_alphas_bar[:-1],[1,0],'constant', 0)
        self.alphas_bar_prev = torch.exp(self.log_alphas_bar_prev)
        self.log_one_minus_alphas_bar_prev = torch.log(1.0 - self.alphas_bar_prev)
        # self.alphas_bar_prev = F.pad(self.alphas_bar[:-1],[1,0],'constant',1)

        # calculate parameters for q(x_t|x_{t-1})
        self.log_sqrt_alphas = 0.5 * self.log_alphas
        self.sqrt_alphas = torch.exp(self.log_sqrt_alphas)
        # self.sqrt_alphas = torch.sqrt(self.alphas)

        # calculate parameters for q(x_t|x_0)
        self.log_sqrt_alphas_bar = 0.5 * self.log_alphas_bar
        self.sqrt_alphas_bar = torch.exp(self.log_sqrt_alphas_bar)
        # self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.log_one_minus_alphas_bar = torch.log(1.0 - self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.exp(0.5 * self.log_one_minus_alphas_bar)
        
        # calculate parameters for q(x_{t-1}|x_t,x_0)
        # log calculation clipped because the \tilde{\beta} = 0 at the beginning
        self.tilde_betas = self.betas * torch.exp(self.log_one_minus_alphas_bar_prev - self.log_one_minus_alphas_bar)
        self.log_tilde_betas_clipped = torch.log(torch.cat((self.tilde_betas[1].view(-1), self.tilde_betas[1:]), 0))
        self.mu_coef_x0 = self.betas * torch.exp(0.5 * self.log_alphas_bar_prev - self.log_one_minus_alphas_bar)
        self.mu_coef_xt = torch.exp(0.5 * self.log_alphas + self.log_one_minus_alphas_bar_prev - self.log_one_minus_alphas_bar)
        self.vars = torch.cat((self.tilde_betas[1:2],self.betas[1:]), 0)
        self.coef1 = torch.exp(-self.log_sqrt_alphas)
        self.coef2 = self.coef1 * self.betas / self.sqrt_one_minus_alphas_bar
        # calculate parameters for predicted x_0
        self.sqrt_recip_alphas_bar = torch.exp(-self.log_sqrt_alphas_bar)
        # self.sqrt_recip_alphas_bar = torch.sqrt(1.0 / self.alphas_bar)
        self.sqrt_recipm1_alphas_bar = torch.exp(self.log_one_minus_alphas_bar - self.log_sqrt_alphas_bar)
        # self.sqrt_recipm1_alphas_bar = torch.sqrt(1.0 / self.alphas_bar - 1)
    @staticmethod
    def _extract(coef:torch.Tensor, t:torch.Tensor, x_shape:tuple) -> torch.Tensor:
        """
        input:

        coef : an array
        t : timestep
        x_shape : the shape of tensor x that has K dims(the value of first dim is batch size)

        output:

        a tensor of shape [batchsize,1,...] where the length has K dims.
        """
        assert t.shape[0] == x_shape[0]

        neo_shape = torch.ones_like(torch.tensor(x_shape))
        neo_shape[0] = x_shape[0]
        neo_shape = neo_shape.tolist()
        chosen = coef[t]
        chosen = chosen.to(t.device)
        return chosen.reshape(neo_shape)

    def q_mean_variance(self, x_0:torch.Tensor, t:torch.Tensor): # -> Tuple[torch.Tensor, torch.Tensor]:
        """
        calculate the parameters of q(x_t|x_0)
        """
        mean = self._extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0
        var = self._extract(1.0 - self.sqrt_alphas_bar, t, x_0.shape)
        return mean, var
    
    def q_sample(self, x_0:torch.Tensor, t:torch.Tensor): # -> Tuple[torch.Tensor, torch.Tensor]:
        """
        sample from q(x_t|x_0)
        """
        eps = torch.randn_like(x_0, requires_grad=False)
        return self._extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 \
            + self._extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * eps, eps
    
    def q_posterior_mean_variance(self, x_0:torch.Tensor, x_t:torch.Tensor, t:torch.Tensor): # -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        calculate the parameters of q(x_{t-1}|x_t,x_0)
        """
        posterior_mean = self._extract(self.mu_coef_x0, t, x_0.shape) * x_0 \
            + self._extract(self.mu_coef_xt, t, x_t.shape) * x_t
        posterior_var_max = self._extract(self.tilde_betas, t, x_t.shape)
        log_posterior_var_min = self._extract(self.log_tilde_betas_clipped, t, x_t.shape)
        log_posterior_var_max = self._extract(torch.log(self.betas), t, x_t.shape)
        log_posterior_var = self.v * log_posterior_var_max + (1 - self.v) * log_posterior_var_min
        neo_posterior_var = torch.exp(log_posterior_var)
        
        return posterior_mean, posterior_var_max, neo_posterior_var
    def p_mean_variance(self, x_t:torch.Tensor, t:torch.Tensor, **model_kwargs): # -> Tuple[torch.Tensor, torch.Tensor]:
        """
        calculate the parameters of p_{theta}(x_{t-1}|x_t)
        """
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        cemb_shape = model_kwargs['cemb'].shape
        pred_eps_cond = self.model(x_t, t, **model_kwargs)
        model_kwargs['cemb'] = torch.zeros(cemb_shape, device = self.device)
        pred_eps_uncond = self.model(x_t, t, **model_kwargs)
        pred_eps = (1 + self.w) * pred_eps_cond - self.w * pred_eps_uncond
        
        assert torch.isnan(x_t).int().sum() == 0, f"nan in tensor x_t when t = {t[0]}"
        assert torch.isnan(t).int().sum() == 0, f"nan in tensor t when t = {t[0]}"
        assert torch.isnan(pred_eps).int().sum() == 0, f"nan in tensor pred_eps when t = {t[0]}"
        p_mean = self._predict_xt_prev_mean_from_eps(x_t, t.type(dtype=torch.long), pred_eps)
        p_var = self._extract(self.vars, t.type(dtype=torch.long), x_t.shape)
        return p_mean, p_var

    def _predict_x0_from_eps(self, x_t:torch.Tensor, t:torch.Tensor, eps:torch.Tensor) -> torch.Tensor:
        return self._extract(coef = self.sqrt_recip_alphas_bar, t = t, x_shape = x_t.shape) \
            * x_t - self._extract(coef = self.sqrt_one_minus_alphas_bar, t = t, x_shape = x_t.shape) * eps

    def _predict_xt_prev_mean_from_eps(self, x_t:torch.Tensor, t:torch.Tensor, eps:torch.Tensor) -> torch.Tensor:
        return self._extract(coef = self.coef1, t = t, x_shape = x_t.shape) * x_t - \
            self._extract(coef = self.coef2, t = t, x_shape = x_t.shape) * eps

    def p_sample(self, x_t:torch.Tensor, t:torch.Tensor, **model_kwargs) -> torch.Tensor:
        """
        sample x_{t-1} from p_{theta}(x_{t-1}|x_t)
        """
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,), f"size of t is not batch size {B}"
        mean, var = self.p_mean_variance(x_t , t, **model_kwargs)
        assert torch.isnan(mean).int().sum() == 0, f"nan in tensor mean when t = {t[0]}"
        assert torch.isnan(var).int().sum() == 0, f"nan in tensor var when t = {t[0]}"
        noise = torch.randn_like(x_t)
        noise[t <= 0] = 0 
        return mean + torch.sqrt(var) * noise
    
    def sample(self, shape:tuple, **model_kwargs) -> torch.Tensor:
        """
        sample images from p_{theta}
        """
        local_rank = 0 #get_rank()
        if local_rank == 0:
            print('Start generating...')
        if model_kwargs == None:
            model_kwargs = {}
        x_t = torch.randn(shape, device = self.device)
        tlist = torch.ones([x_t.shape[0]], device = self.device) * self.T
        for _ in tqdm(range(self.T),dynamic_ncols=True, disable=(local_rank % torch.cuda.device_count() != 0)):
            tlist -= 1
            with torch.no_grad():
                x_t = self.p_sample(x_t, tlist, **model_kwargs)
        x_t = torch.clamp(x_t, -1, 1)
        if local_rank == 0:
            print('ending sampling process...')
        return x_t
    
    def ddim_p_mean_variance(self, x_t:torch.Tensor, t:torch.Tensor, prevt:torch.Tensor, eta:float, **model_kwargs) -> torch.Tensor:
        """
        calculate the parameters of p_{theta}(x_{t-1}|x_t)
        """
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        cemb_shape = model_kwargs['cemb'].shape
        pred_eps_cond = self.model(x_t, t, **model_kwargs)
        model_kwargs['cemb'] = torch.zeros(cemb_shape, device = self.device)
        pred_eps_uncond = self.model(x_t, t, **model_kwargs)
        pred_eps = (1 + self.w) * pred_eps_cond - self.w * pred_eps_uncond
        
        assert torch.isnan(x_t).int().sum() == 0, f"nan in tensor x_t when t = {t[0]}"
        assert torch.isnan(t).int().sum() == 0, f"nan in tensor t when t = {t[0]}"
        assert torch.isnan(pred_eps).int().sum() == 0, f"nan in tensor pred_eps when t = {t[0]}"

        alphas_bar_t = self._extract(coef = self.alphas_bar, t = t, x_shape = x_t.shape)
        alphas_bar_prev = self._extract(coef = self.alphas_bar_prev, t = prevt + 1, x_shape = x_t.shape)
        sigma = eta * torch.sqrt((1 - alphas_bar_prev) / (1 - alphas_bar_t) * (1 - alphas_bar_t / alphas_bar_prev))
        p_var = sigma ** 2
        coef_eps = 1 - alphas_bar_prev - p_var
        coef_eps[coef_eps < 0] = 0
        coef_eps = torch.sqrt(coef_eps)
        p_mean = torch.sqrt(alphas_bar_prev) * (x_t - torch.sqrt(1 - alphas_bar_t) * pred_eps) / torch.sqrt(alphas_bar_t) + \
            coef_eps * pred_eps
        return p_mean, p_var
    
    def ddim_p_sample(self, x_t:torch.Tensor, t:torch.Tensor, prevt:torch.Tensor, eta:float, **model_kwargs) -> torch.Tensor: 
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,), f"size of t is not batch size {B}"
        mean, var = self.ddim_p_mean_variance(x_t , t.type(dtype=torch.long), prevt.type(dtype=torch.long), eta, **model_kwargs)
        assert torch.isnan(mean).int().sum() == 0, f"nan in tensor mean when t = {t[0]}"
        assert torch.isnan(var).int().sum() == 0, f"nan in tensor var when t = {t[0]}"
        noise = torch.randn_like(x_t)
        noise[t <= 0] = 0 
        return mean + torch.sqrt(var) * noise
    
    def ddim_sample(self, shape:tuple, params:argparse.Namespace, dataloader = None, select:str = 'linear', **model_kwargs) -> torch.Tensor:
        eta = params.eta
        num_steps = params.num_steps
        local_rank = 0 #get_rank()
        if local_rank == 0:
            print('Start generating(ddim)...')
        if model_kwargs == None:
            model_kwargs = {}
        # a subsequence of range(0,1000)

        if params.sample_init == 'image':
            if dataloader is None:
                raise ValueError("dataloader is None!")
            else:
                images_list = []
            for i, (fea, lab) in enumerate(tqdm(dataloader, total= len(dataloader) - 1, desc=f'process of sampling {params.dataset}')): 
                for j, t in enumerate(params.time_step):
                    fea = fea.to(self.device)
                    torchvision.utils.save_image(fea, f'./test/results/regular/cifar10/2024-06-03_20-16-22/ddim_sample/original_{t}_{i}.png', normalize=True)
                    img_n = self.diffusion(fea, torch.ones(params.genbatch, device=self.device, dtype=torch.long) * t)
                    torchvision.utils.save_image(img_n, f'./test/results/regular/cifar10/2024-06-03_20-16-22/ddim_sample/noise_{t}_{i}.png', normalize=True)
                    img = self.multi_denoising_from_image(i, params, img_n, t, select, **model_kwargs)
                    images_list.append(img.cpu())
            return images_list
        
        elif params.sample_init == 'noise':
            x_t = torch.randn(shape, device = self.device)
            return self.multi_denoising_from_noise(params, x_t, select, **model_kwargs)
        else:
            raise NotImplementedError(f'Don\'t have initial generator for "sample_init"')
        
    def ddim_sample_con(self, shape: tuple, params: argparse.Namespace, select='linear', **model_kwargs) -> torch.Tensor:
        eta = params.eta
        num_steps = params.num_steps
        local_rank = 0 #get_rank()
        if local_rank == 0:
            print('Start generating(ddim)...')
        if model_kwargs == None:
            model_kwargs = {}
        # a subsequence of range(0,1000)

        if params.sample_init == 'image':
            dataloader = load_sample(params)
            images_list = []

            correct, total = 0, 0
            results = []

            Proto = ProtoEmbedding(params)
            Proto.agent.model.load_state_dict(torch.load('./temp/extractor.pth'), strict=True)
            Proto.agent.cls_head.load_state_dict(torch.load('./temp/head.pth'), strict=True)

            Proto.agent.model.eval()
            Proto.agent.cls_head.eval()

            batch = next(iter(dataloader))

            for i, (_, fea, lab) in enumerate(tqdm(dataloader, total=len(dataloader), desc=f'Adaptive sampling {params.dataset}')):
                fea = fea.to(self.device)
                lab = lab.to(self.device)
                total += fea.size(0)

                model_kwargs['cemb'] = torch.stack([model_kwargs['cemb'][key] for key in lab.cpu().numpy()]) # [batchsize, dim]

                accepted = torch.zeros(fea.size(0), dtype=torch.bool, device=self.device)
                preds = torch.zeros_like(lab)
                best_conf = torch.zeros(fea.size(0), device=self.device)
                best_pred = torch.zeros_like(lab)

                for t in range(50, 201, 10):
                    img_n = self.diffusion(fea, torch.ones(fea.size(0), device=self.device, dtype=torch.long) * t)
                    img_recon = self.multi_denoising_from_image(i, params, img_n, t, select, **model_kwargs)

                    # predict
                    with torch.no_grad():
                        logits = Proto.agent.cls_head(F.normalize(Proto.agent.model(img_recon), dim=1))
                        probs = torch.softmax(logits, dim=1)
                        conf, pred = probs.max(dim=1)

                    # update
                    mask_better = conf > best_conf
                    best_conf[mask_better] = conf[mask_better]
                    best_pred[mask_better] = pred[mask_better]

                    # select
                    mask_accept = conf >= params.gamma
                    if mask_accept.any():
                        preds[mask_accept] = pred[mask_accept]
                        accepted[mask_accept] = True

                    # early exict
                    if accepted.all():
                        break

                # deal unconfident samples
                final_pred = torch.where(accepted, preds, best_pred)
                results.append((final_pred.cpu(), lab.cpu()))

                correct += (final_pred == lab).sum().item()

            accuracy = correct / total
            print(f'Final classification accuracy: {accuracy:.4f}')

            return results
        
        elif params.sample_init == 'noise':
            x_t = torch.randn(shape, device = self.device)
            return self.multi_denoising_from_noise(params, x_t, select, **model_kwargs)
        else:
            raise NotImplementedError(f'Don\'t have initial generator for "sample_init"')
    
    def multi_denoising_from_image(self, iter:int, params:argparse.Namespace, x_t:torch.Tensor, t_end, select:str = 'linear', **model_kwargs) -> torch.Tensor:
        eta = params.eta
        num_steps = params.num_steps

        if model_kwargs == None:
            model_kwargs = {}
        if select == 'linear':
            tseq = list(np.linspace(0, t_end-1, num_steps).astype(int))
        elif select == 'quadratic':
            tseq = list((np.linspace(0, np.sqrt(t_end), num_steps-1)**2).astype(int))
            tseq.insert(0, 0)
            tseq[-1] = self.T - 1
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{select}"')

        tlist = torch.zeros([x_t.shape[0]], device = self.device)
        for i in tqdm(range(num_steps),dynamic_ncols=True):
            with torch.no_grad():
                tlist = tlist * 0 + tseq[-1-i]
                if i != num_steps - 1:
                    prevt = torch.ones_like(tlist, device = self.device) * tseq[-2-i]
                else:
                    prevt = - torch.ones_like(tlist, device = self.device) 
                x_t = self.ddim_p_sample(x_t, tlist, prevt, eta, **model_kwargs)
                # save x_t as image 
                if i == 0 or i == num_steps - 1:   
                    torchvision.utils.save_image(x_t, f'./results/regular/amazon_dslr/2024-11-20_15-23-47/ddim_sample/img_iter{iter}_{t_end}_{tlist[0]}.png', normalize=True) 
                torch.cuda.empty_cache()
        x_t = torch.clamp(x_t, -1, 1)

        return x_t
    
    def diffusion(self, img, t_end, noise=None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(img)
        else:
            noise = noise.to(img.device)

        # √α_t 
        alpha_sqrt = self.alphas_cump_sqrt.index_select(0, t_end).view(-1, 1, 1, 1) # self.alphas_cump_sqrt.shape = [1000]
        #√(1-α_t) 
        alpha_1m_sqrt = self.alphas_cump_1m_sqrt.index_select(0, t_end).view(-1, 1, 1, 1)

        img_n = img * alpha_sqrt + noise * alpha_1m_sqrt

        return img_n

    def multi_denoising_from_noise(self, params:argparse.Namespace, x_t:torch.Tensor, select:str = 'linear', **model_kwargs) -> torch.Tensor:
        eta = params.eta
        num_steps = params.num_steps
        local_rank = 0 #get_rank()
        if local_rank == 0:
            print('Start generating(ddim)...')
        if model_kwargs == None:
            model_kwargs = {}
        # a subsequence of range(0,1000)
        if select == 'linear':
            tseq = list(np.linspace(0, self.T-1, num_steps).astype(int))
        elif select == 'quadratic':
            tseq = list((np.linspace(0, np.sqrt(self.T), num_steps-1)**2).astype(int))
            tseq.insert(0, 0)
            tseq[-1] = self.T - 1
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{select}"')

        tlist = torch.zeros([x_t.shape[0]], device = self.device)
        for i in tqdm(range(num_steps), dynamic_ncols=True):
            with torch.no_grad():
                tlist = tlist * 0 + tseq[-1-i]
                if i != num_steps - 1:
                    prevt = torch.ones_like(tlist, device = self.device) * tseq[-2-i]
                else:
                    prevt = - torch.ones_like(tlist, device = self.device) 
                x_t = self.ddim_p_sample(x_t, tlist, prevt, eta, **model_kwargs)
                # save x_t as image 
                if tlist[0] in [999.0, 754.0, 509.0, 265.0, 0.0]:   
                    torchvision.utils.save_image(x_t, f'./test/results/regular/cifar10/2024-06-03_20-16-22/ddim_sample/{eta}_{tlist[0]}.png', normalize=True) 
                torch.cuda.empty_cache()
        x_t = torch.clamp(x_t, -1, 1)
        if local_rank == 0:
            print('ending sampling process(ddim)...')
        return x_t

    
    def trainloss(self, x_0:torch.Tensor, cemb:torch.Tensor, file=None) -> torch.Tensor:
        """
        calculate the loss of denoising diffusion probabilistic model
        """        
        t_ = torch.randint(50, size = (x_0.shape[0]//10,), device=self.device)
        t = torch.randint(self.T, size = (x_0.shape[0] - t_.shape[0],), device=self.device)
        t = torch.cat([t, t_])

        x_t, eps = self.q_sample(x_0, t)
        pred_eps = self.model(x_t, t, cemb)
        loss = F.mse_loss(pred_eps, eps, reduction='mean')
        file.write(f'DDPM Loss: {loss} \n')
               
        return loss
    
