from logging import config
import os
import random
import argparse
from tkinter import N
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms


import clip
from utils import *

import time


from einops import rearrange


from collections import OrderedDict
from typing import Tuple, Union

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

from clip import clip

from dataloader.data_utils import *



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x










def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')


    parser.add_argument('-dataset', type=str, default='cifar100',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])

    parser.add_argument('-start_session', type=int, default=0)

    parser.add_argument('-dataroot', type=str, default= '/home/sunhongbo/DATA/FSCIL_data/')

    parser.add_argument('-batch_size_base', type=int, default=256)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=256)


    args = parser.parse_args()

    return args



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)




class ResidualAttentionBlock_visual(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model)),
            ("gelu", QuickGELU()),
            #("c_proj", nn.Linear(d_model * 2, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, mask = None):

        if mask is not None:
            attn_mask = mask.to(dtype=x.dtype, device=x.device)
            output, weights = self.attn(x, x, x, need_weights=True, average_attn_weights = False, attn_mask = attn_mask)
            return output, weights
        else:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
            output, weights = self.attn(x, x, x, need_weights=True, average_attn_weights = False, attn_mask = self.attn_mask)
            return output, weights


    def forward(self, x: torch.Tensor, mask = None):
        
        output, weights = self.attention(self.ln_1(x), mask)
        x = x + output

        x = x + self.mlp(self.ln_2(x))
        return x, weights




class new_net(nn.Module):
    def __init__(self):
        super(new_net, self).__init__()


        self.new_linear = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(768, 768//6, bias = False).half().cuda()),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(768//6, 768, bias = False).half().cuda()),
        ]))
    

    def forward(self, x):
        
        x = x

        return x






# prompt network
class prompt(nn.Module):
    def __init__(self, cache_keys=0, clip_model=0, clip_weights=0, dataset_name = None):
        super(prompt, self).__init__()

        self.cache_keys = cache_keys

        self.relu = nn.LeakyReLU(0.2)


        self.linear_map_visual = nn.Linear(clip_weights.size(0), clip_weights.size(0), bias=True).to(clip_model.dtype).cuda()
        self.linear_map_weight = nn.Linear(clip_weights.size(0), clip_weights.size(0), bias=True).to(clip_model.dtype).cuda()

        self.prompt_cls = nn.Parameter(torch.zeros(768).cuda()).cuda()

        self.new_trans = new_net()


        self.n_ctx = 1
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        print("Initializing a generic context")


        if dataset_name == 'cub200':
            ctx_vectors = torch.empty(200, self.n_ctx, ctx_dim, dtype=self.dtype).cuda()
        else:
            ctx_vectors = torch.empty(100, self.n_ctx, ctx_dim, dtype=self.dtype).cuda()

        
        nn.init.normal_(ctx_vectors, std=0.02)

        self.prompt_prefix = " ".join(["X"] * self.n_ctx)

        self.ctx = nn.Parameter(ctx_vectors).to(clip_model.dtype).cuda()  # to be optimized



        self.clip_model = clip_model

        self.text_encoder = TextEncoder(clip_model)



        self.fc1 = nn.Linear(512, 60, bias=True).to(clip_model.dtype).cuda()
        self.fc2 = nn.Linear(512, 5, bias=True).to(clip_model.dtype).cuda()

        self.dataset_name = dataset_name


        self.meta_net_con = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(1024, 512).to(clip_model.dtype).cuda()),
            ("drop", nn.Dropout(p=0.5)),
            ("linear2", nn.Linear(512, 512).to(clip_model.dtype).cuda())
        ]))


        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(512, 1024).to(clip_model.dtype).cuda()),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(1024, 512).to(clip_model.dtype).cuda())
        ]))


        self.sigmoid = nn.Sigmoid()


    def forward(self, x_ori, clip_weights_ori, keys=None, obj_branch=0, classnames=None, session_flag=0, eval_flag=0, map_keys=None, map_net=None):
        
        y = x_ori.clone()


        glo_fea = x_ori.clone()
        glo_fea = glo_fea.squeeze(1)

        if session_flag == 0:
            glo_logits = self.fc1(glo_fea)
        else:
            glo_logits = self.fc2(glo_fea)


        keys1 = keys.clone()
        
        cache_keys_new = rearrange(keys, 'a1 a2 -> a2 a1')

        cache_keys_new_residual = self.relu(self.linear_map_visual(cache_keys_new))
        cache_keys_new_residual = cache_keys_new_residual / cache_keys_new_residual.norm(dim=-1, keepdim=True)
        cache_keys_new = cache_keys_new + cache_keys_new_residual
        
        
        x_new_residual = self.relu(self.linear_map_visual(x_ori))
        x_new_residual = x_new_residual / x_new_residual.norm(dim=-1, keepdim=True)
        x_new= x_ori + x_new_residual


        cache_keys_new = cache_keys_new / cache_keys_new.norm(dim=-1, keepdim=True)
        x_new = x_new / x_new.norm(dim=-1, keepdim=True)

        cache_keys_new = rearrange(cache_keys_new, 'a1 a2 -> a2 a1')

        x = x_new @ cache_keys_new


        if obj_branch == -1:
            clip_weights = rearrange(clip_weights_ori, 'a1 a2 -> a2 a1')
            clip_weights_new = self.relu(self.linear_map_weight(clip_weights))
            clip_weights_new = 0.05*clip_weights_new / clip_weights_new.norm(dim=-1, keepdim=True)  + clip_weights
            clip_weights_new = rearrange(clip_weights_new, 'a1 a2 -> a2 a1')
        else:
    
            keys1_con = rearrange(map_keys, 'a1 a2 -> a2 a1')

            con = self.meta_net_con(keys1_con)

            glo = rearrange(keys1.clone(), 'a1 a2 -> a2 a1')

            gate1 = torch.mul(con, glo)
            gate1 = self.sigmoid(gate1)

            glo = torch.mul(gate1, glo) + con

            text_prompt_residual = self.meta_net(glo)

            classnames = [name.replace("_", " ") for name in classnames]
     
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]


            tokenized_prompts = torch.cat([clip.tokenize(p).cuda() for p in prompts])
            with torch.no_grad():
                embedding = self.clip_model.token_embedding(tokenized_prompts).to(self.clip_model.dtype).cuda()


            n_cls = len(classnames)


            if self.dataset_name == 'cub200':

                ctx = self.ctx[ : 100+(session_flag-1)*10+10, :, :] + text_prompt_residual.unsqueeze(1)

            else:

                ctx = self.ctx[ : 60+(session_flag-1)*5+5, :, :] + text_prompt_residual.unsqueeze(1)


            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)


            prefix = embedding[:, :1, :]
            suffix = embedding[:, 1 + self.n_ctx :, :]

            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )


            clip_weights_new = self.text_encoder(prompts, tokenized_prompts)
            clip_weights_new = clip_weights_new / clip_weights_new.norm(dim=-1, keepdim=True)
            clip_weights_new = rearrange(clip_weights_new, 'a1 a2 -> a2 a1')


        
        return x, y, clip_weights_new, glo_logits








# prompt network of object
class prompt_obj(nn.Module):
    def __init__(self, cache_keys=0, clip_model=0, clip_weights=0, dataset_name = None) :
        super(prompt_obj, self).__init__()

        self.cache_keys = cache_keys

        self.relu = nn.LeakyReLU(0.2)

        self.linear_map_visual = nn.Linear(clip_weights.size(0), clip_weights.size(0), bias=True).to(clip_model.dtype).cuda()
        self.linear_map_weight = nn.Linear(clip_weights.size(0), clip_weights.size(0), bias=True).to(clip_model.dtype).cuda()

        self.prompt_cls = nn.Parameter(torch.zeros(768).cuda()).cuda()

        self.new_trans = new_net()



        self.n_ctx = 1
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        print("Initializing a generic context")


        if dataset_name == 'cub200':
            ctx_vectors = torch.empty(200, self.n_ctx, ctx_dim, dtype=self.dtype).cuda()
        else:
            ctx_vectors = torch.empty(100, self.n_ctx, ctx_dim, dtype=self.dtype).cuda()



        nn.init.normal_(ctx_vectors, std=0.02)

        self.prompt_prefix = " ".join(["X"] * self.n_ctx)

        self.ctx = nn.Parameter(ctx_vectors).to(clip_model.dtype).cuda()  # to be optimized 


        self.clip_model = clip_model

        self.text_encoder = TextEncoder(clip_model)


        self.fc1 = nn.Linear(512, 60, bias=True).to(clip_model.dtype).cuda()
        self.fc2 = nn.Linear(512, 5, bias=True).to(clip_model.dtype).cuda()

        self.dataset_name = dataset_name


        self.meta_net_con = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(1024, 512).to(clip_model.dtype).cuda()),
            ("drop", nn.Dropout(p=0.5)),
            ("linear2", nn.Linear(512, 512).to(clip_model.dtype).cuda())
        ]))


        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(512, 1024).to(clip_model.dtype).cuda()),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(1024, 512).to(clip_model.dtype).cuda())
        ]))

        self.sigmoid = nn.Sigmoid()





    def forward(self, x_ori, clip_weights_ori, keys=None, obj_branch=0, classnames=None, session_flag=0, eval_flag=0, map_keys=None, map_net=None):
        
        y = x_ori.clone()

        obj_fea = x_ori.clone()
        obj_fea = obj_fea.squeeze(1)

        if session_flag == 0:
            obj_logits = self.fc1(obj_fea)
        else:
            obj_logits = self.fc2(obj_fea)



        keys1 = keys.clone()
        
        cache_keys_new = rearrange(keys, 'a1 a2 a3-> a3 a2 a1')

        cache_keys_new_residual = self.relu(self.linear_map_visual(cache_keys_new))
        cache_keys_new_residual = cache_keys_new_residual / cache_keys_new_residual.norm(dim=-1, keepdim=True)
        cache_keys_new = cache_keys_new + cache_keys_new_residual
        
        
        x_new_residual = self.relu(self.linear_map_visual(x_ori))
        x_new_residual = x_new_residual / x_new_residual.norm(dim=-1, keepdim=True)
        x_new= x_ori + x_new_residual


        cache_keys_new = cache_keys_new / cache_keys_new.norm(dim=-1, keepdim=True)
        x_new = x_new / x_new.norm(dim=-1, keepdim=True)

        cache_keys_new = rearrange(cache_keys_new, 'a1 a2 a3 -> a3 a2 a1')


        x = x_new[:,0,:] @ cache_keys_new[:,0,:]


        if obj_branch == -1:
            clip_weights = rearrange(clip_weights_ori, 'a1 a2 -> a2 a1')
            clip_weights_new = self.relu(self.linear_map_weight(clip_weights))
            clip_weights_new = 0.05*clip_weights_new / clip_weights_new.norm(dim=-1, keepdim=True)  + clip_weights
            clip_weights_new = rearrange(clip_weights_new, 'a1 a2 -> a2 a1')
        else:
    
            keys1 = keys1[:,0,:]


            keys1_con = rearrange(map_keys, 'a1 a2 -> a2 a1')

            con = self.meta_net_con(keys1_con)

            obj = rearrange(keys1.clone(), 'a1 a2 -> a2 a1')

            gate1 = torch.mul(con, obj)
            gate1 = self.sigmoid(gate1)

            obj = torch.mul(gate1, obj) + con

            text_prompt_residual = self.meta_net(obj)


            text_prompt_residual = text_prompt_residual.unsqueeze(1)


            classnames = [name.replace("_", " ") for name in classnames]
          
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]


            tokenized_prompts = torch.cat([clip.tokenize(p).cuda() for p in prompts])

            with torch.no_grad():
                embedding = self.clip_model.token_embedding(tokenized_prompts).to(self.clip_model.dtype).cuda()

            n_cls = len(classnames)


            if self.dataset_name == 'cub200':

                ctx = self.ctx[ : 100+(session_flag-1)*10+10, :, :] + text_prompt_residual

            else:

                ctx = self.ctx[ : 60+(session_flag-1)*5+5, :, :] + text_prompt_residual


            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)



            prefix = embedding[:, :1, :]
            suffix = embedding[:, 1 + self.n_ctx :, :]

            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )


            clip_weights_new = self.text_encoder(prompts, tokenized_prompts)
            clip_weights_new = clip_weights_new / clip_weights_new.norm(dim=-1, keepdim=True)
            clip_weights_new = rearrange(clip_weights_new, 'a1 a2 -> a2 a1')



        return x, y, clip_weights_new, obj_logits



class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None


    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x




def run_fg_cpl(cfg, cache_keys, cache_values, test_loader, clip_weights, clip_model, train_loader_F, cache_keys_obj, key=None, value=None, obj=None, flag=0, test_clip_weights=None,
                      total_classnames=None, temp_classnames=None, session_flag=0, dataset_name = None):
    

    file_record = './output/' + cfg['dataset'] + '-' + 'acc_record.txt'

    
    if dataset_name == 'cub200':
        factor = 0.5 
    elif dataset_name == 'cifar100':
        factor = 2.0 
    elif dataset_name == 'mini_imagenet':
        factor = 0.5 
    





    if flag == 0:
        adapter = prompt(cache_keys=cache_keys, clip_model=clip_model, clip_weights=clip_weights, dataset_name = dataset_name)
        adapter_obj = prompt_obj(cache_keys=cache_keys_obj, clip_model=clip_model, clip_weights=clip_weights, dataset_name = dataset_name)

        if dataset_name == 'cub200':
            cfg['train_epoch'] = 50
        else:
            cfg['train_epoch'] = 30


    else:
        adapter = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
        adapter_obj = torch.load(cfg['cache_dir'] + "/best_F_obj_" + str(cfg['shots']) + "shots.pt")
        cfg['train_epoch'] = 10




    if flag == 1:
        lr_new = cfg['lr']/10


        optimizer = torch.optim.AdamW([{'params' : adapter.linear_map_visual.parameters()}, 
                                       {'params' : adapter.linear_map_weight.parameters()}, 

                                       {'params' : adapter.prompt_cls}, 
                                       {'params' : adapter.new_trans.parameters()},                                     

                                
                                       {'params' : adapter.ctx, 'lr': lr_new * 10}, 

                                       {'params' : adapter.meta_net.parameters(), 'lr': lr_new * 1},    
                                       {'params' : adapter.meta_net_con.parameters(), 'lr': lr_new * 1},   


                                       {'params' : adapter.fc1.parameters(), 'lr': lr_new * 1},     
                                       {'params' : adapter.fc2.parameters(), 'lr': lr_new * 1},    

                                        ],
       
                                      lr=lr_new, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
        

        optimizer_obj = torch.optim.AdamW([{'params' : adapter_obj.linear_map_visual.parameters()}, 
                                       {'params' : adapter_obj.linear_map_weight.parameters()}, 

                                       {'params' : adapter_obj.prompt_cls}, 
                                       {'params' : adapter_obj.new_trans.parameters()},                                     

                                       {'params' : adapter_obj.ctx, 'lr': lr_new * 10},

                                       {'params' : adapter_obj.meta_net.parameters(), 'lr': lr_new * 1},     
                                       {'params' : adapter_obj.meta_net_con.parameters(), 'lr': lr_new * 1},   



                                       {'params' : adapter_obj.fc1.parameters(), 'lr': lr_new * 1},     
                                       {'params' : adapter_obj.fc2.parameters(), 'lr': lr_new * 1},    

                                        ],
       
                                      lr=lr_new, eps=1e-4)
        scheduler_obj = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_obj, cfg['train_epoch'] * len(train_loader_F))
        


    else:
        lr_new = cfg['lr']




        optimizer = torch.optim.AdamW([{'params' : adapter.linear_map_visual.parameters()}, 
                                       {'params' : adapter.linear_map_weight.parameters()}, 

                                       {'params' : adapter.prompt_cls}, 
                                       {'params' : adapter.new_trans.parameters()},                                     
                    
                                       {'params' : adapter.ctx}, 

                                       {'params' : adapter.meta_net.parameters(), 'lr': lr_new * 1},     
                                       {'params' : adapter.meta_net_con.parameters(), 'lr': lr_new * 1},  


                                       {'params' : adapter.fc1.parameters(), 'lr': lr_new * 1},     
                                       {'params' : adapter.fc2.parameters(), 'lr': lr_new * 1},    
                                       
                                        ],
       
                                      lr=lr_new, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
        

        optimizer_obj = torch.optim.AdamW([{'params' : adapter_obj.linear_map_visual.parameters()}, 
                                       {'params' : adapter_obj.linear_map_weight.parameters()}, 

                                       {'params' : adapter_obj.prompt_cls}, 
                                       {'params' : adapter_obj.new_trans.parameters()},                                     

                                       {'params' : adapter_obj.ctx}, 

                                       {'params' : adapter_obj.meta_net.parameters(), 'lr': lr_new * 1},     
                                       {'params' : adapter_obj.meta_net_con.parameters(), 'lr': lr_new * 1},     



                                       {'params' : adapter_obj.fc1.parameters(), 'lr': lr_new * 1},     
                                       {'params' : adapter_obj.fc2.parameters(), 'lr': lr_new * 1},    

                                        ],
       
                                      lr=lr_new, eps=1e-4)
        scheduler_obj = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_obj, cfg['train_epoch'] * len(train_loader_F))
        



    beta = 1
    best_acc, best_epoch = 0.0, 0
    


    alpha = 1/0.03  


    glo_open = 1
    obj_open = 1



    factor_clip = 1 

    if dataset_name == 'cub200':
        factor_cache = 1.5
    elif dataset_name == 'cifar100':
        factor_cache = 1.0
    elif dataset_name == 'mini_imagenet':
        factor_cache = 0.5
    



    if flag == 0:
        te_clip_weigts = clip_weights
        te_values = cache_values
        te_keys = cache_keys
        te_keys_obj = cache_keys_obj
        
        te_classnames = temp_classnames

    else:
        te_clip_weigts = torch.cat((test_clip_weights, clip_weights), dim=-1)


        te_values = torch.zeros((value.size(0)+cache_values.size(0)), cache_values.size(1)).cuda()


        if dataset_name == 'cub200':
            te_values [:value.size(0), :cache_values.size(1)-10] = value.clone()
        else:
            te_values [:value.size(0), :cache_values.size(1)-5] = value.clone()



        te_values [value.size(0):, :] = cache_values.clone()
        te_values =  te_values.half()

        te_keys = torch.cat((key, cache_keys), dim=-1)
        te_keys_obj = torch.cat((obj, cache_keys_obj), dim=-1)

        te_classnames = total_classnames + temp_classnames



    map_keys =  torch.cat([te_keys, te_keys_obj[:,0,:]])









    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        adapter_obj.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            
  
            image_features0, image_features0_obj = clip_model.encode_image(images, adapter.prompt_cls, adapter.new_trans)
            image_features = image_features0/image_features0.norm(dim=-1, keepdim=True)
            image_features_obj = image_features0_obj/image_features0_obj.norm(dim=-1, keepdim=True)


            affinity, image_features_new, clip_weights_new, glo_logits_temp = adapter(image_features, clip_weights, keys = te_keys, classnames = te_classnames, session_flag=session_flag, 
                                                                     map_keys=map_keys, map_net=adapter.meta_net) 

            cache_logits1 = ((-1) * (beta - beta * affinity)).exp() @ te_values

            cache_logits1 = cache_logits1 * alpha

            clip_logits1_temp = 100. * image_features_new @ clip_weights_new



            clip_logits1 = clip_logits1_temp.clone()

            tip_logits1 = factor_clip * F.softmax(clip_logits1, dim=-1) + factor_cache * F.softmax(cache_logits1, dim=-1)


            loss1 = factor_clip * F.cross_entropy(clip_logits1, target) + factor_cache * F.cross_entropy(cache_logits1, target)



            affinity, image_features_new, clip_weights_new, obj_logits_temp = adapter_obj(image_features_obj, clip_weights, keys = te_keys_obj, classnames = te_classnames, obj_branch=1, session_flag=session_flag,
                                                                        map_keys=map_keys, map_net=adapter.meta_net) 

            cache_logits2 = ((-1) * (beta - beta * affinity)).exp() @ te_values

            cache_logits2 = cache_logits2* alpha      


            clip_logits2_temp =  100. * image_features_new[:,0,:] @ clip_weights_new



            clip_logits2 = clip_logits2_temp.clone()  

            tip_logits2 = factor_clip * F.softmax(clip_logits2, dim=-1) + factor_cache * F.softmax(cache_logits2, dim=-1)


            loss2 = factor_clip * F.cross_entropy(clip_logits2, target) + factor_cache * F.cross_entropy(cache_logits2, target)


            loss = glo_open*loss1 + obj_open*factor*loss2


            tip_logits = glo_open*tip_logits1 + obj_open*factor*tip_logits2




            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            optimizer_obj.zero_grad()

            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            optimizer_obj.step()
            scheduler_obj.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))



        if train_idx==5 or train_idx==10 or train_idx%10== 0 or train_idx > cfg['train_epoch'] - 5:

            # Eval
            adapter.eval()
            adapter_obj.eval()



            test_features, test_labels, test_features_obj = pre_load_features(cfg, "test", clip_model, test_loader, adapter.prompt_cls, adapter.new_trans)
    
            affinity, image_features_new, clip_weights_new, glo_logits = adapter(test_features, te_clip_weigts, keys = te_keys, classnames = te_classnames, session_flag=session_flag, eval_flag=1, 
                                                                     map_keys=map_keys, map_net=adapter.meta_net)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ te_values
            cache_logits = alpha * cache_logits

            clip_logits = 100. * image_features_new @ clip_weights_new

            tip_logits1 = factor_clip * F.softmax(clip_logits, dim=-1) + factor_cache * F.softmax(cache_logits, dim=-1)










            affinity, image_features_new, clip_weights_new, obj_logits = adapter_obj(test_features_obj, te_clip_weigts, keys = te_keys_obj, classnames = te_classnames, obj_branch=1, session_flag=session_flag, eval_flag=1, map_keys=map_keys, map_net=adapter.meta_net)

            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ te_values
            cache_logits = alpha * cache_logits
        
            clip_logits = 100. * image_features_new[:,0,:] @ clip_weights_new

            tip_logits2 = factor_clip * F.softmax(clip_logits, dim=-1) + factor_cache * F.softmax(cache_logits, dim=-1)



            tip_logits = glo_open*tip_logits1 + obj_open*factor*tip_logits2



            acc = cls_acc(tip_logits, test_labels)

            print("\n****FineFMPL's test accuracy: {:.2f}. ****\n".format(acc))


            if train_idx == cfg['train_epoch'] - 1:
                best_acc = acc
                best_epoch = train_idx
                torch.save(adapter, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
                torch.save(adapter_obj, cfg['cache_dir'] + "/best_F_obj_" + str(cfg['shots']) + "shots.pt")


    print(f"**** After fine-tuning, FineFMPL's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    with open(file_record, 'a') as file:
        file.write("\n\n\n----------------------------------------------------------\n")
        file.write(f"**** After fine-tuning, FineFMPL's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")






cub200_class = ['Black footed Albatross', 'Laysan Albatross', 'Sooty Albatross', 'Groove billed Ani', 'Crested Auklet', 'Least Auklet', 'Parakeet Auklet', 'Rhinoceros Auklet', 
             'Brewer Blackbird', 'Red winged Blackbird', 'Rusty Blackbird', 'Yellow headed Blackbird', 'Bobolink', 'Indigo Bunting', 'Lazuli Bunting', 'Painted Bunting', 
             'Cardinal', 'Spotted Catbird', 'Gray Catbird', 'Yellow breasted Chat', 'Eastern Towhee', 'Chuck will Widow', 'Brandt Cormorant', 'Red faced Cormorant', 'Pelagic Cormorant', 
             'Bronzed Cowbird', 'Shiny Cowbird', 'Brown Creeper', 'American Crow', 'Fish Crow', 'Black billed Cuckoo', 'Mangrove Cuckoo', 'Yellow billed Cuckoo', 'Gray crowned Rosy Finch', 
             'Purple Finch', 'Northern Flicker', 'Acadian Flycatcher', 'Great Crested Flycatcher', 'Least Flycatcher', 'Olive sided Flycatcher', 'Scissor tailed Flycatcher', 'Vermilion Flycatcher', 
             'Yellow bellied Flycatcher', 'Frigatebird', 'Northern Fulmar', 'Gadwall', 'American Goldfinch', 'European Goldfinch', 'Boat tailed Grackle', 'Eared Grebe', 'Horned Grebe', 'Pied billed Grebe',
               'Western Grebe', 'Blue Grosbeak', 'Evening Grosbeak', 'Pine Grosbeak', 'Rose breasted Grosbeak', 'Pigeon Guillemot', 'California Gull', 'Glaucous winged Gull', 'Heermann Gull', 'Herring Gull', 
               'Ivory Gull', 'Ring billed Gull', 'Slaty backed Gull', 'Western Gull', 'Anna Hummingbird', 'Ruby throated Hummingbird', 'Rufous Hummingbird', 'Green Violetear', 'Long tailed Jaeger', 
               'Pomarine Jaeger', 'Blue Jay', 'Florida Jay', 'Green Jay', 'Dark eyed Junco', 'Tropical Kingbird', 'Gray Kingbird', 'Belted Kingfisher', 'Green Kingfisher', 'Pied Kingfisher', 
               'Ringed Kingfisher', 'White breasted Kingfisher', 'Red legged Kittiwake', 'Horned Lark', 'Pacific Loon', 'Mallard', 'Western Meadowlark', 'Hooded Merganser', 'Red breasted Merganser', 
               'Mockingbird', 'Nighthawk', 'Clark Nutcracker', 'White breasted Nuthatch', 'Baltimore Oriole', 'Hooded Oriole', 'Orchard Oriole', 'Scott Oriole', 'Ovenbird', 'Brown Pelican', 'White Pelican', 
               'Western Wood Pewee', 'Sayornis', 'American Pipit', 'Whip poor Will', 'Horned Puffin', 'Common Raven', 'White necked Raven', 'American Redstart', 'Geococcyx', 'Loggerhead Shrike', 
               'Great Grey Shrike', 'Baird Sparrow', 'Black throated Sparrow', 'Brewer Sparrow', 'Chipping Sparrow', 'Clay colored Sparrow', 'House Sparrow', 'Field Sparrow', 'Fox Sparrow', 
               'Grasshopper Sparrow', 'Harris Sparrow', 'Henslow Sparrow', 'Le Conte Sparrow', 'Lincoln Sparrow', 'Nelson Sharp tailed Sparrow', 'Savannah Sparrow', 'Seaside Sparrow', 'Song Sparrow', 
               'Tree Sparrow', 'Vesper Sparrow', 'White crowned Sparrow', 'White throated Sparrow', 'Cape Glossy Starling', 'Bank Swallow', 'Barn Swallow', 'Cliff Swallow', 'Tree Swallow', 'Scarlet Tanager', 
               'Summer Tanager', 'Artic Tern', 'Black Tern', 'Caspian Tern', 'Common Tern', 'Elegant Tern', 'Forsters Tern', 'Least Tern', 'Green tailed Towhee', 'Brown Thrasher', 'Sage Thrasher', 
               'Black capped Vireo', 'Blue headed Vireo', 'Philadelphia Vireo', 'Red eyed Vireo', 'Warbling Vireo', 'White eyed Vireo', 'Yellow throated Vireo', 'Bay breasted Warbler', 'Black and white Warbler', 
               'Black throated Blue Warbler', 'Blue winged Warbler', 'Canada Warbler', 'Cape May Warbler', 'Cerulean Warbler', 'Chestnut sided Warbler', 'Golden winged Warbler', 'Hooded Warbler', 
               'Kentucky Warbler', 'Magnolia Warbler', 'Mourning Warbler', 'Myrtle Warbler', 'Nashville Warbler', 'Orange crowned Warbler', 'Palm Warbler', 'Pine Warbler', 'Prairie Warbler', 
               'Prothonotary Warbler', 'Swainson Warbler', 'Tennessee Warbler', 'Wilson Warbler', 'Worm eating Warbler', 'Yellow Warbler', 'Northern Waterthrush', 'Louisiana Waterthrush', 
               'Bohemian Waxwing', 'Cedar Waxwing', 'American Three toed Woodpecker', 'Pileated Woodpecker', 'Red bellied Woodpecker', 'Red cockaded Woodpecker', 'Red headed Woodpecker',
                'Downy Woodpecker', 'Bewick Wren', 'Cactus Wren', 'Carolina Wren', 'House Wren', 'Marsh Wren', 'Rock Wren', 'Winter Wren', 'Common Yellowthroat']



mini_imagenet_class =['house finch', 'American robin', 'triceratops', 'green mamba', 'harvestman', 'toucan', 'goose', 'jellyfish', 'nematode', 'red king crab', 'dugong', 'Treeing Walker Coonhound', 'Ibizan Hound', 'Saluki', 'Golden Retriever', 'Gordon Setter', 'Komondor', 'Boxer', 'Tibetan Mastiff', 'French Bulldog', 'Alaskan Malamute', 'Dalmatian', 'Newfoundland dog', 'Miniature Poodle', 'Alaskan tundra wolf', 'African wild dog', 'Arctic fox', 'lion', 'meerkat', 'ladybug', 'rhinoceros beetle', 'ant', 'black-footed ferret', 'three-toed sloth', 'rock beauty fish', 'aircraft carrier', 'trash can', 'barrel', 'beer bottle', 'bookstore', 'cannon', 'carousel', 'cardboard box / carton', 'catamaran', 'bell or wind chime', 'clogs', 'cocktail shaker', 'combination lock', 'crate', 'cuirass', 'dishcloth', 'dome', 'electric guitar', 'filing cabinet', 'fire screen', 'frying pan', 'garbage truck', 'hair clip', 'holster', 'gymnastic horizontal bar', 'hourglass', 'iPod', 'lipstick', 'miniskirt', 'missile', 'mixing bowl', 'oboe', 'pipe organ', 'parallel bars', 'pencil case', 'photocopier', 'poncho', 'prayer rug', 'fishing casting reel', 'school bus', 'scoreboard', 'slot machine', 'snorkel', 'solar thermal collector', 'spider web', 'stage', 'tank', 'front curtain', 'tile roof', 'tobacco shop', 'unicycle', 'upright piano', 'vase', 'wok', 'split-rail fence', 'sailboat', 'traffic or street sign', 'consomme', 'trifle', 'hot dog', 'orange', 'cliff', 'coral reef', 'bolete', 'corn cob']



cifar100_class =['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']







def main(txt=None, key=None, value=None, obj=None, flag=0, test_clip_weights=None, total_classnames=None, session_flag=0):

    start_time = time.time()

    # Load config file
    args = get_arguments()



    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    cfg['init_alpha'] = 1.0
    cfg['init_beta'] = 1.0
    cfg['backbone'] = 'ViT-B/16'
    cfg['shots'] = 30

    cfg['augment_epoch'] = 10



    if cfg['dataset'] == 'eurosat':
        cfg['train_epoch'] = 400
    elif cfg['dataset'] == 'imagenet':
        cfg['train_epoch'] = 50
    else:
        cfg['train_epoch'] = 100



    os.makedirs('output', exist_ok=True)

    file_record = './output/' + cfg['dataset'] + '-' + 'acc_record.txt'

    print("\nRunning configs.")
    print(cfg, "\n")



    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
 

    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)



    print("Preparing dataset.")

    
    args = set_up_datasets(args)
    dataset_name = args.dataset

    if dataset_name == 'cub200':
                
        if session_flag == 0:
            classnames = cub200_class[:100]
        else:
            classnames = cub200_class[100 + 10*(session_flag-1) : 100 + 10*session_flag]
        

    if dataset_name == 'mini_imagenet':
                
        if session_flag == 0:
            classnames = mini_imagenet_class[:60]
        else:
            classnames = mini_imagenet_class[60+5*(session_flag-1) : 60+5*session_flag]
        

    if dataset_name == 'cifar100':
                
        if session_flag == 0:
            classnames = cifar100_class[:60]
        else:
            classnames = cifar100_class[60+5*(session_flag-1) : 60+5*session_flag]
        


    template = ['a photo of a {}']


    
    trainset, train_loader_F,  test_loader, train_loader_cache = get_dataloader(args, session_flag)




    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(classnames, template, clip_model)

    print(classnames)

    temp_classnames = classnames

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys_temp, cache_values_temp, cache_keys_obj_temp = build_cache_model(cfg, clip_model, train_loader_cache)

    cache_keys = []
    cache_values = []
    cache_keys_obj = []





    if dataset_name == 'mini_imagenet' or dataset_name == 'cifar100':

        if flag == 1:
            for i in range(5):
                temp_key = torch.mean(cache_keys_temp[:, i*5:i*5+5], dim=-1)
                temp_value = torch.mean(cache_values_temp[i*5:i*5+5, :], dim=0)
                temp_obj = torch.mean(cache_keys_obj_temp[:, :, i*5:i*5+5], dim=-1)
                
                cache_keys.append(temp_key)
                cache_values.append(temp_value)
                cache_keys_obj.append(temp_obj)
        
        else:
            for i in range(60):
                temp_key = torch.mean(cache_keys_temp[:, i*500:i*500+500], dim=-1)
                temp_value = torch.mean(cache_values_temp[i*500:i*500+500, :], dim=0)
                temp_obj = torch.mean(cache_keys_obj_temp[:, :, i*500:i*500+500], dim=-1)
                
                cache_keys.append(temp_key)
                cache_values.append(temp_value)
                cache_keys_obj.append(temp_obj)




    if dataset_name == 'cub200':


        if flag == 1:
            for i in range(10):
                temp_key = torch.mean(cache_keys_temp[:, i*5:i*5+5], dim=-1)
                temp_value = torch.mean(cache_values_temp[i*5:i*5+5, :], dim=0)
                temp_obj = torch.mean(cache_keys_obj_temp[:, :, i*5:i*5+5], dim=-1)
                
                cache_keys.append(temp_key)
                cache_values.append(temp_value)
                cache_keys_obj.append(temp_obj)
        
        else:
            for i in range(100):
                temp_key = torch.mean(cache_keys_temp[:, i*30:i*30+30], dim=-1)
                temp_value = torch.mean(cache_values_temp[i*30:i*30+30, :], dim=0)
                temp_obj = torch.mean(cache_keys_obj_temp[:, :, i*30:i*30+30], dim=-1)
                
                cache_keys.append(temp_key)
                cache_values.append(temp_value)
                cache_keys_obj.append(temp_obj)



    cache_keys = torch.stack(cache_keys)
    cache_values = torch.stack(cache_values)
    cache_keys_obj = torch.stack(cache_keys_obj)
    
    cache_keys = rearrange(cache_keys, 'a1 a2 -> a2 a1')
    cache_keys_obj = rearrange(cache_keys_obj, 'a1 a2 a3-> a2 a3 a1')

    # ------------------------------------------FineFMPL ------------------------------------------
    run_fg_cpl(cfg, cache_keys, cache_values, test_loader, clip_weights, clip_model, train_loader_F, cache_keys_obj, key=key, value=value, obj=obj, flag=flag, test_clip_weights=test_clip_weights,
                      total_classnames = total_classnames, temp_classnames=temp_classnames, session_flag=session_flag, dataset_name=dataset_name)
    


    with open(file_record, 'a') as file:
        file.write("\n")
        file.write('total time: {:.1f}min'.format((time.time()-start_time)/60))
        file.write("\n")

    return cache_keys, cache_values, cache_keys_obj, clip_weights, temp_classnames






if __name__ == '__main__':
    start_time = time.time()
    
    args = get_arguments()
    args = set_up_datasets(args)

    if args.dataset == 'cub200':
        session_number = 11

        for i in range (session_number):
            if i == 0:
                txt = 'session_'+str(i+1)+'.txt'
                cache_keys, cache_values, cache_keys_obj, t_clip_weights, temp_classnames = main(txt=txt, session_flag=i)
                key = cache_keys.clone()
                value = cache_values.clone()
                obj = cache_keys_obj.clone()
                test_clip_weights = t_clip_weights.clone()
                total_classnames = temp_classnames
            else:
                txt = 'session_'+str(i+1)+'.txt'
                cache_keys, cache_values, cache_keys_obj, t_clip_weights, temp_classnames = main(txt=txt, key=key, value=value, obj=obj, flag=1, test_clip_weights=test_clip_weights, total_classnames=total_classnames,
                                                                                                session_flag=i)
                key = torch.cat((key, cache_keys), dim=-1)


                value_temp = torch.zeros((value.size(0)+cache_values.size(0)), cache_values.size(1)).cuda()
                value_temp [:value.size(0), :cache_values.size(1)-10] = value.clone()
                value_temp [value.size(0):, :] = cache_values.clone()
                value = value_temp.clone()


                obj = torch.cat((obj, cache_keys_obj), dim=-1)
                test_clip_weights = torch.cat((test_clip_weights, t_clip_weights), dim=-1)


                total_classnames = total_classnames + temp_classnames



    else:
        session_number = 9 

        for i in range (session_number):
            if i == 0:
                txt = 'session_'+str(i+1)+'.txt'
                cache_keys, cache_values, cache_keys_obj, t_clip_weights, temp_classnames = main(txt=txt, session_flag=i)
                key = cache_keys.clone()
                value = cache_values.clone()
                obj = cache_keys_obj.clone()
                test_clip_weights = t_clip_weights.clone()
                total_classnames = temp_classnames
            else:
                txt = 'session_'+str(i+1)+'.txt'
                cache_keys, cache_values, cache_keys_obj, t_clip_weights, temp_classnames = main(txt=txt, key=key, value=value, obj=obj, flag=1, test_clip_weights=test_clip_weights, total_classnames=total_classnames,
                                                                                                session_flag=i)
                key = torch.cat((key, cache_keys), dim=-1)


                value_temp = torch.zeros((value.size(0)+cache_values.size(0)), cache_values.size(1)).cuda()
                value_temp [:value.size(0), :cache_values.size(1)-5] = value.clone()
                value_temp [value.size(0):, :] = cache_values.clone()
                value = value_temp.clone()


                obj = torch.cat((obj, cache_keys_obj), dim=-1)
                test_clip_weights = torch.cat((test_clip_weights, t_clip_weights), dim=-1)



                total_classnames = total_classnames + temp_classnames


 
 
    print('--------------------------------------------')
    print('total time: {:.1f}min'.format((time.time()-start_time)/60))
    