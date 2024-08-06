from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

import cv2
from torchvision import transforms
tensor_to_image = transforms.ToPILImage()
import time


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)






class ResidualAttentionBlock_visual(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        output, weights = self.attn(x, x, x, need_weights=True, average_attn_weights = False, attn_mask = self.attn_mask)
        return output, weights


    def forward(self, x: torch.Tensor, func = None):
        

        EPSILON = 1e-6


        if func is not None:
     
            fg_probe = x[-1,:,:].clone().unsqueeze(0)
            temp = x.clone()

            temp_new = temp*fg_probe
            temp_new = torch.sign(temp_new) * torch.sqrt(torch.abs(temp_new) + EPSILON)

            if torch.isnan(temp_new).any():
                temp_new = temp.float() * fg_probe.float()
                temp_new = torch.sign(temp_new) * torch.sqrt(torch.abs(temp_new) + EPSILON)
                temp_new = temp_new.half()



            if torch.isnan(temp_new).any():
                print('temp_new error')
            


            temp_new_x = func.new_linear.half()(temp_new)


            if torch.isnan(temp_new_x).any():
                
                temp_new = temp.float() * fg_probe.float()
                temp_new = torch.sign(temp_new) * torch.sqrt(torch.abs(temp_new) + EPSILON)
                temp_new_x = func.new_linear.float()(temp_new) 
                temp_new_x = temp_new_x.half()
         


            if torch.isnan(temp_new_x).any():
                print('temp_new_x error')



            x = x + temp_new_x


        
        output, weights = self.attention(self.ln_1(x))
        x = x + output


        x = x + self.mlp(self.ln_2(x))



        return x, weights



class Transformer_visual(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock_visual(width, heads, attn_mask) for _ in range(layers)])


    def part_select_ori(self, x):
        length = len(x)
        last_map = x[0]
        for i in range(1, length):
            last_map = torch.matmul(x[i], last_map)


        last_map = last_map[:,:,0,1:]


        _, max_inx = last_map.max(2)



        B,C = last_map.size(0),last_map.size(1)
        patch_num = last_map.size(-1)

        H = patch_num ** 0.5
        H = int(H)
        attention_map = last_map.view(B,C,H,H)

        return _, max_inx, attention_map




    def part_select(self, x):
        length = len(x)
        last_map = x[0]
        for i in range(1, length):
            last_map = torch.matmul(x[i], last_map)



        last_map = last_map[:,:,0,1:-1]



        _, max_inx = last_map.max(2)



        B,C = last_map.size(0),last_map.size(1)
        patch_num = last_map.size(-1)

        H = patch_num ** 0.5
        H = int(H)
        attention_map = last_map.view(B,C,H,H)

        return _, max_inx, attention_map



    #saliency map
    def saliency_extraction(self, xf_ori):

        xf = xf_ori.clone()
        
        eps = 1e-8
        b=xf.size(0)
        c=xf.size(1)
        h=xf.size(2)
        w=xf.size(3)

        coord = torch.zeros(b, 4)
        coord = coord.cuda()

        saliency = torch.sum(xf, dim=1)*(1.0/(c+eps))

        saliency = saliency.contiguous()
        saliency = saliency.view(b, -1)

        sa_min = torch.min(saliency, dim=1)[0]
        sa_max = torch.max(saliency, dim=1)[0]
        interval = sa_max - sa_min

        sa_min = sa_min.contiguous()
        sa_min = sa_min.view(b, 1)
        sa_min = sa_min.expand(h, w, b, 1)
        sa_min = sa_min.contiguous()
        sa_min = rearrange(sa_min, 'h w b 1 -> b 1 h w')

        interval = interval.contiguous()
        interval = interval.view(b, 1)
        interval = interval.expand(h, w, b, 1)
        interval = interval.contiguous()
        interval = rearrange(interval, 'h w b 1 -> b 1 h w')

        saliency = saliency.contiguous()
        saliency = saliency.view(b, 1, h, w)

        saliency = saliency - sa_min
        saliency = saliency/(interval+eps)

        saliency = torch.clamp(saliency, eps, 1)

        for i in range(b):
            img1 = saliency[i,:,:,:]
            img2 = img1.view(1, h, w)
            img2 = img2*255
            img2 = img2.detach().cpu()
            img2 = img2.numpy()
            mat1 = np.uint8(img2)
            mat1 = mat1.transpose(1,2,0)
            thres, mat2 = cv2.threshold(mat1,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            contours, hierarchy = cv2.findContours(mat2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            area = []
    
            if len(contours)==0:
                coord[i, 0]=0
                coord[i, 1]=0
                coord[i, 2]=w
                coord[i, 3]=h
            else:

                for k in range(len(contours)):
                    area.append(cv2.contourArea(contours[k]))
                max_idx = np.argmax(np.array(area))

                p, q, r, s = cv2.boundingRect(contours[max_idx]) 
                coord[i, 0]=p
                coord[i, 1]=q
                coord[i, 2]=r
                coord[i, 3]=s

     
        coord = coord.detach()

        return coord, coord





    def forward(self, x: torch.Tensor, y = None, new_trans=None):

        all_weights = []
        for i in range(self.layers - 1):
            x, weights = self.resblocks[i](x)
            all_weights.append(weights)

        x_ori = x.clone()
        output_ori, weights = self.resblocks[-1](x_ori)



        if y is not None:


            x_obj, _ = self.resblocks[-1](x, func = new_trans)

            o1_query = x_obj[-1,:,:].unsqueeze(0)

            out_ori = o1_query.clone()

            o1_key = x_obj[1:-1,:,:]

            o1_query = rearrange(o1_query, 'a1 a2 a3 -> a2 a1 a3') #(256, 1, 768)
            o1_key = rearrange(o1_key, 'a1 a2 a3 -> a2 a1 a3') #(256, 196, 768)
            o1_key_ori = o1_key.clone()


            o1_query = o1_query/o1_query.norm(dim=-1, keepdim=True)
            o1_key = o1_key/o1_key.norm(dim=-1, keepdim=True)

            o1_query = rearrange(o1_query, 'a1 a2 a3 -> a1 a3 a2') #(256, 768, 1)
            o1_value = torch.bmm(o1_key, o1_query)
             

            if torch.isnan(o1_value).any():
                o1_value = torch.bmm(o1_key.float(), o1_query.float())
                o1_value = o1_value.half()


            o1_value = o1_value.squeeze(2)/0.1
            o1_value = F.softmax(o1_value, dim=-1)
            o1_value = o1_value.unsqueeze(2)


            o1_new = o1_value*o1_key_ori

            if torch.isnan(o1_new).any():
                o1_new = o1_value.float() * o1_key_ori.float()
                o1_new = o1_new.half()


            o1_new = torch.sum(o1_new, dim=1, keepdim=True)
        
            o1_new = 1.0*rearrange(o1_new, 'a1 a2 a3 -> a2 a1 a3') +  x_obj[0,:,:].unsqueeze(0) #(1, 256, 768) 


            return output_ori, o1_new
        else:


            _, part_inx, attention_map = self.part_select_ori(all_weights)

            width = attention_map.size(-1)

            parts_ori = output_ori.clone()

            parts_ori =  rearrange(parts_ori, 'a1 a2 a3 -> a2 a1 a3')
            atten_mask = attention_map.contiguous().view(attention_map.size(0), attention_map.size(1), -1)

            atten_mask = atten_mask/(torch.sum(atten_mask, dim=-1, keepdim=True) + 1e-6)

            obj_atten_ori = torch.mean(atten_mask, dim=1, keepdim=True)

            obj_atten =  rearrange(obj_atten_ori.clone(), 'a1 a2 a3 -> a1 a3 a2')


            output_obj =  rearrange(output_ori[1:,:,:].clone(), 'a1 a2 a3 -> a2 a1 a3')


            obj_feature = output_obj * obj_atten
            if torch.isnan(obj_feature).any():
                obj_feature = output_obj.float() * obj_atten.float()
                obj_feature = obj_feature.half()


            obj_feature = torch.sum(obj_feature, dim=1)
            obj_feature = obj_feature.unsqueeze(0)

           

            new_obj = obj_feature



            return output_ori, new_obj





class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer_visual(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))



        self.input_resolution = input_resolution



    def forward(self, x: torch.Tensor, y=None, new_trans=None):


        if x.size(-1)!=self.input_resolution:
            x =  F.interpolate(x, size=[self.input_resolution, self.input_resolution], mode='bicubic')


        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]


        
        posemb = self.positional_embedding.to(x.dtype).unsqueeze(0)

        x = x + posemb



        if y is not None:

            obj_token = y.cuda().to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)

            x = torch.cat([x, obj_token], dim=1)



        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        if y is not None:
            
            x, x_obj = self.transformer(x, y, new_trans)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x_obj = x_obj.permute(1, 0, 2)  # LND -> NLD
            out1 = self.ln_post(x[:, 0, :]) 
            out2 = self.ln_post(x_obj[:, 0:, :]).to(torch.float16)


            if self.proj is not None:
                out1 = out1 @ self.proj
                out2 = out2 @ self.proj



        else:
            x, x_obj_ori = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x_obj_ori = x_obj_ori.permute(1, 0, 2)  # LND -> NLD 

            out1 = self.ln_post(x[:, 0, :])
            out2 = self.ln_post(x_obj_ori[:, 0:, :])


            if self.proj is not None:
                out1 = out1 @ self.proj
                out2 = out2 @ self.proj



        return out1, out2


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, y=None, new_trans=None):
        return self.visual(image.type(self.dtype), y, new_trans)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features, image_features_obj = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)

    return model
