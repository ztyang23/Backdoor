import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Optional, Tuple, List

from llava.model.unet import Unet


class Attention(nn.Module):

    def __init__(self, embed_size, heads, out_dim):
        super(Attention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, out_dim)

    def forward(self, values, keys, query, mask=None, cls_token=False):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix multiplication for query*keys for each training example
        # with every other key
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(attention / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        if cls_token:
            out = self.fc_out(out[:, 0, :])
        else:
            out = self.fc_out(out)
        return out

class TransformerModule(nn.Module):
    def __init__(self, vision_dim, embed_size, token_len, heads=8, num_layers=3, dropout=0.1):
        super(TransformerModule, self).__init__()
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=heads,
            dim_feedforward=512,
            dropout=dropout
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.query_embed = nn.Parameter(torch.zeros(token_len, embed_size)) 
        self.unet_proj = nn.Linear(vision_dim, embed_size)
        # self.unet_proj.weight.data.normal_(0, 0.02)
    
    def init_weight(self):
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_normal_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)
                nn.init.xavier_normal_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0)
        
        self.transformer.apply(_init_weights)
        
        # 初始化可学习的查询嵌入
        nn.init.normal_(self.query_embed, mean=0.0, std=0.02)
        
        # 初始化unet_proj层的偏置（权重已在__init__中初始化）
        nn.init.normal_(self.unet_proj.weight, mean=0.0, std=0.02)
        if self.unet_proj.bias is not None:
            nn.init.constant_(self.unet_proj.bias, 0)


    def forward(self, memory=None, tgt_mask=None, tgt_key_padding_mask=None):
        if memory is None:
            raise ValueError("Memory cannot be None")
            #memory = torch.zeros_like(tgt)  # 如果没有 Encoder，可以传入全零
        
        #memory = memory.permute(0, 2, 3, 1)  # (batch_size, H, W, origin length)
        #memory = memory.flatten(1, 2)  # (batch_size, H*W, origin length)
        memory = self.unet_proj(memory)
        #print(memory.shape)
        #print(memory.shape)
        batch_size = memory.shape[0]
        memory = memory.permute(1, 0, 2)

        tgt = self.query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        #memory = memory.permute(1, 0, 2)

        output = self.transformer(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        output = output.permute(1, 0, 2)
        return output


class UnetTrans(nn.Module):
    def __init__(self, dim, out_dim, token_len, embed_dim):
        super(UnetTrans, self).__init__()
        self.unet = Unet(token_len=token_len)
        self.unet_proj = nn.Linear(out_dim, embed_dim)
        self.transformer = TransformerModule(embed_size=dim, heads=8, out_dim=out_dim)

        self.query_embed = nn.Parameter(torch.zeros(token_len, embed_dim)) 
    
    def forward(self, img, text_embedding, mask=None):
        # x = self.unet(x)
        x = self.unet(img, text_embedding)
        x = x.permute(0, 2, 3, 1)  # (batch_size, H, W, origin length)
        x = x.flatten(1, 2)  # (batch_size, H*W, origin length)
        memory = self.unet_proj(x)
        #print(memory.shape)
        batch_size = memory.shape[0]

        #print(batch_size)

        memory = memory.permute(1, 0, 2) 
        print(memory.shape)
        #print(memory.shape)

        #batch_size, d_model, H, W = memory.shape
        #memory = memory.flatten(2).permute(2, 0, 1)  # (H*W, batch_size, d_model)
        
        # 目标序列（可学习 query）
        tgt = self.query_embed.unsqueeze(1).repeat(1, batch_size, 1)  # (100, batch_size, d_model)
        #print(tgt.shape)
        # Transformer Decoder
        output = self.transformer(tgt, memory=memory)  # (100, batch_size, d_model)
        
        # 分类头
        output = output.transpose(0, 1)
        return output


if __name__ == "__main__":
    #model = Unet()
    #model = TransformerModule(embed_size=32, heads=4, out_dim=32)
    uni_model = UnetTrans(
     dim = 64,
     out_dim = 3,
     token_len=1500,
     embed_dim=64)
        
    uni_model.to("cuda").to(torch.bfloat16)

    text_embedding = torch.ones(2, 64, 1500).to("cuda").to(torch.bfloat16)
    img = torch.ones(2, 3, 336, 336).to("cuda").to(torch.bfloat16)
    
    out = uni_model(img, text_embedding)
    #print(out.shape)