import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim_q // heads) ** -0.5

        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)

        self.to_q = nn.Linear(dim_q, dim_q)
        self.to_k = nn.Linear(dim_kv, dim_q)
        self.to_v = nn.Linear(dim_kv, dim_q)
        self.to_out = nn.Sequential(
            nn.Linear(dim_q, dim_q),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        B, N, C = x.shape
        H = self.heads

        x = self.norm_q(x)
        context = self.norm_kv(context)

        q = self.to_q(x).reshape(B, N, H, -1).transpose(1, 2).contiguous()
        k = self.to_k(context).reshape(B, context.size(1), H, -1).transpose(1, 2).contiguous()
        v = self.to_v(context).reshape(B, context.size(1), H, -1).transpose(1, 2).contiguous()

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v  # (B, H, N, C//H)
        out = out.transpose(1, 2).contiguous().reshape(B, N, C).contiguous()
        return x + self.to_out(out)  # residual connection


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.block(x)
        return self.pool(x), x  # pooled, skip

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class UNetWithTextCond(nn.Module):
    def __init__(self, img_ch=3, base_ch=512, text_dim=768):
        super().__init__()
        self.down1 = DownBlock(img_ch, base_ch)
        self.down2 = DownBlock(base_ch, base_ch * 2)

        self.middle_conv = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1),
        )

        self.up1 = UpBlock(base_ch * 2, base_ch * 2, base_ch)
        self.up2 = UpBlock(base_ch, base_ch, base_ch)

        self.out_conv = nn.Conv2d(base_ch, img_ch, 1)

        # Project text to match KV dim
        self.cross_attn_mid = SimpleCrossAttention(dim_q=base_ch * 2, dim_kv=text_dim)
        self.cross_attn_up1 = SimpleCrossAttention(dim_q=base_ch, dim_kv=text_dim)
        self.cross_attn_up2 = SimpleCrossAttention(dim_q=base_ch, dim_kv=text_dim)

    def forward(self, x, text_emb):
        b, c, h, w = x.shape

        # Encode
        x, skip1 = self.down1(x)  # (B, C, H/2, W/2)
        x, skip2 = self.down2(x)  # (B, 2C, H/4, W/4)

        # Mid block + attention
        x = self.middle_conv(x)  # (B, 2C, H/4, W/4)
        B, C, H_, W_ = x.shape
        x_flat = x.contiguous().view(B, C, -1).transpose(1, 2).contiguous()  # (B, N, C)
        x_flat = self.cross_attn_mid(x_flat, text_emb)
        x = x_flat.transpose(1, 2).contiguous().view(B, C, H_, W_)

        # Decode + attention after up blocks
        x = self.up1(x, skip2)  # (B, C, H/2, W/2)
        B, C, H_, W_ = x.shape
        x_flat = x.contiguous().view(B, C, -1).transpose(1, 2).contiguous()
        x_flat = self.cross_attn_up1(x_flat, text_emb)
        x = x_flat.transpose(1, 2).contiguous().view(B, C, H_, W_)

        x = self.up2(x, skip1)  # (B, C, H, W)
        B, C, H_, W_ = x.shape
        x_flat = x.contiguous().view(B, C, -1).transpose(1, 2).contiguous()
        x_flat = self.cross_attn_up2(x_flat, text_emb)
        x = x_flat.transpose(1, 2).contiguous().view(B, C, H_, W_)

        return self.out_conv(x)