import torch.nn as nn
import gc
import torch
from .handsmade_unet import UNetWithTextCond

class Unet(nn.Module):
    def __init__(self, token_len):
        super().__init__()

        self.unet_text = UNetWithTextCond(text_dim = token_len).to(torch.bfloat16)
        self.unet_image = UNetWithTextCond(text_dim = token_len).to(torch.bfloat16)
        
        self.conv = nn.Conv2d(6, 3, kernel_size=3, padding=1).to(torch.bfloat16)
        
    def forward(self, img, text_embedding):
        timestep = None
        one_embedding = None
        img_output = None
        one_embedding = torch.ones_like(text_embedding).to(text_embedding.device, torch.bfloat16)

        img_output = self.unet_image(
            img, 
            one_embedding,#one_embedding,
        )
        ones_latent = torch.ones_like(img).to(text_embedding.device, torch.bfloat16)
        text_output = self.unet_text(
            ones_latent, 
            text_embedding,
        )
        combined_output = torch.cat([img_output, text_output], dim=1).to(text_embedding.device)
        return self.conv(combined_output)

    
    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
