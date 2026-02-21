import torch
import torch.nn as nn

class QueryAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, queries, key_value, apply_norm=True):
        attn_output, _ = self.attn(query=queries, key=key_value, value=key_value)
        x = attn_output + queries  # 残差连接
        if apply_norm:
            x = self.norm1(x)

        ffn_output = self.ffn(x)
        out = ffn_output + x  # 残差连接
        if apply_norm:
            out = self.norm2(out)

        return out

class MultiLayerQueryAttentionTransformer(nn.Module):
    def __init__(self, d_model, n_queries, n_heads=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.n_queries = n_queries
        self.n_heads = n_heads
        self.num_layers = num_layers

        self.query_embed = nn.Parameter(torch.randn(n_queries, d_model))

        self.layers = nn.ModuleList([
            QueryAttentionLayer(d_model, n_heads) for _ in range(num_layers)
        ])

    def forward(self, x):
        b, m, d = x.shape
        assert d == self.d_model

        queries = self.query_embed.unsqueeze(0).expand(b, -1, -1)

        for i, layer in enumerate(self.layers):
            # 最后一层不做 LayerNorm
            apply_norm = (i != self.num_layers - 1)
            queries = layer(queries, x, apply_norm=apply_norm)

        return queries

    def _init_weights(self):
        nn.init.normal_(self.query_embed, mean=0.0, std=0.02)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # xavier_uniform 是常用初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                # 归一化层权重初始化为1，偏置为0
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_normal_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)
                nn.init.xavier_normal_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0)
