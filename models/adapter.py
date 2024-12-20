from torch import nn
import torch.nn.functional as F

class Adapter(nn.Module):
    # embed_dim：输入/输出
    def __init__(self, adapter_dim, embed_dim):
        super(Adapter, self).__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.down_project = nn.Linear(embed_dim, adapter_dim, bias=False)
        self.up_project = nn.Linear(adapter_dim, embed_dim, bias=False)

    def forward(self, z):
        normalized_z = self.layer_norm(z)
        h = F.relu(self.down_project(normalized_z))
        return self.up_project(h) + z