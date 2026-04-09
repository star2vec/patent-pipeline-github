import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class DMON_DPR_Model(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Encoder GCN conform specificațiilor din secțiunea 4 [cite: 97]
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # Stratul de pooling care generează soft assignments C [cite: 48, 49]
        self.pool = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, adj):
        # Generăm reprezentările nodurilor 
        x = self.conv1(x, adj).relu()
        # Calculăm matricea de asignare C prin softmax [cite: 48, 50]
        c = F.softmax(self.pool(x), dim=-1)
        return x, c