
import torch.nn as nn
import torch.nn.functional as F

class BiochemicalModel(nn.Module):
    def __init__(self, bio_cfg):
        super().__init__()
        self.cfg = bio_cfg
        if(self.cfg.n_layers==1):
            self.layers = nn.ModuleList([nn.Linear(self.cfg.n_bio_prop, self.cfg.out_dim)])
        else:
            self.layers = nn.ModuleList([nn.Linear(self.cfg.n_bio_prop, self.cfg.hidden_dim)])
            for i in range(self.cfg.n_layers-2):
                self.layers.append(nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim))
            self.layers.append(nn.Linear(self.cfg.hidden_dim, self.cfg.out_dim))
    def forward(self, X):
        for i in range(len(self.layers)):
            X = F.relu(self.layers[i](X))
        return X