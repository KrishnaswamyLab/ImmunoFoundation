
import torch.nn as nn

class BiochemicalModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        #Return the embedding
        return x