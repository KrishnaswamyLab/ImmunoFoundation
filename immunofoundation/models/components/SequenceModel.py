
import torch.nn as nn

class SequenceModel(nn.Module):
    def __init__(self,sequence_model_cfg):
        super().__init__()
        self.linear = nn.Linear(sequence_model_cfg.in_dim, sequence_model_cfg.out_dim)
        print(sequence_model_cfg)

    def forward(self,peptides_tokens,mhc_tokens):
        '''
            peptides_tokens: torch.LongTensor of shape [B, L, 1]
            mhc_tokens: torch.LongTensor of shape [B, L, 1]

            returns:
                embeddings: torch.FloatTensor of shape [B, D]
        '''

        #Return the embedding

        embeddings = self.linear(peptides_tokens+mhc_tokens)

        return embeddings