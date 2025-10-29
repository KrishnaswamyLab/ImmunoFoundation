
import torch
import torch.nn as nn


class SequenceModel(nn.Module):
    def __init__(self, sequence_model_cfg):
        super().__init__()
        self.cfg = sequence_model_cfg
        self.use_esm = getattr(sequence_model_cfg, 'use_esm', False)
        if self.use_esm:
            # lazy import of ESM wrapper
            from .ESMSequenceModel import ESMSequenceModel
            self.encoder = ESMSequenceModel(sequence_model_cfg)
            out_dim = getattr(sequence_model_cfg, 'out_dim', 128)
            self.out_dim = out_dim
        else:
            in_dim = getattr(sequence_model_cfg, 'in_dim')
            out_dim = getattr(sequence_model_cfg, 'out_dim')
            self.linear = nn.Linear(in_dim, out_dim)
            self.out_dim = out_dim

        print(sequence_model_cfg)

    def forward(self, peptides_tokens, mhc_tokens):
        '''
            peptides_tokens: torch.LongTensor or one-hot tensor
            mhc_tokens: torch.LongTensor or one-hot tensor

            returns:
                embeddings: torch.FloatTensor of shape [B, D]
        '''

        if self.use_esm:
            # ESMSequenceModel handles conversion and returns (B, out_dim)
            return self.encoder(peptides_tokens, mhc_tokens)

        # fallback simple linear approach: sum inputs and apply linear
        # ensure we have a float tensor
        x = peptides_tokens + mhc_tokens
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.linear(x)