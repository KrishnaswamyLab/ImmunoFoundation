import torch
import torch.nn as nn
from typing import List
import esm

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
PADDING_CHAR = 'J'


class ESMSequenceModel(nn.Module):
    """Simple wrapper around Facebook ESM models.

    Expects token tensors (either integer token ids or one-hot) for peptides and MHC.
    The model will convert tokens to strings using `vocab` from config and pass
    them to a pretrained ESM model. The resulting sequence representation is
    projected to `out_dim`.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        esm_variant = getattr(cfg, 'esm_variant', 'esm2_t6_8M_UR50D')
        self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.CLS, self.EOS, self.PAD = alphabet.cls_idx, alphabet.eos_idx, alphabet.padding_idx

        esm_dim = getattr(self.esm_model, 'embed_dim', 1280)
        out_dim = getattr(cfg, 'out_dim', 128)
        self.proj = nn.Linear(esm_dim, out_dim)

        if getattr(cfg, "freeze_esm", True):
            for p in self.esm_model.parameters():
                p.requires_grad_(False)
            self.esm_model.freeze()
    
    def tokenize(self, sequences):
        data = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
        return self.batch_converter(data)

    def aggregate(self, batch_tokens, token_reps):
        mask = (batch_tokens != PAD) & (batch_tokens != CLS) & (batch_tokens != EOS)  # (B, T)
        # expand mask to (B, T, 1) and compute masked mean
        masked = token_reps * mask.unsqueeze(-1)
        lengths = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)  # (B, 1)
        seq_reprs = masked.sum(dim=1) / lengths              # (B, C)  ← per-sequence embedding
        return self.batch_converter(data)

    def forward(self, peptide_sequences, mhc_sequences):
        peptide_labels, peptide_strs, peptide_tokens = tokenize(peptide_sequences)
        mhc_labels, mhc_strs, mhc_tokens = self.tokenize(mhc_sequences)

        peptide_out = self.esm_model(peptide_tokens, repr_layers=[self.cfg.rep_layer], return_contacts=False)   
        peptide_reps = peptide_out["representations"][33]       
        mhc_out = self.esm_model(mhc_tokens, repr_layers=[self.cfg.rep_layer], return_contacts=False)    
        mhc_reps = mhc_out["representations"][33]
        
        if(self.cfg.aggregate):
            peptide_reps = self.aggregate(peptide_tokens, peptide_reps)
            mhc_reps = self.aggregate(mhc_tokens, mhc_reps)
        return peptide_reps, mhc_reps