import torch
import torch.nn as nn
from typing import List
import esm

class ESM(nn.Module):
    """Simple wrapper around Facebook ESM models.

    Expects token tensors (either integer token ids or one-hot) for peptides and MHC.
    The model will convert tokens to strings using `vocab` from config and pass
    them to a pretrained ESM model. The resulting sequence representation is
    projected to `out_dim`.
    """

    def __init__(self, seq_cfg):
        super().__init__()
        self.cfg = seq_cfg

        esm_variant = getattr(self.cfg, 'esm_variant', 'esm2_t6_8M_UR50D')
        self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(esm_variant)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.CLS, self.EOS, self.PAD = self.alphabet.cls_idx, self.alphabet.eos_idx, self.alphabet.padding_idx

    def tokenize(self, sequences):
        data = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
        return self.batch_converter(data)

    def aggregate(self, batch_tokens, token_reps):
        mask = (batch_tokens != self.PAD) & (batch_tokens != self.CLS) & (batch_tokens != self.EOS)  # (B, T)
        # expand mask to (B, T, 1) and compute masked mean
        masked = token_reps * mask.unsqueeze(-1)
        lengths = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)  # (B, 1)
        seq_reprs = masked.sum(dim=1) / lengths              # (B, C)  ← per-sequence embedding
        return seq_reprs

    def forward(self, peptide_sequences, mhc_sequences):
        peptide_labels, peptide_strs, peptide_tokens = self.tokenize(peptide_sequences)
        mhc_labels, mhc_strs, mhc_tokens = self.tokenize(mhc_sequences)

        peptide_out = self.esm_model(peptide_tokens, repr_layers=[self.cfg.rep_layer], return_contacts=False)   
        peptide_reps = peptide_out["representations"][self.cfg.rep_layer]       
        mhc_out = self.esm_model(mhc_tokens, repr_layers=[self.cfg.rep_layer], return_contacts=False)    
        mhc_reps = mhc_out["representations"][self.cfg.rep_layer]
        
        if(self.cfg.aggregate):
            peptide_reps = self.aggregate(peptide_tokens, peptide_reps)
            mhc_reps = self.aggregate(mhc_tokens, mhc_reps)
        return peptide_reps, mhc_reps