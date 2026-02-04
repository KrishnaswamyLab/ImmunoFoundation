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
        self.register_buffer('aa_to_idx', self._build_aa_mapping())
        self.CLS, self.EOS, self.PAD = self.alphabet.cls_idx, self.alphabet.eos_idx, self.alphabet.padding_idx
        if getattr(self.cfg, "freeze_esm", True):
            for p in self.esm_model.parameters():
                p.requires_grad = False

    def _build_aa_mapping(self):
        standard_aas = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids
        mapping = torch.zeros(len(self.alphabet), dtype=torch.long)
        for i, aa in enumerate(standard_aas):
            esm_idx = self.alphabet.tok_to_idx.get(aa, -1)
            if esm_idx >= 0:
                mapping[esm_idx] = i
        return mapping

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

    def forward(self, sequences, return_tokens = False):
        device = self.esm_model.embed_tokens.weight.device
        labels, strs, tokens = self.tokenize(sequences)
        tokens = tokens.to(device)
        out = self.esm_model(tokens, repr_layers=[self.cfg.rep_layer], return_contacts=False)   
        reps = out["representations"][self.cfg.rep_layer]       
        if(self.cfg.aggregate):
            reps = self.aggregate(tokens, reps)
        if return_tokens:
            return reps[:,1:-1,:], self.aa_to_idx[tokens[:, 1:-1]]
        else:
            return reps[:,1:-1,:]