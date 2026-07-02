import torch
import torch.nn as nn


class FusionBlock(nn.Module):
    def __init__(self, dim, n_heads, dim_ffn, dropout):
        super().__init__()
        self.ln_seq_q = nn.LayerNorm(dim)
        self.ln_str_kv = nn.LayerNorm(dim)
        self.ln_str_q = nn.LayerNorm(dim)
        self.ln_seq_kv = nn.LayerNorm(dim)
        self.attn_seq_to_str = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.attn_str_to_seq = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ln_seq_ffn = nn.LayerNorm(dim)
        self.ln_str_ffn = nn.LayerNorm(dim)
        self.ffn_seq = nn.Sequential(
            nn.Linear(dim, dim_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ffn, dim),
        )
        self.ffn_str = nn.Sequential(
            nn.Linear(dim, dim_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ffn, dim),
        )

    def forward(self, z_seq, z_str, key_padding_mask):
        seq_q = self.ln_seq_q(z_seq)
        str_kv = self.ln_str_kv(z_str)
        attn_out_seq, _ = self.attn_seq_to_str(seq_q, str_kv, str_kv, key_padding_mask=key_padding_mask, need_weights=False)
        z_seq = z_seq + self.dropout(attn_out_seq)

        str_q = self.ln_str_q(z_str)
        seq_kv = self.ln_seq_kv(z_seq)
        attn_out_str, _ = self.attn_str_to_seq(str_q, seq_kv, seq_kv, key_padding_mask=key_padding_mask, need_weights=False)
        z_str = z_str + self.dropout(attn_out_str)

        z_seq = z_seq + self.dropout(self.ffn_seq(self.ln_seq_ffn(z_seq)))
        z_str = z_str + self.dropout(self.ffn_str(self.ln_str_ffn(z_str)))
        return z_seq, z_str


class CrossModalFusion(nn.Module):
    """Bidirectional cross-modal attention between sequence and structure tokens.

    Implements the paper's z_fused = MHA(Q=z_seq, K=z_str, V=z_str) + MHA(Q=z_str, K=z_seq, V=z_seq),
    extended over n_layers blocks. A learned [CLS] token is prepended to each stream; after the
    final block, the two streams are summed and the CLS row is returned as the complex-level
    representation.
    """

    def __init__(self, fusion_cfg):
        super().__init__()
        self.cfg = fusion_cfg
        dim = fusion_cfg.dim
        self.cls_seq = nn.Parameter(torch.zeros(1, 1, dim))
        self.cls_str = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_seq, std=0.02)
        nn.init.trunc_normal_(self.cls_str, std=0.02)
        self.blocks = nn.ModuleList([
            FusionBlock(dim, fusion_cfg.n_heads, fusion_cfg.dim_ffn, fusion_cfg.dropout)
            for _ in range(fusion_cfg.n_layers)
        ])
        self.final_ln = nn.LayerNorm(dim)

    def forward(self, z_seq, z_str, key_padding_mask):
        B = z_seq.size(0)
        cls_seq = self.cls_seq.expand(B, -1, -1)
        cls_str = self.cls_str.expand(B, -1, -1)
        z_seq = torch.cat([cls_seq, z_seq], dim=1)
        z_str = torch.cat([cls_str, z_str], dim=1)
        cls_pad = torch.zeros(B, 1, dtype=key_padding_mask.dtype, device=key_padding_mask.device)
        kpm = torch.cat([cls_pad, key_padding_mask], dim=1)
        for block in self.blocks:
            z_seq, z_str = block(z_seq, z_str, kpm)
        fused = self.final_ln(z_seq + z_str)
        cls_out = fused[:, 0, :]
        return cls_out
