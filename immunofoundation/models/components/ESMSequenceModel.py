import torch
import torch.nn as nn
from typing import List

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
PADDING_CHAR = 'J'


class ESMSequenceModel(nn.Module):
    """Simple wrapper around Facebook ESM models.

    Expects token tensors (either integer token ids or one-hot) for peptides and MHC.
    The model will convert tokens to strings using `vocab` from config and pass
    them to a pretrained ESM model. The resulting sequence representation is
    projected to `out_dim`.
    """

    def __init__(self, cfg, device=torch.device('cpu')):
        super().__init__()
        self.cfg = cfg
        self.device = device
        try:
            import esm
        except Exception as e:
            raise ImportError("ESM package not available. Install `pip install fair-esm` or similar. Error: {}".format(e))

        model_name = getattr(cfg, 'model_name', 'esm2_t6_8M_UR50D')
        # Load a reasonably small default model unless configured otherwise.
        try:
            self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        except Exception:
            # Try fallback by name in pretrained dict
            try:
                model_ctor = esm.pretrained.__dict__.get(model_name)
                if model_ctor is None:
                    raise RuntimeError(f"ESM model '{model_name}' not found in esm.pretrained")
                self.esm_model, self.alphabet = model_ctor()
            except Exception as e:
                raise RuntimeError(f"Unable to load ESM model '{model_name}': {e}")

        self.esm_model = self.esm_model.to(device)
        self.batch_converter = self.alphabet.get_batch_converter()

        esm_dim = getattr(self.esm_model, 'embed_dim', None)
        if esm_dim is None:
            # fallback size
            esm_dim = 1280

        out_dim = getattr(cfg, 'out_dim', 128)
        self.proj = nn.Linear(esm_dim, out_dim)

        if getattr(cfg, 'freeze_esm', True):
            for p in self.esm_model.parameters():
                p.requires_grad = False
            self.esm_model.eval()

    def _tokens_to_strings(self, tokens: torch.Tensor, vocab: str) -> List[str]:
        # tokens may be LongTensor of shape (B, L) or (B, L, 1), or one-hot (B, L, C)
        if tokens.dim() == 3 and tokens.size(-1) != 1:
            # one-hot
            tokens = tokens.argmax(dim=-1)
        if tokens.dim() == 3 and tokens.size(-1) == 1:
            tokens = tokens.squeeze(-1)

        tokens = tokens.cpu().long().numpy()
        out = []
        pad_char = getattr(self.cfg, 'pad_char', PADDING_CHAR)
        for seq in tokens:
            chars = []
            for idx in seq:
                if idx < 0 or idx >= len(vocab):
                    ch = 'X'
                else:
                    ch = vocab[idx]
                    if ch == pad_char:
                        ch = 'X'
                chars.append(ch)
            out.append(''.join(chars))
        return out

    @torch.no_grad()
    def forward(self, peptide_tokens: torch.Tensor, mhc_tokens: torch.Tensor):
        # Combine peptide and MHC by simple concatenation (configurable later)
        vocab = getattr(self.cfg, 'vocab', AMINO_ACIDS + PADDING_CHAR)
        pep_strs = self._tokens_to_strings(peptide_tokens, vocab)
        mhc_strs = self._tokens_to_strings(mhc_tokens, vocab)

        combined = [p + mh for p, mh in zip(pep_strs, mhc_strs)]
        batch = [(str(i), seq) for i, seq in enumerate(combined)]
        labels, strs, toks = self.batch_converter(batch)
        toks = toks.to(self.device)

        # Run through ESM
        results = self.esm_model(toks, repr_layers=[self.esm_model.num_layers if hasattr(self.esm_model, 'num_layers') else max(getattr(self.esm_model, 'layers', [0]))], return_contacts=False)
        reprs = results['representations']
        layer = max(reprs.keys())
        token_reprs = reprs[layer]

        # exclude BOS/EOS
        token_reprs = token_reprs[:, 1:toks.size(1)-1, :]
        seq_repr = token_reprs.mean(1)
        emb = self.proj(seq_repr)
        return emb
