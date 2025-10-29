import torch
import torch.nn as nn
from typing import List

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
PADDING_CHAR = 'J'


class ESMEncoder(nn.Module):
    """
    Lightweight wrapper to get sequence embeddings from an ESM model and project
    them to a desired latent dimension.

    Notes:
    - Expects input as one-hot encoded tensors of shape (B, L, 21) or flattened (B, L*21).
    - Requires the `esm` python package (or the local esm code) to be importable. If
      not available, the constructor will raise an informative ImportError.
    - For compatibility with existing code that expects VAE-like outputs, this
      wrapper returns a single deterministic embedding (mu) and a dummy logvar of zeros.
    """

    def __init__(self, latent_dim: int, device: torch.device = torch.device('cpu'), model_name: str = None, freeze_esm: bool = True):
        super().__init__()
        try:
            import esm
        except Exception as e:
            raise ImportError("ESM package not found. Install `pip install fair-esm` or make the `esm` package importable. Original error: {}".format(e))

        # Default model if not provided. Using a reasonably small model is recommended
        # for quick experiments; change model_name if you want a different pretrained model.
        if model_name is None:
            model_name = "esm1b_t33_650M_UR50S"

        # Load model from esm.pretrained (this may download weights if not available locally)
        # We keep the loading lazy by using pretrained API.
        pretrained = getattr(esm.pretrained, model_name, None)
        if pretrained is None:
            # fallback to esm.pretrained.load_model_and_alphabet if available
            try:
                model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
            except Exception:
                # try the convenience constructor
                try:
                    model, alphabet = esm.pretrained.__dict__[model_name]()
                except Exception as e:
                    raise ImportError(f"Unable to load ESM model '{model_name}': {e}")
        else:
            model, alphabet = pretrained()

        self.esm_model = model.to(device)
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        # Determine embedding dimension from the model (common attr in ESM models)
        try:
            esm_dim = model.embed_dim
        except Exception:
            # fallback: run a dummy forward pass later; but assume common dims
            esm_dim = 1280

        self.proj = nn.Linear(esm_dim, latent_dim)
        self.latent_dim = latent_dim
        self.device = device

        if freeze_esm:
            # Don't update ESM weights by default (fine-tune by setting freeze_esm=False)
            for p in self.esm_model.parameters():
                p.requires_grad = False
            self.esm_model.eval()

    def _onehot_to_strings(self, onehot: torch.Tensor) -> List[str]:
        # onehot: (B, L, C) expected C == len(AMINO_ACIDS)+1
        if onehot.dim() == 2:
            # flattened input (B, L*C)
            b, lc = onehot.shape
            c = len(AMINO_ACIDS) + 1
            L = lc // c
            onehot = onehot.view(b, L, c)

        b, L, c = onehot.shape
        assert c == len(AMINO_ACIDS) + 1, f"Unexpected one-hot channels: {c}"

        # Map indices to characters, convert padding char -> 'X' (unknown) for ESM
        char_map = list(AMINO_ACIDS) + [PADDING_CHAR]
        out = []
        indices = torch.argmax(onehot, dim=-1).cpu().numpy()
        for i in range(b):
            chars = []
            for j in range(L):
                idx = int(indices[i, j])
                ch = char_map[idx] if idx < len(char_map) else 'X'
                if ch == PADDING_CHAR:
                    ch = 'X'
                chars.append(ch)
            out.append(''.join(chars))
        return out

    @torch.no_grad()
    def forward(self, onehot: torch.Tensor) -> torch.Tensor:
        """
        onehot: (B, L, C) or (B, L*C) tensor on any device. Returns: mu (B, latent_dim)
        """
        seqs = self._onehot_to_strings(onehot)

        # Build batch for ESM
        batch = [(str(i), seq) for i, seq in enumerate(seqs)]
        labels, strs, tokens = self.batch_converter(batch)
        tokens = tokens.to(self.device)

        # Forward pass through ESM to get token representations from the last layer
        # Using the esm model in eval mode (weights may be frozen)
        with torch.no_grad():
            results = self.esm_model(tokens, repr_layers=[self.esm_model.num_layers if hasattr(self.esm_model, 'num_layers') else max(getattr(self.esm_model, 'layers', [0])),], return_contacts=False)
            # Attempt to pick the top representation key
            reprs = results['representations']
            # get the final layer key
            layer = max(reprs.keys())
            token_representations = reprs[layer]  # (B, seq_len, esm_dim)

        # Exclude start/end tokens when averaging
        # ESM tokens include BOS/EOS; we take mean across token positions 1:-1
        seq_lens = (tokens != self.alphabet.padding_idx).sum(1)
        # compute mean excluding first and last token
        token_representations = token_representations[:, 1: tokens.size(1)-1, :]
        seq_repr = token_representations.mean(1)  # (B, esm_dim)

        mu = self.proj(seq_repr)
        return mu
