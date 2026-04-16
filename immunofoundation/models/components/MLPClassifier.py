import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from pytorch_lightning import LightningModule


class MLPClassifier(LightningModule):
    """Simple MLP head for immunogenicity classification.

    Supports optional biochemical encoder (an instance of BiochemicalModel or similar)
    to be supplied externally. If provided, its outputs are concatenated to raw features.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dims=(128, 64), lr: float = 1e-4, dropout=0.1, backbone: Optional[nn.Module] = None):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lr = lr
        self.backbone = backbone

        # compute head input dim
        head_in = input_dim
        if backbone is not None:
            # try to infer backbone output dim if it has attribute cfg.out_dim
            out_dim = getattr(backbone, 'cfg', None)
            if out_dim is not None and hasattr(backbone.cfg, 'out_dim'):
                head_in += int(backbone.cfg.out_dim)
            else:
                # fallback: attempt forward with zeros later will adjust
                head_in += 0

        layers = []
        prev = head_in
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))

        self.head = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features: torch.Tensor):
        """features: (B, input_dim)"""
        x = features
        if self.backbone is not None:
            try:
                with torch.no_grad():
                    enc = self.backbone(x)
            except Exception:
                # if backbone expects different shape, try passing through .forward
                enc = self.backbone(x)
            # enc expected (B, dim) or (B, L, dim) -> flatten last dims
            if enc.dim() > 2:
                enc = enc.mean(dim=1)
            x = torch.cat([x, enc], dim=-1)

        logits = self.head(x)
        return logits

    def training_step(self, batch, batch_idx):
        x = batch['features']
        y = batch['labels']
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y).float().mean()
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['features']
        y = batch['labels']
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y).float().mean()
        self.log('val/loss', loss, on_epoch=True)
        self.log('val/acc', acc, on_epoch=True)
        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def load_backbone_weights_from_checkpoint(self, ckpt_path: str, prefixes_to_try=('bio', 'biochemical', 'biochem')):
        """Try to load matching keys from a checkpoint into `self.backbone` if available.

        This is best-effort and will skip keys that don't match.
        """
        if self.backbone is None:
            return
        try:
            import torch
            ckpt = torch.load(ckpt_path, map_location='cpu')
            # checkpoints saved by Lightning often have 'state_dict' key
            state = ckpt.get('state_dict', ckpt)
            # build mapping for backbone keys
            own_state = self.backbone.state_dict()
            new_state = {}
            for k, v in state.items():
                for p in prefixes_to_try:
                    if p in k.lower():
                        # heuristic: strip common prefixes like 'model.' or similar
                        new_key = k
                        # if key contains 'backbone' or 'biochem' try to map directly
                        if new_key in own_state and own_state[new_key].shape == v.shape:
                            new_state[new_key] = v
                        else:
                            # try to remove leading module names until match
                            parts = new_key.split('.')
                            for i in range(len(parts)):
                                candidate = '.'.join(parts[i:])
                                if candidate in own_state and own_state[candidate].shape == v.shape:
                                    new_state[candidate] = v
                                    break
                        break
            if len(new_state) == 0:
                # last resort: try strict=False load of full state
                try:
                    self.backbone.load_state_dict(state, strict=False)
                    return
                except Exception:
                    return
            own_state.update(new_state)
            self.backbone.load_state_dict(own_state)
        except Exception:
            # don't fail training if this cannot be loaded
            return
