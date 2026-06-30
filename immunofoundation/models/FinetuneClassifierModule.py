"""
Finetune classifier wrapper for the ImmunoFoundation backbone.

"""

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class FinetuneClassifierModule(LightningModule):
    """LightningModule that reuses ImmunoFoundationMonomerModule as an encoder
    and places a small classifier head on top to predict immunogenicity.
    """

    def __init__(self, backbone, num_classes: int = 2, bio_dim: int = 0, hidden_dims=None, lr: float = 1e-3, class_weights=None):
        super().__init__()
        self.backbone = backbone
        self.lr = lr
        self.bio_dim = int(bio_dim)

        # infer dims from backbone config if available
        seq_dim = getattr(self.backbone.model_cfg.sequence, 'out_dim', None)
        struct_dim = getattr(self.backbone.model_cfg.structure, 'out_dim', None)
        if seq_dim is None or struct_dim is None:
            raise ValueError('Backbone must have sequence.out_dim and structure.out_dim in its model_cfg')

        in_dim = int(seq_dim) + int(struct_dim) + int(bio_dim)

        # default to a simple 5-layer MLP if not provided
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64, 32]
        layers = []

        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev = h
        layers.append(nn.Linear(prev, int(num_classes)))
        self.classifier = nn.Sequential(*layers)

        # Set up class weights for CrossEntropyLoss if provided
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        # backbone.encode returns seq_embeddings, struct_embeddings
        seq_embeddings, struct_embeddings = self.backbone.encode(batch)

        # pooling: masked mean over residues if masks present
        if 'masks' in batch:
            mask = batch['masks'].unsqueeze(-1)  # (B, L, 1)
            inv_mask = (1 - mask)
            seq_pool = (seq_embeddings * inv_mask).sum(dim=1) / inv_mask.sum(dim=1).clamp(min=1)
            struct_pool = (struct_embeddings * inv_mask).sum(dim=1) / inv_mask.sum(dim=1).clamp(min=1)
        else:
            seq_pool = seq_embeddings.mean(dim=1)
            struct_pool = struct_embeddings.mean(dim=1)

        bio = batch.get('biochem')
        if bio is None:
            # pad with zeros to match expected bio_dim
            if self.bio_dim > 0:
                print("[WARNING] bio_dim > 0 but 'biochem' features not found in batch. Padding with zeros.")
                bio = torch.zeros(seq_pool.size(0), self.bio_dim, dtype=torch.float32, device=seq_pool.device)
            else:
                bio = None

            # Classifier input: [pooled, bio] if bio_dim > 0 and bio present in batch
            if self.bio_dim > 0 and batch.get('bio') is not None:
                print("[WARNING] bio_dim > 0 but 'biochem' features not found in batch. Using 'bio' features instead.")
                bio = batch['bio']
                x = torch.cat([seq_pool, struct_pool, bio], dim=-1)
            else:
                x = torch.cat([seq_pool, struct_pool], dim=-1)

        # Save for debug printing in training/validation step
        self._last_classifier_input = x.detach().cpu() if not hasattr(self, '_last_classifier_input') else x.detach().cpu()
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch.get('label')
        if labels is None:
            labels = batch.get('labels')
        if labels is None:
            raise ValueError('Batch must contain label for supervised finetuning')
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.long, device=logits.device)
        else:
            labels = labels.to(logits.device)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()

        # # Debug print for first batch of each epoch
        # if batch_idx == 0:
        #     print("[DEBUG][TRAIN] Classifier input (first batch):", self._last_classifier_input[:5])
        #     print("[DEBUG][TRAIN] Predictions:", preds[:10].detach().cpu().numpy())
        #     print("[DEBUG][TRAIN] Labels:", labels[:10].detach().cpu().numpy())
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch.get('label')
        if labels is None:
            labels = batch.get('labels')
        if labels is None:
            raise ValueError(
                "Validation batch is missing labels (key 'label' or 'labels'). "
                "Ensure your merged CSV contains an 'immunogenicity' column and that the validation split includes labels."
            )
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.long, device=logits.device)
        else:
            labels = labels.to(logits.device)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()

        # # Debug print for first batch of each epoch
        # if batch_idx == 0:
        #     print("[DEBUG][VAL] Classifier input (first batch):", self._last_classifier_input[:5])
        #     print("[DEBUG][VAL] Predictions:", preds[:10].detach().cpu().numpy())
        #     print("[DEBUG][VAL] Labels:", labels[:10].detach().cpu().numpy())
        
        self.log('val/loss', loss, on_epoch=True)
        self.log('val/acc', acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
