import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from immunofoundation.models.components.CrossModalFusion import CrossModalFusion
from immunofoundation.models.components.ESM import ESM
from immunofoundation.models.components.SequenceModel import SequenceModel
from immunofoundation.models.components.StructureModel import StructureModel


_WT_FORWARD_KEYS = (
    # Full path
    "peptide_sequence", "mhc_sequence",
    "peptide_coords", "mhc_coords",
    "peptide_adjs", "mhc_adjs",
    # Cached fast path
    "peptide_z_seq", "peptide_z_str",
    "mhc_z_seq", "mhc_z_str",
    # Common
    "peptide_pad_mask", "mhc_pad_mask",
)


class ImmunoFoundationFinetuneModule(LightningModule):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.aa_embedding_model = ESM(model_cfg.sequence)
        self.sequence_model = SequenceModel(model_cfg.sequence)
        self.structure_model = StructureModel(model_cfg.structure)

        self.aggregator_type = str(getattr(model_cfg, "aggregator", "fusion")).lower()
        if self.aggregator_type == "fusion":
            self.fusion = CrossModalFusion(model_cfg.fusion)
            self.head = nn.Linear(model_cfg.fusion.dim, 1)
        elif self.aggregator_type == "pool":
            # mean-pool peptide_seq, peptide_str, mhc_seq, mhc_str over residues, then concat
            self.fusion = None
            head_cfg = getattr(model_cfg, "head", None) or {}
            def _hget(k, d):
                if isinstance(head_cfg, dict):
                    return head_cfg.get(k, d)
                return getattr(head_cfg, k, d)
            in_dim = 4 * model_cfg.fusion.dim
            hidden_dim = int(_hget("hidden_dim", in_dim // 2))
            dropout = float(_hget("dropout", 0.1))
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            print(f"[ImmunoFoundationFinetuneModule] aggregator=pool, head MLP {in_dim}->{hidden_dim}->1, dropout={dropout}", flush=True)
        else:
            raise ValueError(f"Unknown aggregator {self.aggregator_type!r}; expected 'fusion' or 'pool'")

        self.freeze_backbone = bool(getattr(model_cfg, "freeze_backbone", False))
        if self.freeze_backbone:
            for p in self.sequence_model.parameters():
                p.requires_grad = False
            for p in self.structure_model.parameters():
                p.requires_grad = False
            self.sequence_model.eval()
            self.structure_model.eval()
            n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"[ImmunoFoundationFinetuneModule] freeze_backbone=True (sequence_model + structure_model frozen). Trainable params: {n_trainable:,}", flush=True)

        pos_weight = float(getattr(model_cfg, "pos_weight", None) or 3.34)
        self.register_buffer("pos_weight_buf", torch.tensor([pos_weight]))
        # Eval loss is always BCE (interpretable, monitor-friendly).
        self.eval_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_buf)

        contrastive_cfg = getattr(model_cfg, "contrastive", None) or {}
        def _cget(key, default):
            if isinstance(contrastive_cfg, dict):
                return contrastive_cfg.get(key, default)
            return getattr(contrastive_cfg, key, default)
        self.contrastive_enabled = bool(_cget("enabled", False))
        self.lambda_cw = float(_cget("lambda_cw", 0.1))
        self.lambda_off_diag = float(_cget("lambda_off_diag", 5e-3))

        self.loss_type = str(getattr(model_cfg, "loss_type", "bce")).lower()
        if self.loss_type == "aucm":
            from libauc.losses import AUCMLoss
            # imratio = n_pos / n_total. We know pos_weight = n_neg / n_pos, so:
            imratio = 1.0 / (1.0 + pos_weight)
            aucm_cfg = getattr(model_cfg, "aucm", None) or {}
            margin = float(getattr(aucm_cfg, "margin", 1.0) if not isinstance(aucm_cfg, dict) else aucm_cfg.get("margin", 1.0))
            self.train_loss = AUCMLoss(margin=margin, imratio=imratio)
            print(f"[ImmunoFoundationFinetuneModule] train loss=AUCMLoss(margin={margin}, imratio={imratio:.4f}); eval loss=BCE", flush=True)
        elif self.loss_type == "bce":
            self.train_loss = self.eval_loss
        else:
            raise ValueError(f"Unknown loss_type {self.loss_type!r}; expected 'bce' or 'aucm'")

        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()
        self.test_auroc = BinaryAUROC()
        self.test_auprc = BinaryAveragePrecision()
        self._val_scores: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []
        self._test_scores: list[torch.Tensor] = []
        self._test_labels: list[torch.Tensor] = []

    def forward_embedding(self, batch):
        # Cached fast path or full path → produce per-residue [B, L, D] tensors.
        if "peptide_z_seq" in batch:
            peptide_seq = batch["peptide_z_seq"]
            peptide_str = batch["peptide_z_str"]
            mhc_seq = batch["mhc_z_seq"]
            mhc_str = batch["mhc_z_str"]
        else:
            with torch.no_grad():
                peptide_esm = self.aa_embedding_model(batch["peptide_sequence"])
                mhc_esm = self.aa_embedding_model(batch["mhc_sequence"])

            assert peptide_esm.size(1) == batch["peptide_coords"].size(1), (
                f"peptide ESM len {peptide_esm.size(1)} != coords len {batch['peptide_coords'].size(1)}"
            )
            assert mhc_esm.size(1) == batch["mhc_coords"].size(1), (
                f"mhc ESM len {mhc_esm.size(1)} != coords len {batch['mhc_coords'].size(1)}"
            )

            peptide_seq = self.sequence_model(peptide_esm)
            mhc_seq = self.sequence_model(mhc_esm)
            peptide_str = self.structure_model(batch["peptide_adjs"], batch["peptide_coords"])
            mhc_str = self.structure_model(batch["mhc_adjs"], batch["mhc_coords"])

        pep_pad = batch["peptide_pad_mask"]
        mhc_pad = batch["mhc_pad_mask"]

        if self.aggregator_type == "pool":
            return self._mean_pool_concat(peptide_seq, peptide_str, pep_pad,
                                          mhc_seq, mhc_str, mhc_pad)

        z_seq = torch.cat([peptide_seq, mhc_seq], dim=1)
        z_str = torch.cat([peptide_str, mhc_str], dim=1)
        pad_mask = torch.cat([pep_pad, mhc_pad], dim=1)
        return self.fusion(z_seq, z_str, pad_mask)

    @staticmethod
    def _masked_mean(z, pad):
        # z: [B, L, D]; pad: [B, L] bool with True=pad
        valid = (~pad).float().unsqueeze(-1)
        return (z * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

    def _mean_pool_concat(self, pep_seq, pep_str, pep_pad, mhc_seq, mhc_str, mhc_pad):
        return torch.cat([
            self._masked_mean(pep_seq, pep_pad),
            self._masked_mean(pep_str, pep_pad),
            self._masked_mean(mhc_seq, mhc_pad),
            self._masked_mean(mhc_str, mhc_pad),
        ], dim=1)

    def forward(self, batch):
        cls_out = self.forward_embedding(batch)
        return self.head(cls_out).squeeze(-1)

    def forward_repr(self, cls_out):
        """Return the contrastive-loss target representation.

        For aggregator='pool' with an MLP head, this is the post-hidden activation
        (Linear+GELU+Dropout output), i.e. the input to the final classification Linear.
        For aggregator='fusion' (Linear head), there's no hidden layer, so we use
        the fusion CLS embedding (cls_out itself).
        """
        if self.aggregator_type == "pool" and isinstance(self.head, nn.Sequential):
            # head = [Linear(in_dim, hidden), GELU, Dropout, Linear(hidden, 1)]
            # take everything except the final Linear (i.e. layers up to & including Dropout)
            return self.head[:-1](cls_out)
        return cls_out

    def _wt_subbatch(self, batch):
        """Build a sub-batch of wildtype features keyed without the `wt_` prefix
        so it can be passed through `forward_embedding`."""
        return {k: batch[f"wt_{k}"] for k in _WT_FORWARD_KEYS if f"wt_{k}" in batch}

    def _cw_loss(self, z_c, z_w, labels):
        """Cancer-vs-Wildtype contrastive loss (eqs 14, 15).

        L2-normalizes each row, then:
          L_sim   = (1/N) [ Σ_i ((Z_C Z_W^T)_ii - y_i)^2 + λ_off · Σ_{i≠j} ((Z_C Z_W^T)_ij)^2 ]
          L_indep = (1/D) [ Σ_i ((Z_C^T Z_W)_ii - 1)^2  + λ_off · Σ_{i≠j} ((Z_C^T Z_W)_ij)^2 ]
        Returns (l_sim, l_indep).
        """
        zc = F.normalize(z_c, dim=1)
        zw = F.normalize(z_w, dim=1)
        N, D = zc.shape
        lam = self.lambda_off_diag

        S = zc @ zw.t()
        diag_S = torch.diagonal(S)
        off_S = S - torch.diag_embed(diag_S)
        l_sim = ((diag_S - labels.float()) ** 2).sum() + lam * (off_S ** 2).sum()
        l_sim = l_sim / N

        C = zc.t() @ zw
        diag_C = torch.diagonal(C)
        off_C = C - torch.diag_embed(diag_C)
        l_indep = ((diag_C - 1.0) ** 2).sum() + lam * (off_C ** 2).sum()
        l_indep = l_indep / D

        return l_sim, l_indep

    def training_step(self, batch, batch_idx):
        cls_mut = self.forward_embedding(batch)
        logits = self.head(cls_mut).squeeze(-1)
        labels = batch["immunogenicity"]
        if self.loss_type == "aucm":
            # AUCMLoss expects sigmoid'd predictions in [0, 1].
            scores = torch.sigmoid(logits)
            loss_cls = self.train_loss(scores, labels)
        else:
            loss_cls = self.train_loss(logits, labels)

        if self.contrastive_enabled and "wt_peptide_pad_mask" in batch:
            wt_batch = self._wt_subbatch(batch)
            cls_wt = self.forward_embedding(wt_batch)
            z_mut = self.forward_repr(cls_mut)
            z_wt = self.forward_repr(cls_wt)
            l_sim, l_indep = self._cw_loss(z_mut, z_wt, labels)
            l_cw = l_sim + l_indep
            loss = loss_cls + self.lambda_cw * l_cw
            self.log("train/loss_cls", loss_cls, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/l_sim", l_sim, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/l_indep", l_indep, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/l_cw", l_cw, on_step=True, on_epoch=True, sync_dist=True)
        else:
            loss = loss_cls

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch["immunogenicity"]
        loss = self.eval_loss(logits, labels)
        scores = torch.sigmoid(logits)
        self.val_auroc.update(scores, labels.long())
        self.val_auprc.update(scores, labels.long())
        self._val_scores.append(scores.detach().cpu())
        self._val_labels.append(labels.detach().cpu().long())
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch["immunogenicity"]
        loss = self.eval_loss(logits, labels)
        scores = torch.sigmoid(logits)
        self.test_auroc.update(scores, labels.long())
        self.test_auprc.update(scores, labels.long())
        self._test_scores.append(scores.detach().cpu())
        self._test_labels.append(labels.detach().cpu().long())
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self._val_scores.clear()
        self._val_labels.clear()

    def on_validation_epoch_end(self):
        self.log("val/auroc", self.val_auroc.compute(), prog_bar=True, sync_dist=True)
        self.log("val/auprc", self.val_auprc.compute(), prog_bar=True, sync_dist=True)
        self.val_auroc.reset()
        self.val_auprc.reset()
        if self._val_scores:
            scores = torch.cat(self._val_scores)
            labels = torch.cat(self._val_labels)
            self._log_ppv_n("val", scores, labels)

    def on_test_epoch_start(self):
        self._test_scores.clear()
        self._test_labels.clear()

    def on_test_epoch_end(self):
        self.log("test/auroc", self.test_auroc.compute(), prog_bar=True, sync_dist=True)
        self.log("test/auprc", self.test_auprc.compute(), prog_bar=True, sync_dist=True)
        self.test_auroc.reset()
        self.test_auprc.reset()
        if self._test_scores:
            scores = torch.cat(self._test_scores)
            labels = torch.cat(self._test_labels)
            self._log_ppv_n("test", scores, labels)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.sequence_model.eval()
            self.structure_model.eval()
        return self

    def _log_ppv_n(self, prefix, scores, labels):
        n = int(labels.sum().item())
        if n == 0:
            return
        order = torch.argsort(scores, descending=True)
        sorted_labels = labels[order].float()
        ppv_at_n = sorted_labels[:n].sum().item() / n
        self.log(f"{prefix}/ppv@n", ppv_at_n, prog_bar=False, sync_dist=True)
        self.log(f"{prefix}/n_positives", float(n), prog_bar=False, sync_dist=True)

    def configure_optimizers(self):
        opt_cfg = getattr(self.model_cfg, "optimizer", None) or {}
        lr = getattr(opt_cfg, "lr", 1e-4) if not isinstance(opt_cfg, dict) else opt_cfg.get("lr", 1e-4)
        wd = getattr(opt_cfg, "weight_decay", 1e-6) if not isinstance(opt_cfg, dict) else opt_cfg.get("weight_decay", 1e-6)
        params = list(filter(lambda p: p.requires_grad, self.parameters()))
        if self.loss_type == "aucm":
            from libauc.optimizers import PESG
            # PESG co-optimizes the AUCMLoss inner params (a, b, alpha) with model params.
            return PESG(params, loss_fn=self.train_loss, lr=lr, mode="adam", weight_decay=wd, verbose=False)
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
