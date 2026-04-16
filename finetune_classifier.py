#!/usr/bin/env python3
"""Project-aligned finetuning script.

This reuses the repository's `ImmunoMonomerDataset` and `ImmunoFoundationMonomerModule`.
It expects the CSV to include a `cif_path` column pointing to mmCIF files and an
`immunogenicity` column (0/1). It will attempt to load a provided checkpoint into
the backbone (best-effort) and finetune a classifier head.
"""
import os
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import yaml

from immunofoundation.data.components.ImmunoMonomerDataset import ImmunoMonomerDataset, custom_collate_mono
from immunofoundation.models.ImmunoFoundationMonomerModule import ImmunoFoundationMonomerModule
from immunofoundation.models.FinetuneClassifierModule import FinetuneClassifierModule


def make_data_cfg(csv_path, batch_size=16, train_size=0.8, num_workers=4):
    # Build a lightweight data_cfg object compatible with ImmunoMonomerDataset
    mask = SimpleNamespace(mask_rate=0.5, max_distance=8, max_neighbors=12)
    structure = SimpleNamespace(adj=True, k=15)
    data_cfg = SimpleNamespace(csv_path=csv_path, train_size=train_size, batch_size=batch_size, num_workers=num_workers, mono=True, mask=mask, structure=structure)
    return data_cfg


def main():
    # Load config from YAML file
    config_path = 'configs/finetune_classifier.yaml'  # Update path if needed
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Use config values
    csv_path = config['csv']
    out_dir = config['out_dir']
    max_epochs = config.get('max_epochs', 50)
    batch_size = config.get('batch_size', 8)
    num_workers = config.get('num_workers', 8)
    matmul_precision = config.get('matmul_precision', 'medium')
    bio_dim = config.get('bio_dim', 32)
    hidden_dims = config.get('hidden_dims', [512, 256, 128, 64, 32])
    class_weights = config.get('class_weights', None)

    os.makedirs(out_dir, exist_ok=True)

    # Optionally set float32 matmul precision to leverage Tensor Cores (perf vs numeric tradeoff)
    if matmul_precision is not None and matmul_precision != 'none':
        try:
            torch.set_float32_matmul_precision(matmul_precision)
            print(f"Set torch float32 matmul precision -> {matmul_precision}")
        except Exception as e:
            print(f"Warning: failed to set float32 matmul precision: {e}")

    data_cfg = make_data_cfg(csv_path, batch_size=batch_size, num_workers=num_workers)
    train_ds = ImmunoMonomerDataset(data_cfg, is_training=True)
    val_ds = ImmunoMonomerDataset(data_cfg, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_mono, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_mono, num_workers=num_workers)

    # Build backbone using defaults similar to training config
    seq = SimpleNamespace(model_type='esm', freeze_esm=True, aggregate=False, esm_dim=1280, rep_layer=33, esm_variant='esm2_t33_650M_UR50D', out_dim=32, n_heads=8, dim_ffn=256, n_layers=10)
    struct = SimpleNamespace(model_type='transformer', out_dim=32, n_heads=8, dim_ffn=256, n_layers=4)
    bio = SimpleNamespace(model_type='mlp', n_bio_prop=93, hidden_dim=64, out_dim=32, n_layers=4)
    model_cfg = SimpleNamespace(sequence=seq, structure=struct, bio_chem=bio)

    backbone = ImmunoFoundationMonomerModule(model_cfg)

    # Attempt to load checkpoint into backbone (non-fatal)
    checkpoint_path = config.get('checkpoint', None)
    if checkpoint_path is not None:
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            state = ckpt.get('state_dict', ckpt)
            backbone.load_state_dict(state, strict=False)
            print('Loaded checkpoint (partial) into backbone')
        except Exception as e:
            print('Warning: failed to load checkpoint into backbone:', e)

    # infer num_classes from training dataset labels (if present)
    sample = None
    try:
        sample = train_ds[0]
    except Exception:
        pass
    if sample is not None and 'label' in sample:
        # compute unique labels in small pass
        labels = []
        for i in range(min(len(train_ds), 1000)):
            try:
                labels.append(train_ds[i]['label'])
            except Exception:
                break
        num_classes = int(max(labels)) + 1 if len(labels) > 0 else 2
    else:
        num_classes = 2

    finetune = FinetuneClassifierModule(
        backbone,
        num_classes=num_classes,
        bio_dim=model_cfg.bio_chem.out_dim,
        hidden_dims=hidden_dims,
        class_weights=class_weights
    )

    checkpoint_cb = ModelCheckpoint(dirpath=out_dir, filename='finetune-{epoch:02d}-{val_loss:.4f}', save_top_k=3, monitor='val/loss', mode='min')
    # GPU-aware trainer: use a GPU if available
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = 1 if torch.cuda.is_available() else None
    # callback to print per-epoch metrics (loss/acc) to stdout for easy inspection
    class PrintMetricsCallback(Callback):
        def __init__(self, out_dir):
            super().__init__()
            self.out_dir = out_dir
            self.csv_path = os.path.join(self.out_dir, 'metrics.csv')
            # write header if not exists
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, 'w') as fh:
                    fh.write('epoch,train_loss,train_acc,val_loss,val_acc\n')

        def on_validation_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            # gather epoch and metrics with safe extraction
            epoch = int(trainer.current_epoch) if hasattr(trainer, 'current_epoch') else None
            def _safe_get(k):
                v = metrics.get(k, None)
                if v is None:
                    return None
                try:
                    if hasattr(v, 'item'):
                        return float(v.item())
                    elif isinstance(v, (int, float)):
                        return float(v)
                    else:
                        return float(v)
                except Exception:
                    return None

            train_loss = _safe_get('train/loss')
            train_acc = _safe_get('train/acc')
            val_loss = _safe_get('val/loss')
            val_acc = _safe_get('val/acc')

            out = {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/acc': train_acc,
                'val/loss': val_loss,
                'val/acc': val_acc,
            }
            print('Epoch metrics:', out)

            # append to CSV (use empty string for missing)
            with open(self.csv_path, 'a') as fh:
                fh.write(f"{epoch},{'' if train_loss is None else train_loss},{'' if train_acc is None else train_acc},{'' if val_loss is None else val_loss},{'' if val_acc is None else val_acc}\n")

    print_cb = PrintMetricsCallback(out_dir=out_dir)
    # Disable the default sanity validation steps (they can produce an extra validation run
    # before training starts which confuses single-epoch metrics printing). Set to 0 so
    # we only see the real validation at the end of each epoch.
    trainer = Trainer(max_epochs=max_epochs, callbacks=[checkpoint_cb, print_cb], accelerator=accelerator, devices=devices, num_sanity_val_steps=0)

    # Run training (Lightning will log metrics per epoch). The FinetuneClassifierModule
    # already logs train/loss and train/acc (on_step/on_epoch) and val/loss and val/acc (on_epoch).
    trainer.fit(finetune, train_loader, val_loader)

    # Optionally run test evaluation on a held-out test split using the best checkpoint
    run_test = config.get('test', False)
    test_csv = config.get('test_csv', None)
    if run_test:
        # determine test CSV path
        if test_csv is None:
            test_csv = os.path.join(out_dir, 'test.csv')
        if not os.path.exists(test_csv):
            print(f"Test requested but no test CSV found at {test_csv}. Skipping test run.")
        else:
            print('Building test dataloader from', test_csv)
            test_cfg = make_data_cfg(test_csv, batch_size=batch_size, num_workers=num_workers)
            test_ds = ImmunoMonomerDataset(test_cfg, is_training=False)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_mono, num_workers=num_workers)

            # find best checkpoint saved by the checkpoint callback
            best_ckpt = checkpoint_cb.best_model_path if getattr(checkpoint_cb, 'best_model_path', None) else None
            if not best_ckpt or not os.path.exists(best_ckpt):
                # fallback: pick the most recent finetune-*.ckpt in out_dir
                import glob
                ckpts = sorted(glob.glob(os.path.join(out_dir, 'finetune-*.ckpt')), key=os.path.getmtime)
                best_ckpt = ckpts[-1] if ckpts else None

            if best_ckpt is None:
                print('No checkpoint found to run test with. Skipping test run.')
            else:
                print('Running test with checkpoint:', best_ckpt)
                # load model from checkpoint (ensures weights match expected backbone signature)
                test_model = FinetuneClassifierModule.load_from_checkpoint(best_ckpt, backbone=backbone)
                test_model.eval()
                # run Lightning test and persist results
                test_res = trainer.test(test_model, dataloaders=test_loader, ckpt_path=None)
                try:
                    import json
                    out_path = os.path.join(out_dir, 'test_results.json')
                    with open(out_path, 'w') as fh:
                        json.dump(test_res, fh, indent=2)
                    print('Wrote test results to', out_path)
                except Exception as e:
                    print('Test finished but failed to write results:', e)


if __name__ == '__main__':
    main()
