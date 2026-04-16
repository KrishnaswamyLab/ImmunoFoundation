import os
from typing import List, Optional

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class MLPDataset(Dataset):
    """Simple dataset to load CSV rows and return numeric features + label.

    Behavior:
    - Attempts to infer label column (common names). If `label_col` provided, use it.
    - Uses `feature_cols` if provided, otherwise selects all numeric columns except label and id-like columns.
    - Returns dict with 'features' (float32 tensor) and 'label' (long tensor)
    """

    DEFAULT_LABEL_NAMES = ["label", "labels", "immunogenicity", "is_immunogenic", "y"]

    def __init__(self, csv_path: str, label_col: Optional[str] = None, feature_cols: Optional[List[str]] = None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        self.df = pd.read_csv(csv_path)
        if label_col is None:
            label_col = self._infer_label_col()
        if label_col is None:
            raise ValueError("Could not infer a label column. Please pass `label_col` explicitly.")
        self.label_col = label_col

        if feature_cols is None:
            # choose numeric dtypes except the label
            numeric_df = self.df.select_dtypes(include=["number"]).copy()
            if label_col in numeric_df.columns:
                numeric_df = numeric_df.drop(columns=[label_col])
            feature_cols = list(numeric_df.columns)
        if len(feature_cols) == 0:
            raise ValueError("No numeric feature columns found. Provide `feature_cols` explicitly.")

        self.feature_cols = feature_cols
        # drop rows with NaNs in selected cols
        keep_mask = self.df[self.feature_cols + [self.label_col]].notnull().all(axis=1)
        self.df = self.df.loc[keep_mask].reset_index(drop=True)

        # prepare numpy arrays for speed
        self.features = self.df[self.feature_cols].astype(np.float32).to_numpy()
        self.labels = self.df[self.label_col].to_numpy()

        # if labels are not integer, try to map
        if not np.issubdtype(self.labels.dtype, np.integer):
            unique = sorted(pd.unique(self.labels))
            self.label_map = {v: i for i, v in enumerate(unique)}
            self.labels = np.array([self.label_map[x] for x in self.labels], dtype=np.int64)
        else:
            self.label_map = None

    def _infer_label_col(self) -> Optional[str]:
        for name in self.DEFAULT_LABEL_NAMES:
            if name in self.df.columns:
                return name
        # try common suffixes
        for col in self.df.columns:
            if col.lower().startswith("label") or col.lower().endswith("label"):
                return col
        return None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx]).float()
        y = int(self.labels[idx])
        return {"features": x, "label": torch.tensor(y, dtype=torch.long)}


def collate_fn(batch):
    """Collate function to stack features and labels."""
    features = torch.stack([b["features"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return {"features": features, "labels": labels}
