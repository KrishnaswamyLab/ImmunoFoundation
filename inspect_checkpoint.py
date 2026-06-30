#!/usr/bin/env python3
"""Utility: print keys and tensor shapes inside a checkpoint file.

Helps determine which submodules can be loaded into a new model.
"""
import argparse
import torch


def summarize_state_dict(state_dict, top_n=200):
    keys = list(state_dict.keys())
    print(f"Total keys: {len(keys)}")
    print("First keys (name -> shape):")
    for k in keys[:top_n]:
        v = state_dict[k]
        try:
            shape = v.shape
        except Exception:
            shape = type(v)
        print(f"  {k} -> {shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', help='Path to checkpoint (.ckpt or .pth)')
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    else:
        sd = ckpt

    if not isinstance(sd, dict):
        print('Checkpoint content is not a mapping; top-level type:', type(sd))
        return

    summarize_state_dict(sd)


if __name__ == '__main__':
    main()
