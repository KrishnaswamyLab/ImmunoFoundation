###########################################################################################
# Radial basis and cutoff functions for edge embeddings.
# Adapted from: https://github.com/VirtualProteins/GNN_UNet
# Original authors: Ilyes Batatia, Gregor Simm (MIT License)
###########################################################################################

import numpy as np
import torch


class BesselBasis(torch.nn.Module):
    """
    Radial Bessel basis functions.
    Klicpera et al., Directional Message Passing for Molecular Graphs, ICLR 2020. Eq. (7)
    """

    def __init__(self, r_max: float, num_basis: int = 8, trainable: bool = False):
        super().__init__()
        bessel_weights = (
            np.pi
            / r_max
            * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numerator = torch.sin(self.bessel_weights * x)
        return self.prefactor * (numerator / x)


class PolynomialCutoff(torch.nn.Module):
    """
    Smooth polynomial envelope cutoff.
    Klicpera et al., Directional Message Passing for Molecular Graphs, ICLR 2020. Eq. (8)
    """

    def __init__(self, r_max: float, p: int = 6):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        envelope = (
            1.0
            - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * torch.pow(x / self.r_max, self.p)
            + self.p * (self.p + 2.0) * torch.pow(x / self.r_max, self.p + 1)
            - (self.p * (self.p + 1.0) / 2) * torch.pow(x / self.r_max, self.p + 2)
        )
        return envelope * (x < self.r_max).type(torch.get_default_dtype())


class RadialEmbeddingBlock(torch.nn.Module):
    """Combines Bessel basis with polynomial cutoff for radial edge embeddings."""

    def __init__(self, r_max: float, num_bessel: int, num_polynomial_cutoff: int):
        super().__init__()
        self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel

    def forward(self, edge_lengths: torch.Tensor) -> torch.Tensor:
        bessel = self.bessel_fn(edge_lengths)
        cutoff = self.cutoff_fn(edge_lengths)
        return bessel * cutoff
