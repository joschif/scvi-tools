from typing import Any, Callable, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.distributions import kl_divergence as kl

from scvi.module import VAE

from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi._types import LatentDataType
from scvi.module.base import LossOutput
from scvi.losses import DecomposedKLDivergence
from scvi.autotune._types import Tunable


# Disentangling VAE model
class DISVAE(VAE):
    """
    Variational auto-encoder model.

    This is an implementation of the scVI model described in :cite:p:`Lopez18`.

    Parameters
    ----------
    n_input
        Number of input genes
    n_cells
        Number of cells
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    mi_weight
        Weight of the Index-Code mutual information term.
    tc_weight
        Weight of the total correlation term.
    kld_weight
        Weight of the dimension-wise KL term.
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers in encoder/decoder. This option
        only applies when `n_layers` > 1. The covariates are concatenated to the input of subsequent hidden layers.
    use_layer_norm
        Whether to use layer norm in layers
    use_size_factor_key
        Use size_factor AnnDataField defined by the user as scaling factor in mean of conditional distribution.
        Takes priority over `use_observed_lib_size`.
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    latent_data_type
        None or the type of latent data.
    """

    def __init__(
        self,
        n_input: int,
        n_cells: int = 10000,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: Tunable[int] = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: str = "normal",
        mi_weight: float = 1.0,
        tc_weight: float = 1.0,
        kld_weight: float = 1.0,
        decompose_method: Literal["factor", "tc"] = "factor",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        var_activation: Optional[Callable] = None,
        latent_data_type: Optional[LatentDataType] = None,
    ):
        super().__init__(
            n_input,
            n_labels,
            n_batch,
            n_hidden,
            n_latent,
            n_layers,
            n_continuous_cov,
            n_cats_per_cov,
            dropout_rate,
            dispersion,
            log_variational,
            gene_likelihood,
            latent_distribution,
            encode_covariates,
            deeply_inject_covariates,
            use_batch_norm,
            use_layer_norm,
            use_size_factor_key,
            use_observed_lib_size,
            library_log_means,
            library_log_vars,
            var_activation,
            latent_data_type,
        )

        self.mi_weight = mi_weight
        self.tc_weight = tc_weight
        self.kld_weight = kld_weight
        self.decompose_method = decompose_method

        self.dkl = DecomposedKLDivergence(data_size=n_cells)

    # Redefine loss to dientangeling loss
    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        """Computes the loss function for the model."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        kl_components_z = self.dkl(
            inference_outputs["qz"], generative_outputs["pz"], inference_outputs["z"]
        )
        if not self.use_observed_lib_size:
            kl_divergence_l = kl(
                inference_outputs["ql"],
                generative_outputs["pl"],
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.tensor(0.0, device=x.device)

        reconst_loss = -generative_outputs["px"].log_prob(x).sum(-1)

        if self.decompose_method == "factor":
            loss_z = (
                self.kld_weight * kl_components_z["kld"]
                + self.tc_weight * kl_components_z["tc"]
            )
        else:
            loss_z = (
                self.kld_weight * kl_components_z["dw_kld"]
                + self.tc_weight * kl_components_z["tc"]
                + self.mi_weight * kl_components_z["mi"]
            )

        loss_for_warmup = loss_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * loss_for_warmup + kl_local_no_warmup

        loss = torch.mean(reconst_loss + weighted_kl_local)

        kl_local = dict(
            kl_divergence_l=kl_divergence_l, kl_divergence_z=kl_components_z["kld"]
        )
        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_local,
            extra_metrics=dict(
                {
                    "kld": kl_components_z["kld"].mean(),
                    "tc": kl_components_z["tc"].mean(),
                    "mi": kl_components_z["mi"].mean(),
                    "dw_kld": kl_components_z["dw_kld"].mean(),
                }
            ),
        )
