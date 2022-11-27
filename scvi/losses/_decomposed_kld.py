import math
import torch
from torch import logsumexp
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl


# Probability distribution utils
def matrix_log_density_gaussian(x, mean, stdev):
    """Calculates log density of a Gaussian for all combination of batch pairs of
    `x` and `mean`. I.e. return tensor of shape `(batch_size, batch_size, dim)`.

    Arguments:
        x: Float value at which to compute the density. Shape: (batch_size, dim).
        mean: Float value indicating the mean. Shape: (batch_size, dim).
        stdev: Float value indicating the standard deviation. Shape: (batch_size, dim).
        batch_size: Integer indicating the batch size.
    Returns:
        log_density: Log density of a Gaussian. Shape: (batch_size, batch_size, dim).
    """
    x = x.unsqueeze(1)
    mean = mean.unsqueeze(0)
    stdev = stdev.unsqueeze(0)
    return log_density_gaussian(x, mean, stdev)


def log_density_gaussian(x, mean, stdev):
    """Calculates log density of a Gaussian.

    Arguments:
        x: Float value at which to compute the density.
        mean: Float value indicating the mean.
        stdev: Float value indicating the log variance.
    Returns:
        log_density: Log density of a Gaussian.
    """
    normal_dist = Normal(mean, stdev)
    log_density = normal_dist.log_prob(x)
    return log_density


# Implementation adapted from
# * https://github.com/YannDubs/disentangling-vae
# * https://github.com/julian-carpenter/beta-TCVAE
# * https://github.com/rtqichen/beta-tcvae
class DecomposedKLDivergence:
    r"""
    Calculates KL divergence between given and standard gaussian distributions.
    Implements three ways to calculate the KL divergence:
        1. Regular KLD as
            KL(p, q) = H(p, q) - H(p) = -\int p(x)log(q(x))dx - -\int p(x)log(p(x))dx
                = 0.5 * [log(|s2|/|s1|) - 1 + tr(s1/s2) + (m1-m2)^2/s2]
                = 0.5 * [-log(|s1|) - 1 + tr(s1) + m1^2] (if m2 = 0, s2 = 1)
        2. As proposed in [FactorVAE](https://arxiv.org/abs/1802.05983), which adds a
            total correlation (TC) term which can be weighted individually.
        3. As proposed in [Beta-TCVAE](https://arxiv.org/abs/1905.09837), which fully
            decomposes the KLD into TC, MI and dimension-wise KLD terms with
            individual weights.

    Parameters
    ----------
    data_size
        Number of cells in the training set.
    """

    def __init__(self, data_size: int = 1000):
        self.data_size = data_size

    def __call__(self, q, p, z):
        """
        Parameters
        ----------
        q
            A distribution.
        p
            A distribution.
        z
            Sample from the latent distribution.
        """

        # Calculate KLD between prior and latent distributions
        kl_divergence = kl(q, p).sum(dim=1)

        batch_size = torch.tensor(z.shape[0], dtype=torch.float32)
        norm_const = torch.log(batch_size * self.data_size)

        # Calculate log p(z)
        # Zero mean and unit variance -> prior
        zeros = torch.zeros_like(z)
        log_pz = torch.sum(log_density_gaussian(z, zeros, 1), dim=1)

        # Calculate log q(z|x)
        log_qz_cond_x = torch.sum(log_density_gaussian(z, q.loc, q.scale), dim=1)
        log_qz_prob = matrix_log_density_gaussian(z, q.loc, q.scale)

        log_qz = logsumexp(log_qz_prob.sum(2), dim=1, keepdim=False) - norm_const

        log_qz_prod = torch.sum(
            logsumexp(log_qz_prob, dim=1, keepdim=False) - norm_const,
            dim=1,
            keepdim=False,
        )

        # Calculate the total correlation term
        # TC[z] = KL[q(z)||\prod_i z_i]
        total_correlation = log_qz - log_qz_prod
        # Calculate the mutual information term
        # MI[z] = KL[q(z)||\sum_i z_i]
        mutual_information = log_qz_cond_x - log_qz
        # Calculate the dimension-wise KL term
        # KL[z] = KL[q(z)||p(z)]
        dw_kl_divergence = log_qz_prod - log_pz

        return dict(
            kld=kl_divergence,
            tc=total_correlation,
            mi=mutual_information,
            dw_kld=dw_kl_divergence,
        )
