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
        logvar: Float value indicating the standard deviation. Shape: (batch_size, dim).
        batch_size: Integer indicating the batch size.
    Returns:
        log_density: Log density of a Gaussian. Shape: (batch_size, batch_size, dim).
    """
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
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
    x = torch.tensor(x, dtype=torch.float32)
    mean = torch.tensor(mean, dtype=torch.float32)
    stdev = torch.tensor(stdev, dtype=torch.float32)
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

        log_pz, log_qz, log_qz_prod, log_qz_cond_x = self._get_kld_components(
            q.loc, q.scale, z
        )
        # Calculate the total correlation term
        # TC[z] = KL[q(z)||\prod_i z_i]
        total_correlation = torch.mean(log_qz - log_qz_prod)
        mutual_information = torch.mean(log_qz_cond_x - log_qz)
        # dw_kl_divergence is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dw_kl_divergence = torch.mean(log_qz_prod - log_pz)

        return dict(
            kld=kl_divergence,
            tc=total_correlation,
            mi=mutual_information,
            dw_kld=dw_kl_divergence,
        )

    def _get_kld_components(self, loc, scale, z):
        batch_size = z.shape[0]
        norm_const = torch.log(batch_size * self.data_size)

        # Calculate log p(z)
        # Zero mean and unit variance -> prior
        zeros = torch.zeros_like(z)
        log_pz = torch.sum(log_density_gaussian(z, zeros, 1), axis=1)

        # Calculate log q(z|x)
        log_qz_cond_x = torch.sum(log_density_gaussian(z, loc, scale), axis=1)
        log_qz_prob = matrix_log_density_gaussian(z, loc, scale)

        log_qz = (
            logsumexp(
                torch.sum(log_qz_prob, axis=2, keepdims=False), axis=1, keepdims=False
            )
            - norm_const
        )

        log_qz_prod = torch.sum(
            logsumexp(log_qz_prob, axis=1, keepdims=False) - norm_const,
            axis=1,
            keepdims=False,
        )

        return log_pz, log_qz, log_qz_prod, log_qz_cond_x
