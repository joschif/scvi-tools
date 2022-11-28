import math
import torch
from scipy.special import gamma
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl


def total_correlation(loc, scale, z):
    """Estimate the total correlation on a batch."""
    batch_size = torch.tensor(z.shape[0], dtype=torch.float32)
    norm_const = torch.log(batch_size * 1000)
    log_qz_prob = Normal(loc.unsqueeze(0), scale.unsqueeze(0)).log_prob(z.unsqueeze(1))
    # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
    # + constant) for each sample in the batch, which is a vector of size
    # [batch_size,].
    log_qz_product = torch.sum(
        torch.logsumexp(log_qz_prob, dim=1, keepdim=False) - norm_const,
        dim=1,
        keepdim=False,
    )
    # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
    # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
    log_qz = torch.logsumexp(log_qz_prob.sum(2), dim=1, keepdim=False) - norm_const
    return log_qz - log_qz_product


def hsic(z, s):
    def K_hcv(x1, x2, gamma=1.0):
        dist_table = x1.unsqueeze(0) - x2.unsqueeze(1)
        return torch.transpose(
            torch.exp(-gamma * torch.sum(dist_table**2, dim=2)), 0, 1
        )

    d_z = list(z.shape)[1]
    d_s = list(s.shape)[1]

    gz = 2 * gamma(0.5 * (d_z + 1)) / gamma(0.5 * d_z)
    gs = 2 * gamma(0.5 * (d_s + 1)) / gamma(0.5 * d_s)

    zz = K_hcv(z, z, gamma=1.0 / (2.0 * gz))
    ss = K_hcv(s, s, gamma=1.0 / (2.0 * gs))

    hsic = 0
    hsic += torch.mean(zz * ss)
    hsic += torch.mean(zz) * torch.mean(ss)
    hsic -= 2 * torch.mean(torch.mean(zz, axis=1) * torch.mean(ss, axis=1))
    return hsic.sqrt()
