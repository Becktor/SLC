import torch


def variance_score(a, b):
    return ((torch.pow(a, 2)).mean(1) - torch.pow(b, 2)).mean()


def mutual_information_score(p, p_hat):
    mi = (p * torch.log(p)).mean(1) - p_hat * torch.log(p_hat)
    return mi.mean()


def KL_divergence_score(z_logsigma, z_mu):
    kl_div = 0.5 * torch.sum(torch.exp(z_logsigma) + torch.square(z_mu) - z_logsigma - 1, axis=-1)
    return kl_div