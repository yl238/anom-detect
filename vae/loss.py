import torch
import torch.nn as nn

class VAELoss(nn.Module):
    """
            This criterion is an implementation of VAELoss
    """

    def __init__(self, size_average=False, kl_weight=1):
        super(VAELoss, self).__init__()
        self.size_average = size_average
        self.kl_weight = kl_weight

    def forward(self, recon_x, x, mu, logvar):
        """
        :param recon_x: generating images. [bsz, C, H, W]
        :param x: origin images. [bsz, C, H, W]
        :param mu: latent mean. [bsz, z_dim]
        :param logvar: latent log variance. [bsz, z_dim]
        :return loss, loss_details.
            loss: a scalar. negative of elbo
            loss_details: {'KL': KL, 'reconst_logp': -reconst_err}
        """
        bsz = x.shape[0]
        reconst_err = (x - recon_x).pow(2).reshape(bsz, -1)
        reconst_err = 0.5 * torch.sum(reconst_err, dim=-1)

        # KL(q || p) = -log_sigma + sigma^2/2 + mu^2/2 - 1/2
        KL = (-logvar + logvar.exp() + mu.pow(2) - 1) * 0.5
        KL = torch.sum(KL, dim=-1)
        if self.size_average:
            KL = torch.mean(KL)
            reconst_err = torch.mean(reconst_err)
        else:
            KL = torch.sum(KL)
            reconst_err = torch.sum(reconst_err)
        loss = reconst_err + self.kl_weight * KL
        return loss, {'KL': KL, 'reconst_logp': -reconst_err}

    def forward_without_reduce(self, recon_x, x, mu, logvar):
        """
        This also compute the vae loss but it's without take mean or take sum
        :param recon_x: generating images. [bsz, C, H, W]
        :param x: origin images. [bsz, C, H, W]
        :param mu: latent mean. [bsz, z_dim]
        :param logvar: latent log variance. [bsz, z_dim]
        :return: losses. [bsz] and loss details
        """
        bsz = x.shape[0]
        reconst_err = (x - recon_x).pow(2).reshape(bsz, -1)
        reconst_err = 0.5 * torch.sum(reconst_err, dim=-1)

        # KL(q || p) = -log_sigma + sigma^2/2 + mu^2/2 - 1/2
        KL = (-logvar + logvar.exp() + mu.pow(2) - 1) * 0.5
        KL = torch.sum(KL, dim=-1)

        # [bsz]
        losses = reconst_err + self.kl_weight * KL
        return losses, {'KL': KL, 'reconst_logp': -reconst_err}
