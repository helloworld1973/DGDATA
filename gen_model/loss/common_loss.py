import torch
import torch.nn.functional as F


def kl_divergence_reserve_structure(mu, logvar, target_variance):
    # mu: the mean from the encoder's latent space
    # logvar: log variance from the encoder's latent space
    var = logvar.exp()
    kld = 0.5 * (var / target_variance - 1 - torch.log(var / target_variance))
    loss_mu = F.mse_loss(mu, torch.zeros_like(mu))
    return torch.sum(kld, dim=1).mean() + loss_mu

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.mean(torch.sum(entropy, dim=1))
    return entropy


def Entropylogits(input, redu='mean'):
    input_ = F.softmax(input, dim=1)
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    if redu == 'mean':
        entropy = torch.mean(torch.sum(entropy, dim=1))
    elif redu == 'None':
        entropy = torch.sum(entropy, dim=1)
    return entropy
