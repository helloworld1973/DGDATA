import torch


def get_params(alg, lr_decay1, lr_decay2, init_lr, nettype):
    if nettype == 'DGTRN-final':
        params = [
            {'params': alg.CVAE_encoder.parameters(), 'lr': lr_decay2 * init_lr},
            {'params': alg.CVAE_reparameterize.parameters(), 'lr': lr_decay2 * init_lr},
            {'params': alg.CVAE_decoder.parameters(), 'lr': lr_decay2 * init_lr},
            {'params': alg.classify_source.parameters(), 'lr': lr_decay2 * init_lr},
            {'params': alg.domains.parameters(), 'lr': lr_decay2 * init_lr},
            {'params': alg.temporal_states.parameters(), 'lr': lr_decay2 * init_lr}

        ]
        return params
    elif nettype == 'DGTRN-temporal':
        params = [
            {'params': alg.tCVAE_encoder.parameters(), 'lr': lr_decay2 * init_lr},
            {'params': alg.tCVAE_reparameterize.parameters(), 'lr': lr_decay2 * init_lr},
            {'params': alg.tCVAE_decoder.parameters(), 'lr': lr_decay2 * init_lr},
            {'params': alg.tclassify.parameters(), 'lr': lr_decay2 * init_lr},
            {'params': alg.tdomains.parameters(), 'lr': lr_decay2 * init_lr},
            {'params': alg.ttemporal_states.parameters(), 'lr': lr_decay2 * init_lr}
        ]
        return params
    elif nettype == 'DGTRN-all':
        params = [
            {'params': alg.featurizer.parameters(), 'lr': lr_decay1 * init_lr},
            {'params': alg.aCVAE_encoder.parameters(), 'lr': lr_decay2 * init_lr},
            {'params': alg.aCVAE_reparameterize.parameters(), 'lr': lr_decay2 * init_lr},
            {'params': alg.aCVAE_decoder.parameters(), 'lr': lr_decay2 * init_lr},
            {'params': alg.aclassify.parameters(), 'lr': lr_decay2 * init_lr},
            {'params': alg.adomains.parameters(), 'lr': lr_decay2 * init_lr}
        ]
        return params


def get_optimizer(alg, lr_decay1, lr_decay2, lr, optim_Adam_weight_decay, optim_Adam_beta, nettype):
    params = get_params(alg, lr_decay1, lr_decay2, lr, nettype=nettype)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=optim_Adam_weight_decay, betas=(optim_Adam_beta, 0.9))
    return optimizer
