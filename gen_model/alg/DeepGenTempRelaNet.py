import torch
from torch import nn, mean
import torch.nn.functional as F
import numpy as np
from gen_model.loss.common_loss import kl_divergence_reserve_structure
from gen_model.network.Adver_network import ReverseLayerF, Discriminator
from gen_model.network.common_network import cvae_encoder, cvae_decoder, cvae_reparameterize, linear_classifier
from gen_model.network.feature_extraction_network import CNN_Feature_Extraction_Network, \
    Transformer_Feature_Extraction_Network
from scipy.spatial.distance import cdist


class DeepGenTempRelaNet(nn.Module):
    def __init__(self, conv1_in_channels, conv1_out_channels, conv2_out_channels, kernel_size_num, in_features_size,
                 hidden_size, dis_hidden, num_class, num_temporal_states, reverseLayer_latent_domain_alpha, variance,
                 alpha, beta, gamma, delta, epsilon):
        super(DeepGenTempRelaNet, self).__init__()

        self.hidden_size = hidden_size
        self.num_class = num_class
        self.num_temporal_states = num_temporal_states
        self.ReverseLayer_latent_domain_alpha = reverseLayer_latent_domain_alpha
        self.Variance = variance

        self.Alpha = alpha
        self.Beta = beta
        self.Gamma = gamma
        self.Delta = delta
        self.Epsilon = epsilon

        self.conv1_in_channels = conv1_in_channels
        self.conv1_out_channels = conv1_out_channels
        self.conv2_out_channels = conv2_out_channels
        self.kernel_size_num = kernel_size_num
        self.in_features_size = in_features_size
        self.dis_hidden = dis_hidden

        self.featurizer = CNN_Feature_Extraction_Network(self.conv1_in_channels, self.conv1_out_channels,
                                                         self.conv2_out_channels, self.kernel_size_num,
                                                         self.in_features_size)

        self.aCVAE_encoder = cvae_encoder(self.in_features_size, self.hidden_size)
        self.aCVAE_reparameterize = cvae_reparameterize()
        self.aCVAE_decoder = cvae_decoder(self.in_features_size, self.hidden_size)
        self.aclassify = linear_classifier(self.hidden_size, 2 * self.num_class * self.num_temporal_states)
        self.adomains = linear_classifier(self.hidden_size, 2)

        self.tCVAE_encoder = cvae_encoder(self.in_features_size, self.hidden_size)
        self.tCVAE_reparameterize = cvae_reparameterize()
        self.tCVAE_decoder = cvae_decoder(self.in_features_size, self.hidden_size)
        self.tclassify = Discriminator(self.hidden_size, self.dis_hidden, 2 * self.num_class)
        self.tdomains = Discriminator(self.hidden_size, self.dis_hidden, 2)
        self.ttemporal_states = linear_classifier(self.hidden_size, self.num_temporal_states)

        self.CVAE_encoder = cvae_encoder(self.in_features_size, self.hidden_size)
        self.CVAE_reparameterize = cvae_reparameterize()
        self.CVAE_decoder = cvae_decoder(self.in_features_size, self.hidden_size)
        self.classify_source = linear_classifier(self.hidden_size, self.num_class)
        self.domains = Discriminator(self.hidden_size, self.dis_hidden, 2)
        self.temporal_states = linear_classifier(self.hidden_size, self.num_temporal_states)

    def update_a(self, minibatches, opt):
        all_x = minibatches[0].float()
        all_c = minibatches[1].long()
        all_ts = minibatches[2].long()
        all_d = minibatches[3].long()

        all_y = all_ts * 2 * self.num_class + all_c

        all_x_after_fe = self.featurizer(all_x)
        all_mu, all_logvar = self.aCVAE_encoder(all_x_after_fe)
        all_z = self.aCVAE_reparameterize(all_mu, all_logvar)
        all_x_recon = self.aCVAE_decoder(all_z)
        predict_all_class_labels = self.aclassify(all_z)
        predict_all_domain_labels = self.adomains(all_z)

        CLASS_L = F.cross_entropy(predict_all_class_labels, all_y)
        DOMAIN_L = F.cross_entropy(predict_all_domain_labels, all_d)
        RECON_L = mean((all_x_recon - all_x_after_fe) ** 2)
        KLD_L = kl_divergence_reserve_structure(all_mu, all_logvar, self.Variance)

        loss = self.Alpha * RECON_L + self.Beta * KLD_L + self.Gamma * CLASS_L + self.Delta * DOMAIN_L
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'total': loss.item(), 'reconstruct': RECON_L.item(), 'KL': KLD_L.item(),
                'classes': CLASS_L.item(), 'domains': DOMAIN_L.item()}

    def update_t(self, minibatches, opt):
        all_x = minibatches[0].float()
        all_c = minibatches[1].long()
        all_ts = minibatches[2].long()
        all_d = minibatches[3].long()

        all_x_after_fe = self.featurizer(all_x)
        all_mu, all_logvar = self.tCVAE_encoder(all_x_after_fe)
        all_z = self.tCVAE_reparameterize(all_mu, all_logvar)
        all_x_recon = self.tCVAE_decoder(all_z)

        predict_all_temporal_state_labels = self.ttemporal_states(all_z)
        TEMPORAL_L = F.cross_entropy(predict_all_temporal_state_labels, all_ts)

        disc_class_in1 = ReverseLayerF.apply(all_z, self.ReverseLayer_latent_domain_alpha)
        disc_class_out1 = self.tclassify(disc_class_in1)
        disc_CLASS_L = F.cross_entropy(disc_class_out1, all_c)

        disc_d_in1 = ReverseLayerF.apply(all_z, self.ReverseLayer_latent_domain_alpha)
        disc_d_out1 = self.tdomains(disc_d_in1)
        disc_DOMAIN_L = F.cross_entropy(disc_d_out1, all_d)

        RECON_L = mean((all_x_recon - all_x_after_fe) ** 2)
        KLD_L = kl_divergence_reserve_structure(all_mu, all_logvar, self.Variance)

        loss = self.Alpha * RECON_L + self.Beta * KLD_L + self.Gamma * disc_CLASS_L + self.Delta * disc_DOMAIN_L + self.Epsilon * TEMPORAL_L
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'total': loss.item(), 'reconstruct': RECON_L.item(), 'KL': KLD_L.item(),
                'disc_classes': disc_CLASS_L.item(), 'disc_domains': disc_DOMAIN_L.item(),
                'temporal': TEMPORAL_L.item()}

    def update(self, ST_data, S_data, opt):
        all_x = ST_data[0].float()
        all_c = ST_data[1].long()
        all_ts = ST_data[2].long()
        all_d = ST_data[3].long()

        all_x_after_fe = self.featurizer(all_x)
        all_mu, all_logvar = self.CVAE_encoder(all_x_after_fe)
        all_z = self.CVAE_reparameterize(all_mu, all_logvar)
        all_x_recon = self.CVAE_decoder(all_z)
        RECON_L = mean((all_x_recon - all_x_after_fe) ** 2)
        KLD_L = kl_divergence_reserve_structure(all_mu, all_logvar, self.Variance)

        predict_all_temporal_state_labels = self.temporal_states(all_z)
        TEMPORAL_L = F.cross_entropy(predict_all_temporal_state_labels, all_ts)

        disc_d_in1 = ReverseLayerF.apply(all_z, self.ReverseLayer_latent_domain_alpha)
        disc_d_out1 = self.domains(disc_d_in1)
        disc_DOMAIN_L = F.cross_entropy(disc_d_out1, all_d)

        S_x = S_data[0].float()
        S_c = S_data[1].long()
        S_ts = S_data[2].long()
        S_d = ST_data[3].long()
        S_x_after_fe = self.featurizer(S_x)
        S_mu, S_logvar = self.CVAE_encoder(S_x_after_fe)
        S_z = self.CVAE_reparameterize(S_mu, S_logvar)
        predict_S_class_labels = self.classify_source(S_z)
        S_CLASS_L = F.cross_entropy(predict_S_class_labels, S_c)

        loss = self.Alpha * RECON_L + self.Beta * KLD_L + self.Gamma * S_CLASS_L + self.Delta * disc_DOMAIN_L + self.Epsilon * TEMPORAL_L
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'total': loss.item(), 'reconstruct': RECON_L.item(), 'KL': KLD_L.item(),
                'source_classes': S_CLASS_L.item(), 'disc_domains': disc_DOMAIN_L.item(),
                'temporal': TEMPORAL_L.item()}

    def GPU_set_tlabel(self, ST_torch_loader, S_torch_loader, T_torch_loader, device):
        self.tCVAE_encoder.eval()
        self.tCVAE_reparameterize.eval()
        self.ttemporal_states.eval()
        self.featurizer.eval()

        start_test = True
        with torch.no_grad():
            iter_test = iter(ST_torch_loader)
            for _ in range(len(ST_torch_loader)):
                data = next(iter_test)
                inputs = data[0]
                inputs = inputs.float()
                index = data[-1]
                all_mu, all_logvar = self.tCVAE_encoder(self.featurizer(inputs))
                all_z = self.tCVAE_reparameterize(all_mu, all_logvar)
                outputs = self.ttemporal_states(all_z)
                if start_test:
                    all_fea = all_z.float().cpu()
                    all_output = outputs.float().cpu()
                    all_index = index
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, all_z.float().cpu()), 0)
                    all_output = torch.cat(
                        (all_output, outputs.float().cpu()), 0)
                    all_index = np.hstack((all_index, index))
        all_output = nn.Softmax(dim=1)(all_output)

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)

        for _ in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, initc, 'cosine')
            pred_label = dd.argmin(axis=1)

        ST_torch_loader.dataset.tensors = (
            ST_torch_loader.dataset.tensors[0].to(device), ST_torch_loader.dataset.tensors[1].to(device),
            torch.tensor(pred_label).to(device),
            ST_torch_loader.dataset.tensors[3].to(device), ST_torch_loader.dataset.tensors[4].to(device))

        S_torch_loader.dataset.tensors = (
            S_torch_loader.dataset.tensors[0].to(device), S_torch_loader.dataset.tensors[1].to(device),
            torch.tensor(pred_label[0:len(S_torch_loader.dataset.tensors[0])]).to(device),
            S_torch_loader.dataset.tensors[3].to(device), S_torch_loader.dataset.tensors[4].to(device))

        T_torch_loader.dataset.tensors = (
            T_torch_loader.dataset.tensors[0].to(device), T_torch_loader.dataset.tensors[1].to(device),
            torch.tensor(pred_label[
                         len(S_torch_loader.dataset.tensors[0]): len(S_torch_loader.dataset.tensors[0]) + len(
                             T_torch_loader.dataset.tensors[0])]).to(device),
            T_torch_loader.dataset.tensors[3].to(device), T_torch_loader.dataset.tensors[4].to(device))

        self.tCVAE_encoder.train()
        self.tCVAE_reparameterize.train()
        self.ttemporal_states.train()
        self.featurizer.train()

    def compute_weighted_features(self, features, weights, temporal_lags):
        # Ensure the time series is long enough
        if len(features) < temporal_lags+1:
            raise ValueError("Time series is too short!")

        new_features = []
        for k in range(temporal_lags):
            new_features.append(features[k])
        # Start from the 5th time step
        for t in range(temporal_lags, len(features)):
            weighted_sum = sum(features[t - i - 1] * weights[i] for i in range(temporal_lags))
            new_feature = features[t] + weighted_sum
            new_features.append(new_feature)

        return new_features

    def GPU_set_dlabel_temporal_relation(self, ST_torch_loader, S_torch_loader, T_torch_loader,
                                         temporal_relation_weight, temporal_lags, device):
        self.featurizer.eval()
        self.tCVAE_encoder.eval()
        self.tCVAE_reparameterize.eval()
        self.ttemporal_states.eval()

        start_test = True
        with torch.no_grad():
            iter_test = iter(ST_torch_loader)
            for _ in range(len(ST_torch_loader)):
                data = next(iter_test)
                inputs = data[0]
                inputs = inputs.float()
                index = data[-1]
                index = index.cpu()
                all_c = data[1].long()
                all_c = all_c.cpu()
                all_ts = data[2].long()

                features = self.featurizer(inputs)
                features = features.cpu()
                labels = all_c

                # Create a dictionary to store features for each unique label
                features_dict = {}
                unique_labels = torch.unique(labels)
                for label in unique_labels:
                    mask = labels == label
                    features_dict[label.item()] = features[mask]
                # Iterate over the features_dict and apply the function
                new_features_dict = {}
                for label, time_series in features_dict.items():
                    new_features_dict[label] = self.compute_weighted_features(time_series, temporal_relation_weight,
                                                                              temporal_lags)

                # Extract tensors from dictionary and put them in a list
                lists_of_tensors = [v for k, v in new_features_dict.items()]
                # Concatenate tensors inside each list
                concatenated_tensors = [torch.stack(tensor_list, dim=0) for tensor_list in lists_of_tensors]
                # Concatenate the resulting tensors together
                combined_tensor = torch.cat(concatenated_tensors, dim=0)

                combined_tensor = combined_tensor.to(device)

                all_mu, all_logvar = self.tCVAE_encoder(combined_tensor)
                all_z = self.tCVAE_reparameterize(all_mu, all_logvar)
                outputs = self.ttemporal_states(all_z)
                feas = all_z

                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_index = index
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat(
                        (all_output, outputs.float().cpu()), 0)
                    all_index = np.hstack((all_index, index))
        all_output = nn.Softmax(dim=1)(all_output)

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)

        for _ in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, initc, 'cosine')
            pred_label = dd.argmin(axis=1)

        ST_torch_loader.dataset.tensors = (
            ST_torch_loader.dataset.tensors[0].to(device), ST_torch_loader.dataset.tensors[1].to(device), torch.tensor(pred_label).to(device),
            ST_torch_loader.dataset.tensors[3].to(device), ST_torch_loader.dataset.tensors[4].to(device))

        S_torch_loader.dataset.tensors = (
            S_torch_loader.dataset.tensors[0].to(device), S_torch_loader.dataset.tensors[1].to(device),
            torch.tensor(pred_label[0:len(S_torch_loader.dataset.tensors[0])]).to(device),
            S_torch_loader.dataset.tensors[3].to(device), S_torch_loader.dataset.tensors[4].to(device))

        T_torch_loader.dataset.tensors = (
            T_torch_loader.dataset.tensors[0].to(device), T_torch_loader.dataset.tensors[1].to(device),
            torch.tensor(pred_label[
                         len(S_torch_loader.dataset.tensors[0]): len(S_torch_loader.dataset.tensors[0]) + len(
                             T_torch_loader.dataset.tensors[0])]).to(device),
            T_torch_loader.dataset.tensors[3].to(device), T_torch_loader.dataset.tensors[4].to(device))

        self.featurizer.train()
        self.tCVAE_encoder.train()
        self.tCVAE_reparameterize.train()
        self.ttemporal_states.train()

    def set_tlabel(self, ST_torch_loader, S_torch_loader, T_torch_loader):
        self.tCVAE_encoder.eval()
        self.tCVAE_reparameterize.eval()
        self.ttemporal_states.eval()
        self.featurizer.eval()

        start_test = True
        with torch.no_grad():
            iter_test = iter(ST_torch_loader)
            for _ in range(len(ST_torch_loader)):
                data = next(iter_test)
                inputs = data[0]
                inputs = inputs.float()
                index = data[-1]
                all_mu, all_logvar = self.tCVAE_encoder(self.featurizer(inputs))
                all_z = self.tCVAE_reparameterize(all_mu, all_logvar)
                outputs = self.ttemporal_states(all_z)
                if start_test:
                    all_fea = all_z.float().cpu()
                    all_output = outputs.float().cpu()
                    all_index = index
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, all_z.float().cpu()), 0)
                    all_output = torch.cat(
                        (all_output, outputs.float().cpu()), 0)
                    all_index = np.hstack((all_index, index))
        all_output = nn.Softmax(dim=1)(all_output)

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)

        for _ in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, initc, 'cosine')
            pred_label = dd.argmin(axis=1)

        ST_torch_loader.dataset.tensors = (
            ST_torch_loader.dataset.tensors[0], ST_torch_loader.dataset.tensors[1], torch.tensor(pred_label),
            ST_torch_loader.dataset.tensors[3], ST_torch_loader.dataset.tensors[4])

        S_torch_loader.dataset.tensors = (
            S_torch_loader.dataset.tensors[0], S_torch_loader.dataset.tensors[1],
            torch.tensor(pred_label[0:len(S_torch_loader.dataset.tensors[0])]),
            S_torch_loader.dataset.tensors[3], S_torch_loader.dataset.tensors[4])

        T_torch_loader.dataset.tensors = (
            T_torch_loader.dataset.tensors[0], T_torch_loader.dataset.tensors[1],
            torch.tensor(pred_label[
                         len(S_torch_loader.dataset.tensors[0]): len(S_torch_loader.dataset.tensors[0]) + len(
                             T_torch_loader.dataset.tensors[0])]),
            T_torch_loader.dataset.tensors[3], T_torch_loader.dataset.tensors[4])

        self.tCVAE_encoder.train()
        self.tCVAE_reparameterize.train()
        self.ttemporal_states.train()
        self.featurizer.train()

    def predict(self, x):
        mu, _ = self.CVAE_encoder(self.featurizer(x))
        return self.classify_source(mu), mu


class DeepGenTempRelaNet_Transformer(nn.Module):
    def __init__(self, embed_size, heads, num_layers, in_features_size,
                 hidden_size, dis_hidden, num_class, num_temporal_states, reverseLayer_latent_domain_alpha, variance,
                 alpha, beta, gamma, delta, epsilon):
        super(DeepGenTempRelaNet_Transformer, self).__init__()

        self.hidden_size = hidden_size
        self.num_class = num_class
        self.num_temporal_states = num_temporal_states
        self.ReverseLayer_latent_domain_alpha = reverseLayer_latent_domain_alpha
        self.Variance = variance

        self.Alpha = alpha
        self.Beta = beta
        self.Gamma = gamma
        self.Delta = delta
        self.Epsilon = epsilon

        self.embed_size = embed_size
        self.heads = heads
        self.num_layers = num_layers
        self.in_features_size = in_features_size
        self.dis_hidden = dis_hidden

        self.featurizer = Transformer_Feature_Extraction_Network(self.embed_size, self.heads, self.num_layers, self.in_features_size)

        self.aCVAE_encoder = cvae_encoder(self.in_features_size, self.hidden_size)
        self.aCVAE_reparameterize = cvae_reparameterize()
        self.aCVAE_decoder = cvae_decoder(self.in_features_size, self.hidden_size)
        self.aclassify = linear_classifier(self.hidden_size, 2 * self.num_class * self.num_temporal_states)
        self.adomains = linear_classifier(self.hidden_size, 2)

        self.tCVAE_encoder = cvae_encoder(self.in_features_size, self.hidden_size)
        self.tCVAE_reparameterize = cvae_reparameterize()
        self.tCVAE_decoder = cvae_decoder(self.in_features_size, self.hidden_size)
        self.tclassify = Discriminator(self.hidden_size, self.dis_hidden, 2 * self.num_class)
        self.tdomains = Discriminator(self.hidden_size, self.dis_hidden, 2)
        self.ttemporal_states = linear_classifier(self.hidden_size, self.num_temporal_states)

        self.CVAE_encoder = cvae_encoder(self.in_features_size, self.hidden_size)
        self.CVAE_reparameterize = cvae_reparameterize()
        self.CVAE_decoder = cvae_decoder(self.in_features_size, self.hidden_size)
        self.classify_source = linear_classifier(self.hidden_size, self.num_class)
        self.domains = Discriminator(self.hidden_size, self.dis_hidden, 2)
        self.temporal_states = linear_classifier(self.hidden_size, self.num_temporal_states)

    def update_a(self, minibatches, opt):
        all_x = minibatches[0].float()
        all_c = minibatches[1].long()
        all_ts = minibatches[2].long()
        all_d = minibatches[3].long()

        all_y = all_ts * 2 * self.num_class + all_c

        all_x_after_fe = self.featurizer(all_x)
        all_mu, all_logvar = self.aCVAE_encoder(all_x_after_fe)
        all_z = self.aCVAE_reparameterize(all_mu, all_logvar)
        all_x_recon = self.aCVAE_decoder(all_z)
        predict_all_class_labels = self.aclassify(all_z)
        predict_all_domain_labels = self.adomains(all_z)

        CLASS_L = F.cross_entropy(predict_all_class_labels, all_y)
        DOMAIN_L = F.cross_entropy(predict_all_domain_labels, all_d)
        RECON_L = mean((all_x_recon - all_x_after_fe) ** 2)
        KLD_L = kl_divergence_reserve_structure(all_mu, all_logvar, self.Variance)

        loss = self.Alpha * RECON_L + self.Beta * KLD_L + self.Gamma * CLASS_L + self.Delta * DOMAIN_L
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'total': loss.item(), 'reconstruct': RECON_L.item(), 'KL': KLD_L.item(),
                'classes': CLASS_L.item(), 'domains': DOMAIN_L.item()}

    def update_t(self, minibatches, opt):
        all_x = minibatches[0].float()
        all_c = minibatches[1].long()
        all_ts = minibatches[2].long()
        all_d = minibatches[3].long()

        all_x_after_fe = self.featurizer(all_x)
        all_mu, all_logvar = self.tCVAE_encoder(all_x_after_fe)
        all_z = self.tCVAE_reparameterize(all_mu, all_logvar)
        all_x_recon = self.tCVAE_decoder(all_z)

        predict_all_temporal_state_labels = self.ttemporal_states(all_z)
        TEMPORAL_L = F.cross_entropy(predict_all_temporal_state_labels, all_ts)

        disc_class_in1 = ReverseLayerF.apply(all_z, self.ReverseLayer_latent_domain_alpha)
        disc_class_out1 = self.tclassify(disc_class_in1)
        disc_CLASS_L = F.cross_entropy(disc_class_out1, all_c)

        disc_d_in1 = ReverseLayerF.apply(all_z, self.ReverseLayer_latent_domain_alpha)
        disc_d_out1 = self.tdomains(disc_d_in1)
        disc_DOMAIN_L = F.cross_entropy(disc_d_out1, all_d)

        RECON_L = mean((all_x_recon - all_x_after_fe) ** 2)
        KLD_L = kl_divergence_reserve_structure(all_mu, all_logvar, self.Variance)

        loss = self.Alpha * RECON_L + self.Beta * KLD_L + self.Gamma * disc_CLASS_L + self.Delta * disc_DOMAIN_L + self.Epsilon * TEMPORAL_L
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'total': loss.item(), 'reconstruct': RECON_L.item(), 'KL': KLD_L.item(),
                'disc_classes': disc_CLASS_L.item(), 'disc_domains': disc_DOMAIN_L.item(),
                'temporal': TEMPORAL_L.item()}

    def update(self, ST_data, S_data, opt):
        all_x = ST_data[0].float()
        all_c = ST_data[1].long()
        all_ts = ST_data[2].long()
        all_d = ST_data[3].long()

        all_x_after_fe = self.featurizer(all_x)
        all_mu, all_logvar = self.CVAE_encoder(all_x_after_fe)
        all_z = self.CVAE_reparameterize(all_mu, all_logvar)
        all_x_recon = self.CVAE_decoder(all_z)
        RECON_L = mean((all_x_recon - all_x_after_fe) ** 2)
        KLD_L = kl_divergence_reserve_structure(all_mu, all_logvar, self.Variance)

        predict_all_temporal_state_labels = self.temporal_states(all_z)
        TEMPORAL_L = F.cross_entropy(predict_all_temporal_state_labels, all_ts)

        disc_d_in1 = ReverseLayerF.apply(all_z, self.ReverseLayer_latent_domain_alpha)
        disc_d_out1 = self.domains(disc_d_in1)
        disc_DOMAIN_L = F.cross_entropy(disc_d_out1, all_d)

        S_x = S_data[0].float()
        S_c = S_data[1].long()
        S_ts = S_data[2].long()
        S_d = ST_data[3].long()
        S_x_after_fe = self.featurizer(S_x)
        S_mu, S_logvar = self.CVAE_encoder(S_x_after_fe)
        S_z = self.CVAE_reparameterize(S_mu, S_logvar)
        predict_S_class_labels = self.classify_source(S_z)
        S_CLASS_L = F.cross_entropy(predict_S_class_labels, S_c)

        loss = self.Alpha * RECON_L + self.Beta * KLD_L + self.Gamma * S_CLASS_L + self.Delta * disc_DOMAIN_L + self.Epsilon * TEMPORAL_L
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'total': loss.item(), 'reconstruct': RECON_L.item(), 'KL': KLD_L.item(),
                'source_classes': S_CLASS_L.item(), 'disc_domains': disc_DOMAIN_L.item(),
                'temporal': TEMPORAL_L.item()}

    def GPU_set_tlabel(self, ST_torch_loader, S_torch_loader, T_torch_loader, device):
        self.tCVAE_encoder.eval()
        self.tCVAE_reparameterize.eval()
        self.ttemporal_states.eval()
        self.featurizer.eval()

        start_test = True
        with torch.no_grad():
            iter_test = iter(ST_torch_loader)
            for _ in range(len(ST_torch_loader)):
                data = next(iter_test)
                inputs = data[0]
                inputs = inputs.float()
                index = data[-1]
                all_mu, all_logvar = self.tCVAE_encoder(self.featurizer(inputs))
                all_z = self.tCVAE_reparameterize(all_mu, all_logvar)
                outputs = self.ttemporal_states(all_z)
                if start_test:
                    all_fea = all_z.float().cpu()
                    all_output = outputs.float().cpu()
                    all_index = index
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, all_z.float().cpu()), 0)
                    all_output = torch.cat(
                        (all_output, outputs.float().cpu()), 0)
                    all_index = np.hstack((all_index, index))
        all_output = nn.Softmax(dim=1)(all_output)

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)

        for _ in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, initc, 'cosine')
            pred_label = dd.argmin(axis=1)

        ST_torch_loader.dataset.tensors = (
            ST_torch_loader.dataset.tensors[0].to(device), ST_torch_loader.dataset.tensors[1].to(device),
            torch.tensor(pred_label).to(device),
            ST_torch_loader.dataset.tensors[3].to(device), ST_torch_loader.dataset.tensors[4].to(device))

        S_torch_loader.dataset.tensors = (
            S_torch_loader.dataset.tensors[0].to(device), S_torch_loader.dataset.tensors[1].to(device),
            torch.tensor(pred_label[0:len(S_torch_loader.dataset.tensors[0])]).to(device),
            S_torch_loader.dataset.tensors[3].to(device), S_torch_loader.dataset.tensors[4].to(device))

        T_torch_loader.dataset.tensors = (
            T_torch_loader.dataset.tensors[0].to(device), T_torch_loader.dataset.tensors[1].to(device),
            torch.tensor(pred_label[
                         len(S_torch_loader.dataset.tensors[0]): len(S_torch_loader.dataset.tensors[0]) + len(
                             T_torch_loader.dataset.tensors[0])]).to(device),
            T_torch_loader.dataset.tensors[3].to(device), T_torch_loader.dataset.tensors[4].to(device))

        self.tCVAE_encoder.train()
        self.tCVAE_reparameterize.train()
        self.ttemporal_states.train()
        self.featurizer.train()

    def compute_weighted_features(self, features, weights, temporal_lags):
        # Ensure the time series is long enough
        if len(features) < temporal_lags+1:
            raise ValueError("Time series is too short!")

        new_features = []
        for k in range(temporal_lags):
            new_features.append(features[k])
        # Start from the 5th time step
        for t in range(temporal_lags, len(features)):
            weighted_sum = sum(features[t - i - 1] * weights[i] for i in range(temporal_lags))
            new_feature = features[t] + weighted_sum
            new_features.append(new_feature)

        return new_features

    def GPU_set_dlabel_temporal_relation(self, ST_torch_loader, S_torch_loader, T_torch_loader,
                                         temporal_relation_weight, temporal_lags, device):
        self.featurizer.eval()
        self.tCVAE_encoder.eval()
        self.tCVAE_reparameterize.eval()
        self.ttemporal_states.eval()

        start_test = True
        with torch.no_grad():
            iter_test = iter(ST_torch_loader)
            for _ in range(len(ST_torch_loader)):
                data = next(iter_test)
                inputs = data[0]
                inputs = inputs.float()
                index = data[-1]
                index = index.cpu()
                all_c = data[1].long()
                all_c = all_c.cpu()
                all_ts = data[2].long()

                features = self.featurizer(inputs)
                features = features.cpu()
                labels = all_c

                # Create a dictionary to store features for each unique label
                features_dict = {}
                unique_labels = torch.unique(labels)
                for label in unique_labels:
                    mask = labels == label
                    features_dict[label.item()] = features[mask]
                # Iterate over the features_dict and apply the function
                new_features_dict = {}
                for label, time_series in features_dict.items():
                    new_features_dict[label] = self.compute_weighted_features(time_series, temporal_relation_weight,
                                                                              temporal_lags)

                # Extract tensors from dictionary and put them in a list
                lists_of_tensors = [v for k, v in new_features_dict.items()]
                # Concatenate tensors inside each list
                concatenated_tensors = [torch.stack(tensor_list, dim=0) for tensor_list in lists_of_tensors]
                # Concatenate the resulting tensors together
                combined_tensor = torch.cat(concatenated_tensors, dim=0)

                combined_tensor = combined_tensor.to(device)

                all_mu, all_logvar = self.tCVAE_encoder(combined_tensor)
                all_z = self.tCVAE_reparameterize(all_mu, all_logvar)
                outputs = self.ttemporal_states(all_z)
                feas = all_z

                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_index = index
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat(
                        (all_output, outputs.float().cpu()), 0)
                    all_index = np.hstack((all_index, index))
        all_output = nn.Softmax(dim=1)(all_output)

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)

        for _ in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, initc, 'cosine')
            pred_label = dd.argmin(axis=1)

        ST_torch_loader.dataset.tensors = (
            ST_torch_loader.dataset.tensors[0].to(device), ST_torch_loader.dataset.tensors[1].to(device), torch.tensor(pred_label).to(device),
            ST_torch_loader.dataset.tensors[3].to(device), ST_torch_loader.dataset.tensors[4].to(device))

        S_torch_loader.dataset.tensors = (
            S_torch_loader.dataset.tensors[0].to(device), S_torch_loader.dataset.tensors[1].to(device),
            torch.tensor(pred_label[0:len(S_torch_loader.dataset.tensors[0])]).to(device),
            S_torch_loader.dataset.tensors[3].to(device), S_torch_loader.dataset.tensors[4].to(device))

        T_torch_loader.dataset.tensors = (
            T_torch_loader.dataset.tensors[0].to(device), T_torch_loader.dataset.tensors[1].to(device),
            torch.tensor(pred_label[
                         len(S_torch_loader.dataset.tensors[0]): len(S_torch_loader.dataset.tensors[0]) + len(
                             T_torch_loader.dataset.tensors[0])]).to(device),
            T_torch_loader.dataset.tensors[3].to(device), T_torch_loader.dataset.tensors[4].to(device))

        self.featurizer.train()
        self.tCVAE_encoder.train()
        self.tCVAE_reparameterize.train()
        self.ttemporal_states.train()

    def set_tlabel(self, ST_torch_loader, S_torch_loader, T_torch_loader):
        self.tCVAE_encoder.eval()
        self.tCVAE_reparameterize.eval()
        self.ttemporal_states.eval()
        self.featurizer.eval()

        start_test = True
        with torch.no_grad():
            iter_test = iter(ST_torch_loader)
            for _ in range(len(ST_torch_loader)):
                data = next(iter_test)
                inputs = data[0]
                inputs = inputs.float()
                index = data[-1]
                all_mu, all_logvar = self.tCVAE_encoder(self.featurizer(inputs))
                all_z = self.tCVAE_reparameterize(all_mu, all_logvar)
                outputs = self.ttemporal_states(all_z)
                if start_test:
                    all_fea = all_z.float().cpu()
                    all_output = outputs.float().cpu()
                    all_index = index
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, all_z.float().cpu()), 0)
                    all_output = torch.cat(
                        (all_output, outputs.float().cpu()), 0)
                    all_index = np.hstack((all_index, index))
        all_output = nn.Softmax(dim=1)(all_output)

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)

        for _ in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, initc, 'cosine')
            pred_label = dd.argmin(axis=1)

        ST_torch_loader.dataset.tensors = (
            ST_torch_loader.dataset.tensors[0], ST_torch_loader.dataset.tensors[1], torch.tensor(pred_label),
            ST_torch_loader.dataset.tensors[3], ST_torch_loader.dataset.tensors[4])

        S_torch_loader.dataset.tensors = (
            S_torch_loader.dataset.tensors[0], S_torch_loader.dataset.tensors[1],
            torch.tensor(pred_label[0:len(S_torch_loader.dataset.tensors[0])]),
            S_torch_loader.dataset.tensors[3], S_torch_loader.dataset.tensors[4])

        T_torch_loader.dataset.tensors = (
            T_torch_loader.dataset.tensors[0], T_torch_loader.dataset.tensors[1],
            torch.tensor(pred_label[
                         len(S_torch_loader.dataset.tensors[0]): len(S_torch_loader.dataset.tensors[0]) + len(
                             T_torch_loader.dataset.tensors[0])]),
            T_torch_loader.dataset.tensors[3], T_torch_loader.dataset.tensors[4])

        self.tCVAE_encoder.train()
        self.tCVAE_reparameterize.train()
        self.ttemporal_states.train()
        self.featurizer.train()

    def predict(self, x):
        mu, _ = self.CVAE_encoder(self.featurizer(x))
        return self.classify_source(mu)