import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm
from torch import sigmoid, exp, randn_like


class linear_classifier(nn.Module):
    def __init__(self, input_dim, class_num):
        super(linear_classifier, self).__init__()
        self.fc = nn.Linear(input_dim, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x


class cvae_encoder(nn.Module):
    def __init__(self, in_features, hidden_size):
        super(cvae_encoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn_fc1 = nn.BatchNorm1d(int(in_features / 2), affine=True)
        self.fc1 = nn.Linear(in_features, int(in_features / 2))
        self.fc21 = nn.Linear(int(in_features / 2), hidden_size)
        self.bn_fc21 = nn.BatchNorm1d(hidden_size, affine=True)
        self.fc22 = nn.Linear(int(in_features / 2), hidden_size)
        self.bn_fc22 = nn.BatchNorm1d(hidden_size, affine=True)

    def forward(self, x):
        x = self.fc1(x) # self.relu(self.bn_fc1(self.fc1(x)))
        mu = self.fc21(x) # self.bn_fc21(self.fc21(x))
        logvar = self.fc22(x)
        return mu, logvar


class cvae_decoder(nn.Module):
    def __init__(self, in_features, hidden_size):
        super(cvae_decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(hidden_size, int(in_features / 2))
        self.bn_fc3 = nn.BatchNorm1d(int(in_features / 2), affine=True)
        self.fc4 = nn.Linear(int(in_features / 2), in_features)

    def forward(self, z):
        z = self.fc3(z) # self.relu(self.bn_fc3(self.fc3(z)))
        return sigmoid(self.fc4(z))


class cvae_reparameterize(nn.Module):
    def __init__(self):
        super(cvae_reparameterize, self).__init__()

    def forward(self, mu, logvar):
        std = exp(0.5 * logvar)
        eps = randn_like(std)
        return eps.mul(std).add_(mu)
