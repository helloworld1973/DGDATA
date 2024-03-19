import torch.nn as nn


class CNN_Feature_Extraction_Network(nn.Module):
    def __init__(self, conv1_in_channels, conv1_out_channels, conv2_out_channels, kernel_size_num, in_features_size):
        super(CNN_Feature_Extraction_Network, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=conv1_in_channels, out_channels=conv1_out_channels, kernel_size=(1, kernel_size_num)),
            nn.BatchNorm2d(conv1_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=conv1_out_channels, out_channels=conv2_out_channels,
                      kernel_size=(1, kernel_size_num)),
            nn.BatchNorm2d(conv2_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.in_features = in_features_size

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = x.view(-1, self.in_features)
        return x


class Transformer_Feature_Extraction_Network(nn.Module):
    def __init__(self, embed_size, heads, num_layers, in_features_size):
        super(Transformer_Feature_Extraction_Network, self).__init__()

        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,  # No decoding required in this use case
        )

        self.in_features = in_features_size

    def forward(self, x):
        x = x.squeeze(2)  # Remove the dimension of size 1, resulting shape: [7638, 6, 75]
        x = x.permute(2, 0, 1)  # Permute to desired shape, resulting shape: [75, 7638, 6]

        # Transformer expects input as (seq_len, batch_size, embed_size)
        # Pass through the Transformer encoder
        x = self.transformer.encoder(x)
        print(x.shape)

        # Flatten the output
        x = x.permute(1, 0, 2).contiguous().view(-1, self.in_features)
        print(x.shape)
        return x