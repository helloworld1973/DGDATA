import torch
import torch.nn as nn
import torch.optim as optim


# Linear regression model definition
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.scalar_weights = nn.Parameter(torch.randn(input_dim, requires_grad=True))
        self.input_dim = input_dim

    def forward(self, x):
        # Multiply each timestep by its scalar weight and sum all of them
        # x shape: (batch_size, 5, 2208)
        # output shape: (batch_size, 2208)
        output = (x * self.scalar_weights.view(self.input_dim, 1)).sum(dim=1)
        return output


def train_time_series_regression(device, features_dict, window_size=5, epochs=100, lr=0.01):
    """
    Train a linear regression model on multiple time series data.

    Parameters:
    - features_dict (dict): Dictionary of time series data.
    - window_size (int): Number of lags to be used for prediction.
    - epochs (int): Number of training epochs.
    - lr (float): Learning rate.

    Returns:
    - model (torch.nn.Module): Trained linear regression model.
    """

    # Training function
    def train_linear_regression_model(inputs_tensor, outputs_tensor, epochs, lr, device):

        model = LinearRegressionModel(window_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Transfer tensors to the chosen device
        inputs_tensor = inputs_tensor.to(device)
        outputs_tensor = outputs_tensor.to(device)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(inputs_tensor)
            loss = criterion(outputs, outputs_tensor)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.scalar_weights.data.clamp_(min=0)  # Ensure weights are non-negative

            # Removed the extra optimizer.step() as it was duplicated

        return model

    # Create a consolidated dataset from all time series
    consolidated_inputs = []
    consolidated_outputs = []

    for ts in features_dict.values():
        # Transform data
        inputs = [ts[i:i + window_size] for i in range(len(ts) - window_size)]
        outputs = [ts[i] for i in range(window_size, len(ts))]

        consolidated_inputs.extend(inputs)
        consolidated_outputs.extend(outputs)

    inputs_tensor = torch.stack(consolidated_inputs).float()
    outputs_tensor = torch.stack(consolidated_outputs, dim=0)

    # Train model on consolidated dataset
    model = train_linear_regression_model(inputs_tensor, outputs_tensor, epochs, lr, device)
    return model
