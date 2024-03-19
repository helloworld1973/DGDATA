import numpy as np
import torch
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches


def get_temporal_diff_train_data(S_x, S_y, T_x, T_y, batch_size, num_D, width, num_class):
    S_x_func = torch.tensor(S_x).to(torch.float32)
    S_x_func = S_x_func.view(-1, num_D, 1, width)
    S_y_func = torch.tensor(S_y)
    S_predict_ts = torch.zeros(len(S_y_func))
    S_d = [0 for i in range(len(S_y_func))]
    S_d_func = torch.tensor(S_d)
    S_idx = torch.arange(len(S_y_func))
    S_torch_dataset = Data.TensorDataset(S_x_func, S_y_func, S_predict_ts, S_d_func, S_idx)
    S_torch_loader = Data.DataLoader(dataset=S_torch_dataset, batch_size=batch_size, shuffle=False)

    T_x_func = torch.tensor(T_x).to(torch.float32)
    T_x_func = T_x_func.view(-1, num_D, 1, width)
    T_y_func = torch.tensor(T_y)
    T_predict_ts = torch.zeros(len(T_y_func))
    T_d = [1 for i in range(len(T_y_func))]
    T_d_func = torch.tensor(T_d)
    T_idx = torch.arange(len(S_y_func), len(S_y_func) + len(T_y_func))
    T_torch_dataset = Data.TensorDataset(T_x_func, T_y_func, T_predict_ts, T_d_func, T_idx)
    T_torch_loader = Data.DataLoader(dataset=T_torch_dataset, batch_size=batch_size, shuffle=False)

    ST_x = np.concatenate((S_x, T_x))
    ST_x_func = torch.tensor(ST_x).to(torch.float32)
    ST_x_func = ST_x_func.view(-1, num_D, 1, width)
    T_y_new = [i + num_class for i in T_y]
    ST_y = S_y + T_y_new
    ST_y_func = torch.tensor(ST_y)
    ST_predict_ts = torch.zeros(len(ST_y_func))
    ST_d = S_d + T_d
    ST_d_func = torch.tensor(ST_d)
    ST_idx = torch.arange(len(ST_y_func))
    ST_torch_dataset = Data.TensorDataset(ST_x_func, ST_y_func, ST_predict_ts, ST_d_func, ST_idx)
    ST_torch_loader = Data.DataLoader(dataset=ST_torch_dataset, batch_size=batch_size, shuffle=False)

    return S_torch_loader, T_torch_loader, ST_torch_loader


def GPU_get_temporal_diff_train_data(S_x, S_y, T_x, T_y, batch_size, num_D, width, num_class, device):
    # Define the device

    # Process S data
    S_x_func = torch.tensor(S_x, dtype=torch.float32).view(-1, num_D, 1, width).to(device)
    S_y_func = torch.tensor(S_y).to(device)
    S_predict_ts = torch.zeros(len(S_y_func)).to(device)
    S_d_func = torch.zeros(len(S_y_func)).to(device)  # Simplified from a loop
    S_idx = torch.arange(len(S_y_func)).to(device)
    S_torch_dataset = Data.TensorDataset(S_x_func, S_y_func, S_predict_ts, S_d_func, S_idx)
    S_torch_loader = Data.DataLoader(dataset=S_torch_dataset, batch_size=batch_size, shuffle=False)

    # Process T data
    T_x_func = torch.tensor(T_x, dtype=torch.float32).view(-1, num_D, 1, width).to(device)
    T_y_func = torch.tensor(T_y).to(device)
    T_predict_ts = torch.zeros(len(T_y_func)).to(device)
    T_d_func = torch.ones(len(T_y_func)).to(device)  # Simplified from a loop
    T_idx = torch.arange(len(S_y_func), len(S_y_func) + len(T_y_func)).to(device)
    T_torch_dataset = Data.TensorDataset(T_x_func, T_y_func, T_predict_ts, T_d_func, T_idx)
    T_torch_loader = Data.DataLoader(dataset=T_torch_dataset, batch_size=batch_size, shuffle=False)

    # Process combined ST data
    ST_x = np.concatenate((S_x, T_x))
    ST_x_func = torch.tensor(ST_x, dtype=torch.float32).view(-1, num_D, 1, width).to(device)
    T_y_new = [i + num_class for i in T_y]
    ST_y = S_y + T_y_new
    ST_y_func = torch.tensor(ST_y).to(device)
    ST_predict_ts = torch.zeros(len(ST_y_func)).to(device)
    ST_d_func = torch.cat([S_d_func, T_d_func]).to(device)
    ST_idx = torch.arange(len(ST_y_func)).to(device)
    ST_torch_dataset = Data.TensorDataset(ST_x_func, ST_y_func, ST_predict_ts, ST_d_func, ST_idx)
    ST_torch_loader = Data.DataLoader(dataset=ST_torch_dataset, batch_size=batch_size, shuffle=False)

    return S_torch_loader, T_torch_loader, ST_torch_loader


def draw_TSNE(S_mu, S_y, T_mu, T_y, epoch):
    # Sample data
    # T_mu = np.random.rand(100, 10)  # Replace with your actual data
    # T_y = np.random.randint(0, 5, 100)  # Replace with your actual labels
    # S_mu = np.random.rand(100, 10)  # Replace with your actual data
    # S_y = np.random.randint(0, 5, 100)  # Replace with your actual labels

    # Combine the features and labels from both sets
    combined_features = np.vstack((T_mu, S_mu))
    combined_labels = np.hstack((T_y, S_y))

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    transformed_features = tsne.fit_transform(combined_features)

    # Split the transformed features back into target and source sets
    T_mu_transformed = transformed_features[:len(T_mu)]
    S_mu_transformed = transformed_features[len(T_mu):]

    # Plotting
    plt.figure(figsize=(10, 8))

    # Unique labels
    unique_labels = np.unique(combined_labels)
    # Define colors for each label
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    # Plot each group with a different color and marker
    for label, color in zip(unique_labels, colors):
        # Filter points by label
        T_indices = (combined_labels[:len(T_mu)] == label)
        S_indices = (combined_labels[len(T_mu):] == label)

        # Plot target user points
        plt.scatter(T_mu_transformed[T_indices, 0], T_mu_transformed[T_indices, 1],
                    color=color, marker='x')

        # Plot source user points
        plt.scatter(S_mu_transformed[S_indices, 0], S_mu_transformed[S_indices, 1],
                    color=color, marker='*')

    # Create a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='none', label='Source User', markersize=10, markerfacecolor='black'),
        Line2D([0], [0], marker='x', color='none', label='Target User', markersize=10, markerfacecolor='black')
    ]
    for label, color in zip(unique_labels, colors):
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', label=f'Label {label}', markersize=10, markerfacecolor=color))

    plt.legend(handles=legend_elements, loc='upper right')

    plt.title('t-SNE visualization of Target and Source User Features')
    plt.grid(True)

    # Save the figure
    plt.savefig('tsne_visualization_' + str(epoch) + '.png')

    plt.show()
