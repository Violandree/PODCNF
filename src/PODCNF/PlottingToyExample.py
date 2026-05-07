import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_samples_3D(test_loader, s=None, n_points_to_plot=1000):

    d_full = test_loader.dataset.data.cpu().numpy()
    sample_indices_d = np.random.choice(d_full.shape[0], n_points_to_plot, replace=False)
    d = d_full[sample_indices_d]

    fig = plt.figure(figsize=(15, 6))

    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    ax0.set_title('Data', fontsize=16)
    ax0.scatter(d[:, 0], d[:, 1], d[:, 2], c='red', alpha=0.3, s=15)

    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.set_title('Sample', fontsize=16)

    if s is not None:
        s_full = s.detach().cpu().numpy()
        sample_indices_s = np.random.choice(s_full.shape[0], n_points_to_plot, replace=False)
        s_plot = s_full[sample_indices_s]
        ax1.scatter(s_plot[:, 0], s_plot[:, 1], s_plot[:, 2], c='blue', alpha=0.3, s=15)

    plt.tight_layout()

    plt.show()

def plot_conditional(model, x_values, n_samples, device):

    model.eval()

    fig, axes = plt.subplots(1, len(x_values),
                            figsize=(20, 5),
                            sharey=True)

    for i, x_val in enumerate(x_values):
        ax = axes[i]
        with torch.no_grad():
            x_condition = torch.full((n_samples, 1), x_val).to(device) # Ensure x_condition is on the device
            y_samples = model.sample(x_condition).cpu().numpy()

        y_sin_samples = y_samples[:, 0]
        y_bim_samples = y_samples[:, 1]

        sns.kdeplot(x=y_sin_samples, y=y_bim_samples, ax=ax,
                    cmap="viridis", fill=True, thresh=0.05)

        ax.set_title(f'p(y | x = {x_val:.2f})')
        ax.set_xlabel('y_sin')
        if i == 0:
            ax.set_ylabel('y_bim')
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('p(y|x)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()