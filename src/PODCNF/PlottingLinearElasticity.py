import math
import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.linalg import svd
from tqdm import tqdm

import dolfin as fe

from dlroms import *

from src.DataGeneration.LinearElasticityData import FOMsampler

def visualize_elasticity_variability(mass_val, delta_val, Vh):

    n_plots = 4
    option = 1

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes_flat = axes.flatten()

    for i in range(n_plots):
        ax = axes_flat[i]
        plt.sca(ax)

        seed_in = np.random.randint(0, 2**32 - 1)
        _, _, _, u_data, theta_rad = FOMsampler(seed_in, mass_val, delta_val, option=option)

        if isinstance(theta_rad, (np.ndarray, list)):
             theta_deg = math.degrees(theta_rad.item())
        else:
             theta_deg = math.degrees(theta_rad)

        u_func = fe.Function(Vh)
        if hasattr(u_data, 'detach'):
            u_vec = u_data.detach().cpu().numpy()
        else:
            u_vec = u_data
        u_func.vector()[:] = u_vec

        fe.plot(u_func, mode="displacement", cmap="jet")

        ax.set_aspect('equal')
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.05, 1.05)

        ax.set_title(f"$\\theta = {theta_deg:.1f}^\\circ$", fontsize=14)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    plt.show()

def plot_data(g_tensor, u_tensor, Vh, V_lam, mu=None, theta=None):
    # Importiamo dolfin come 'df' per evitare conflitti con 'dlroms' che sovrascrive 'fe'
    import dolfin as df
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    u_func = df.Function(Vh)
    if torch.is_tensor(u_tensor):
        u_data = u_tensor.detach().cpu().numpy().flatten()
    else:
        u_data = np.array(u_tensor).flatten()
    u_func.vector().set_local(u_data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- 1. Material (g) ---
    plt.sca(axes[0])

    title_mat = "Property of the material"
    theta_val = None

    if theta is not None:
        try:
            theta_val = theta.item() if torch.is_tensor(theta) else theta
            theta_deg = np.degrees(theta_val)
            title_mat += f"\n$\\theta={theta_deg:.1f}^\\circ$"
        except:
            pass

    axes[0].set_title(title_mat)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    if theta_val is not None:
        N_px = 400
        x_grid = np.linspace(0, 1, N_px)
        y_grid = np.linspace(0, 1, N_px)
        X, Y = np.meshgrid(x_grid, y_grid)
        x_c, y_c = 0.5, 0.5
        cos_m = np.cos(theta_val)
        sin_m = np.sin(theta_val)
        mask = (Y - y_c) * cos_m - (X - x_c) * sin_m > 0
        Z = np.where(mask, 7.1, 0.1)
        axes[0].imshow(Z, extent=[0, 1, 0, 1], origin='lower', cmap='YlOrBr', interpolation='nearest')
        axes[0].plot([0, 1], [0.5, 0.5], color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
        x_line = np.linspace(0, 1, 100)
        y_line = 0.5 + (x_line - 0.5) * np.tan(theta_val)
        mask_line = (y_line >= 0) & (y_line <= 1)
        if np.any(mask_line):
            axes[0].plot(x_line[mask_line], y_line[mask_line], 'k-', linewidth=2.5)

    else:
        g_func = df.Function(V_lam)
        if torch.is_tensor(g_tensor):
            g_d = g_tensor.detach().cpu().numpy().flatten()
        else:
            g_d = np.array(g_tensor).flatten()
        g_func.vector().set_local(g_d)
        df.plot(g_func, cmap='YlOrBr')

    # --- 2. Deformation (u) ---
    plt.sca(axes[1])
    title_def = "True solution (Deformation)"
    if mu is not None:
        try:
            if torch.is_tensor(mu):
                mu_vals = mu.detach().cpu().numpy().flatten()
            else:
                mu_vals = np.array(mu).flatten()
            title_def += f"\n$\\mu$ = [{mu_vals[0]:.2f}, {mu_vals[1]:.2f}]"
        except:
            pass

    axes[1].set_title(title_def)
    mesh = Vh.mesh()
    original_coords = mesh.coordinates().copy()

    try:
        # Usiamo df.ALE e df.plot per essere sicuri di usare FEniCS
        df.ALE.move(mesh, u_func)

        # Calcolo magnitudo per il colore
        u_magnitude = df.sqrt(df.dot(u_func, u_func))
        V_scalar = df.FunctionSpace(mesh, 'CG', 1)
        mag_plot = df.project(u_magnitude, V_scalar)

        c2 = df.plot(mag_plot, cmap='jet')
        plt.colorbar(c2, ax=axes[1], shrink=0.8, label='Displacement magnitude')

    except Exception as e:
        # Fallback in caso di errore su ALE.move
        print(f"Warning: ALE move failed, plotting displacement directly. Error: {e}")
        df.plot(u_func, mode='displacement')

    finally:
        # Ripristina sempre le coordinate originali della mesh
        mesh.coordinates()[:] = original_coords

    plt.tight_layout()
    plt.show()

def plot_conditional_same_mu(n_generations,
                             g_sele, mu_sele,
                             u_true, u_rec,
                             V_lam, Vh,
                             theta=None):

    if torch.is_tensor(g_sele): g_data = g_sele.detach().cpu().numpy().flatten()
    else: g_data = np.array(g_sele).flatten()

    if torch.is_tensor(mu_sele): mu_val = mu_sele.detach().cpu().numpy().flatten()
    else: mu_val = np.array(mu_sele).flatten()

    # Sostituito df con fe
    g_func = fe.Function(V_lam)
    g_func.vector().set_local(g_data)

    u_true_func = fe.Function(Vh)
    if torch.is_tensor(u_true): u_t_data = u_true.detach().cpu().numpy().flatten()
    else: u_t_data = np.array(u_true).flatten()
    u_true_func.vector().set_local(u_t_data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- A. MATERIALE
    plt.sca(axes[0])
    title_mat = "Property of the material"

    theta_val = None
    if theta is not None:
        try:
            theta_val = theta.item() if torch.is_tensor(theta) else theta
            theta_deg = np.degrees(theta_val)
            title_mat += f"\n$\\theta={theta_deg:.1f}^\\circ$"
        except:
            pass

    axes[0].set_title(title_mat)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    if theta_val is not None:
        N_px = 400
        x_grid = np.linspace(0, 1, N_px)
        y_grid = np.linspace(0, 1, N_px)
        X, Y = np.meshgrid(x_grid, y_grid)

        x_c, y_c = 0.5, 0.5
        cos_m = np.cos(theta_val)
        sin_m = np.sin(theta_val)

        mask = (Y - y_c) * cos_m - (X - x_c) * sin_m > 0
        Z = np.where(mask, 7.1, 0.1)

        axes[0].imshow(Z, extent=[0, 1, 0, 1], origin='lower', cmap='YlOrBr', interpolation='nearest')
        axes[0].plot([0, 1], [0.5, 0.5], color='gray', linestyle='--', linewidth=1.5, alpha=0.8)

        x_line = np.linspace(0, 1, 100)
        y_line = 0.5 + (x_line - 0.5) * np.tan(theta_val)
        mask_line = (y_line >= 0) & (y_line <= 1)
        if np.any(mask_line):
            axes[0].plot(x_line[mask_line], y_line[mask_line], 'k-', linewidth=2.5)
    else:
        fe.plot(g_func, cmap='YlOrBr')

    # --- B. TRUE SOLUTION
    plt.sca(axes[1])
    axes[1].set_title(f"True solution (Deformation)\n($m={mu_val[0]:.2f}, \\delta={mu_val[1]:.2f}$)")

    mesh = Vh.mesh()
    original_coords = mesh.coordinates().copy()
    try:
        # Sostituito df con fe
        fe.ALE.move(mesh, u_true_func)
        u_mag = fe.sqrt(fe.dot(u_true_func, u_true_func))
        V_scal = fe.FunctionSpace(mesh, 'CG', 1)
        mag_plot = fe.project(u_mag, V_scal)

        c = fe.plot(mag_plot, cmap='jet')
        plt.colorbar(c, ax=axes[1], shrink=0.8, label='Displacement magnitude')
    finally:
        mesh.coordinates()[:] = original_coords

    plt.tight_layout()
    plt.show()

    # 2. GENERATED SAMPLES
    print(f"{n_generations} Generated samples:")

    cols = int(math.ceil(math.sqrt(n_generations)))
    rows = int(math.ceil(n_generations / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharex=True, sharey=True)
    axes_flat = axes.flatten() if n_generations > 1 else [axes]

    for i in range(n_generations):
        ax = axes_flat[i]
        plt.sca(ax)
        ax.set_title(f"Sample {i+1}")
        ax.set_aspect('equal', 'box')

        if isinstance(u_rec, list): s_tens = u_rec[i]
        else: s_tens = u_rec[i]

        u_gen_func = fe.Function(Vh)
        if torch.is_tensor(s_tens): s_data = s_tens.detach().cpu().numpy().flatten()
        else: s_data = np.array(s_tens).flatten()
        u_gen_func.vector().set_local(s_data)

        mesh = Vh.mesh()
        orig_coords = mesh.coordinates().copy()
        try:
            # Sostituito df con fe
            fe.ALE.move(mesh, u_gen_func)
            u_mag = fe.sqrt(fe.dot(u_gen_func, u_gen_func))
            V_scal = fe.FunctionSpace(mesh, 'CG', 1)
            mag_plot = fe.project(u_mag, V_scal)

            c = fe.plot(mag_plot, cmap='jet')

            cbar = plt.colorbar(c, ax=ax, shrink=0.6, label='Displacement magnitude')
            cbar.set_label('Displacement magnitude', size=12)

        except Exception:
            fe.plot(u_gen_func, mode='displacement')
        finally:
            mesh.coordinates()[:] = orig_coords

        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(n_generations, len(axes_flat)):
        axes_flat[i].axis('off')

    fig.suptitle(f"Possible configurations (Generated)\n($m$ = {mu_val[0]:.2f}, $\\delta$ = {mu_val[1]:.2f})", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_marginal_conditional_density(u_replica_true, u_rec, WD, i, label, mu):

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    fig.suptitle(f"{label}: $\\mu = [{mu[0]:.4f}, {mu[1]:.4f}]$")

    s = ['x', 'y']

    for k in range(2):
        sns.histplot(u_replica_true[:, i + k], label='True Data', ax=axes[k],
                     color='blue', stat='density', kde=True)
        sns.histplot(u_rec[:, i + k], label='Generated Data', ax=axes[k],
                     color='red', stat='density', kde=True)
        axes[k].set_title(f'Marginal DIM {label} ({s[k]}) - WD: {WD[i+k]:.4f}')
        axes[k].legend()

    plt.tight_layout()
    plt.show()

def plot_2D_conditional_density(u_replica_true, u_rec, i, WD_2, label, mu_sele):

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    col_map = 'twilight'
    sns.kdeplot(x=u_replica_true[:, i], y=u_replica_true[:, i+1],
                cmap = col_map, fill=True, thresh=0.05, ax = axes[0])
    # axes[0].set_title(f'True Data: $\\mu = [{mu_sele[0]:.4f}, {mu_sele[1]:.4f}]$')
    # axes[0].set_xlabel(f"x-{i}")
    # axes[0].set_ylabel(f"y-{i+1}")
    sns.kdeplot(x=u_rec[:, i], y=u_rec[:, i+1],
                cmap = col_map, fill=True, thresh=0.05, ax = axes[1])
    # axes[1].set_xlabel(f"x-{i}")
    # axes[1].set_ylabel(f"y-{i+1}")
    # axes[1].set_title('Generated Data')

    # plt.suptitle(f"{label} - Wasser_dist: {WD_2[i]:.4f}")
    plt.show()

def svd_residuals(u, V, mu, Vh):
    # from fenics import Function, FunctionSpace, project, sqrt, inner, plot
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import pearsonr

    # Gestione input Tensori/Numpy
    if isinstance(u, torch.Tensor): u = u.detach().cpu().numpy()
    if isinstance(V, torch.Tensor): V = V.detach().cpu().numpy()
    if isinstance(mu, torch.Tensor): mu = mu.detach().cpu().numpy()

    # 1. Ricostruzione (numpy)
    # u ~ c @ V.T
    c = np.dot(u, V)
    u_rec = np.dot(c, V.T)

    residuals = u - u_rec

    # Calcolo norme ed errori relativi per le statistiche
    u_norms = np.linalg.norm(u, axis=1)
    error_norms = np.linalg.norm(residuals, axis=1)

    # Gestione divisione per zero
    relative_errors = np.divide(error_norms, u_norms, out=np.zeros_like(error_norms), where=u_norms!=0)

    print(f"Mean Relative Error: {np.mean(relative_errors):.4%}")
    print(f"Max Relative Error:  {np.max(relative_errors):.4%}")

    # 2. Preparazione dati per i grafici
    mass = mu[:, 0]
    delta = mu[:, 1]

    # Calcolo Correlazioni di Pearson
    corr_mass = pearsonr(mass, relative_errors)[0]
    corr_delta = pearsonr(delta, relative_errors)[0]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # --- GRAFICO 1: Error vs Mass ---
    # Scatter
    sc1 = ax[0].scatter(mass, relative_errors, alpha=0.6, c=relative_errors, cmap='viridis', edgecolors='k', linewidth=0.3)

    # Calcolo Linea di Tendenza (Regressione Lineare)
    m_slope, q_interc = np.polyfit(mass, relative_errors, 1)
    x_vals = np.array([mass.min(), mass.max()])
    y_vals = m_slope * x_vals + q_interc

    # Plot Linea Tratteggiata
    ax[0].plot(x_vals, y_vals, 'r--', linewidth=2)

    ax[0].set_title(f'Error vs Mass (Corr: {corr_mass:.2f})')
    ax[0].set_xlabel('Mass')
    ax[0].set_ylabel('L2-relative Error')
    ax[0].legend()
    plt.colorbar(sc1, ax=ax[0])

    # --- GRAFICO 2: Error vs Delta ---
    # Scatter
    sc2 = ax[1].scatter(delta, relative_errors, alpha=0.6, c=relative_errors, cmap='viridis', edgecolors='k', linewidth=0.3)

    # Calcolo Linea di Tendenza
    m_slope, q_interc = np.polyfit(delta, relative_errors, 1)
    x_vals = np.array([delta.min(), delta.max()])
    y_vals = m_slope * x_vals + q_interc

    # Plot Linea Tratteggiata
    ax[1].plot(x_vals, y_vals, 'r--', linewidth=2)

    ax[1].set_title(f'Error vs Delta (Corr: {corr_delta:.2f})')
    ax[1].set_xlabel('Delta')
    ax[1].set_ylabel('L2-relative Error')
    ax[1].legend()
    plt.colorbar(sc2, ax=ax[1])

    plt.tight_layout()
    plt.show()

    # 3. Calcolo del campo medio dell'errore assoluto
    mean_abs_residual_field = np.mean(np.abs(residuals), axis=0)

    return mean_abs_residual_field

def analyze_bases_variation_main(u, mu, n_bases_list=range(1, 51, 2)):

    if isinstance(u, torch.Tensor): u = u.detach().cpu().numpy()
    if isinstance(mu, torch.Tensor): mu = mu.detach().cpu().numpy()

    n_samples = u.shape[0]
    ntrain = int(n_samples * 0.75)
    nval = int(ntrain + n_samples * 0.2)
    nval = min(nval, n_samples)

    u_train = u[:ntrain]
    u_val = u[ntrain:nval]
    mu_val = mu[ntrain:nval]

    # SVD
    spatial_modes, s, _ = svd(u_train.T, full_matrices=False)

    s_energy = (s**2) / np.sum(s**2)
    cum_energy = np.cumsum(s_energy)

    mean_errors = []
    max_errors = []
    corrs = {'Mass': [], 'Delta': []}

    # Loop calcolo errori
    for k in n_bases_list:
        V_k = spatial_modes[:, :k]
        coeffs = np.dot(u_val, V_k)
        u_rec = np.dot(coeffs, V_k.T)

        residuals = u_val - u_rec
        u_norms = np.linalg.norm(u_val, axis=1)
        err_norms = np.linalg.norm(residuals, axis=1)

        # Gestione divisione per zero
        rel_errors = np.divide(err_norms, u_norms, out=np.zeros_like(err_norms), where=u_norms!=0)

        mean_errors.append(np.mean(rel_errors))
        max_errors.append(np.max(rel_errors))

        # Correlazione con i parametri fisici
        if mu_val.shape[1] >= 2:
            corrs['Mass'].append(pearsonr(mu_val[:, 0], rel_errors)[0])
            corrs['Delta'].append(pearsonr(mu_val[:, 1], rel_errors)[0])


    fig = plt.figure(figsize=(18, 12))

    gs = gridspec.GridSpec(2, 4, figure=fig)

    # --- Plot 1: Mean Error
    ax1 = fig.add_subplot(gs[0, 0:2])

    l1, = ax1.plot(n_bases_list, np.array(mean_errors)*100, 'b-o', label='Mean Relative Error (%)')
    ax1.set_ylabel('Mean Error (%)', color='b', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xlabel('Number of Bases (N)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_title('Reconstruction Error vs Number of Bases', fontsize=16)

    # Twin axis per l'energia
    ax1_twin = ax1.twinx()
    energies_in_range = [cum_energy[k-1] for k in n_bases_list]
    l2, = ax1_twin.plot(n_bases_list, energies_in_range, 'g--', label='Cumulative Energy', linewidth=2)
    ax1_twin.set_ylabel('Cumulative Energy', color='g', fontsize=14)
    ax1_twin.tick_params(axis='y', labelcolor='g')

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    # --- Plot 2: Max Error
    ax2 = fig.add_subplot(gs[0, 2:4])

    ax2.plot(n_bases_list, np.array(max_errors)*100, 'r-s', label='Max Relative Error (%)')
    ax2.set_ylabel('Max Error (%)', fontsize=14)
    ax2.set_xlabel('Number of Bases (N)', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_title('Worst-Case Error vs Number of Bases', fontsize=16)
    ax2.legend(fontsize=12)

    # --- Plot 3: Correlations
    ax3 = fig.add_subplot(gs[1, 1:3])

    for param, values in corrs.items():
        if len(values) > 0:
            marker = 'o' if param == 'Mass' else 'x'
            # Usa colori distinti per Mass e Delta
            color = 'purple' if param == 'Mass' else 'orange'
            ax3.plot(n_bases_list, values, label=f'Corr(Err, {param})', marker=marker, color=color, linewidth=2)

    ax3.set_ylabel('Pearson Correlation', fontsize=14)
    ax3.set_xlabel('Number of Bases (N)', fontsize=14)
    ax3.axhline(0, color='black', linewidth=1.5, alpha=0.7) # Linea dello zero più visibile
    ax3.legend(fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.set_title('Error Correlation vs Control Parameters (Mass, Delta)', fontsize=16)

    plt.tight_layout()
    plt.show()

def Likelihood_Comparison_Main(flow, u, mu, V, mu_scaler, c_scaler, device,
                                    n_val, n_samples, ref_idx=None):

    if isinstance(mu, torch.Tensor):
        mu0_raw = mu[ref_idx].cpu().numpy().reshape(1, -1)
    else:
        mu0_raw = mu[ref_idx].reshape(1, -1)

    mu0_scaled = mu_scaler.transform(mu0_raw)
    mu0_tensor = torch.tensor(mu0_scaled, dtype=torch.float32).to(device)

    if isinstance(u, torch.Tensor):
        u0 = u[ref_idx].cpu().numpy()
    else:
        u0 = u[ref_idx]

    if isinstance(V, torch.Tensor):
        V_np = V.cpu().numpy()
    else:
        V_np = V

    c0 = u0 @ V_np
    c0_scaled = c_scaler.transform(c0.reshape(1, -1))
    c0_tensor = torch.tensor(c0_scaled, dtype=torch.float32).to(device)

    flow.eval()
    with torch.no_grad():
        target_like = flow.log_prob(mu0_tensor, c0_tensor).item()

    test_indices = list(range(n_val, n_samples))
    all_likelihoods = []
    all_distances = []

    print(f"Comparing Log-Likelihoods against ref_idx {ref_idx}...")

    with torch.no_grad():
        for idx in tqdm(test_indices, desc="Execution"):
            if isinstance(u, torch.Tensor):
                u_k = u[idx].cpu().numpy()
                mu_k = mu[idx].cpu().numpy()
            else:
                u_k = u[idx]
                mu_k = mu[idx]

            dist_mu = np.linalg.norm(mu0_raw - mu_k.reshape(1, -1))

            c_k = u_k @ V_np
            c_k = c_k.reshape(1, -1)
            c_k_scaled = c_scaler.transform(c_k)
            c_k_tensor = torch.tensor(c_k_scaled, dtype=torch.float32).to(device)

            log_prob = flow.log_prob(mu0_tensor, c_k_tensor)

            all_likelihoods.append(log_prob.item())
            all_distances.append(dist_mu)

    arr_dist = np.array(all_distances)
    arr_like = np.array(all_likelihoods)
    min_like = np.min(arr_like)

    plt.figure(figsize=(15, 7))

    sc = plt.scatter(arr_dist, arr_like, c=arr_like, cmap='viridis', alpha=0.6, s=30)

    plt.scatter([0], [target_like], c='red', s=200, label=r'Target $(\mu_0, u_0)$', edgecolors='black', zorder=10)
    plt.axhline(y=target_like, color='red', linestyle='--', alpha=0.4)
    plt.yscale('symlog', linthresh=1000)
    top_lim = max(200, target_like + 50)
    bottom_lim = min_like * 1.1 if min_like < 0 else min_like * 0.9
    plt.ylim(bottom=bottom_lim, top=top_lim)
    ticks_candidates = [100, 0, -1000, -1e4, -1e6, -1e8, -1e10, -1e12, -1e13]
    ticks_visible = [t for t in ticks_candidates if t >= bottom_lim]
    if min_like < -1e13:
        ticks_visible.append(min_like)

    plt.yticks(ticks_visible)
    plt.tick_params(axis='y', labelsize=13)
    plt.tick_params(axis='x', labelsize=13)

    plt.xlabel(r'$||\mu_k - \mu_0||_2$', fontsize=15)
    plt.ylabel(r'Log-Likelihood $\log p(u_k | \mu_0)$', fontsize=15)
    title_str = f'Global Robustness Analysis\nTarget Likelihood: {target_like:.2f}'
    if mu0_raw.shape[1] >= 2:
        title_str += f'\n $\mu_0 = [{mu0_raw[0, 0]:.4f}, {mu0_raw[0, 1]:.4f}]$'
    else:
        title_str += f'\n $\mu_0 = [{mu0_raw[0, 0]:.4f}]$'

    plt.title(title_str, fontsize=16)
    cbar = plt.colorbar(sc, label='Log-Likelihood', shrink=0.7)
    cbar.ax.tick_params(labelsize=12)

    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    plt.legend(fontsize=14)

    plt.tight_layout()
    plt.show()

    return arr_dist, arr_like