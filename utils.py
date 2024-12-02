import torch
import matplotlib.pyplot as plt
import networkx as nx
import os


def interpolant(x: torch.Tensor, interp_way: str) -> torch.Tensor:
    """
    Perform interpolation on tensor `x` based on the given interpolation method.
    
    Parameters:
        x (Tensor): Input tensor of shape (B, T, N, N).
        interp_way (str): Interpolation method, can be 'left', 'right', 'mid'.
        
    Returns:
        Tensor: Interpolated tensor with the same shape as `x`.
    """
    if interp_way == 'left':
        return x[:, :-1]
    elif interp_way == 'right':
        return x[:, 1:]
    elif interp_way == 'mid':
        return (x[:, :-1] + x[:, 1:]) / 2
    else:
        return x


def log_training(log_file: str, iteration: int, max_iter: int, losses: list, 
                 total_loss: torch.Tensor, current_lr: float):
    """Log the training progress to console and a log file, including learning rate."""
    log_message = (
        f"Iteration {iteration}/{max_iter}, "
        + ", ".join([f"Loss {i}: {loss.item():.6f}" for i, loss in enumerate(losses)])
        + f", Total Loss: {total_loss.item():.6f}, Learning Rate: {current_lr:.8f}"
    )
    print(log_message)
    with open(log_file, 'a') as log_f:
        log_f.write(log_message + '\n')


def save_graph_visualization(save_dir: str, rho: torch.Tensor, weights: torch.Tensor, 
                             iteration: int, eps: float = 1e-5, is_directed: bool = False):
    """
    Save the visualization of graphs with node features and edge weights.
    
    Parameters:
        save_dir (str): Directory where the graph image will be saved.
        rho (Tensor): Node features of shape (B, T, N).
        weights (Tensor): Edge weights of shape (B, T, N, N).
        iteration (int): Current iteration, used in the filename.
        eps (float): Threshold for edge weight to be considered (default 1e-5).
        is_directed (bool): Whether the graph is directed (default False).
    """
    B, T, N = rho.shape
    
    half = (T + 1) // 2
    fig, axes = plt.subplots(2, half, figsize=(half * 6, 12))
    axes = axes.flatten()
    
    pos = None  # Layout for graph positions, reused across time steps
    
    for t in range(T):
        ax = axes[t]
        
        node_features = rho[:, t, :]  # (B, N)
        edge_weights = weights[:, t, :, :]  # (B, N, N)
        
        G = nx.DiGraph() if is_directed else nx.Graph()
        for i in range(N):
            G.add_node(i, feature=node_features[0, i].item())

        for i in range(N):
            for j in range(N):
                if edge_weights[0, i, j].item() > eps and not G.has_edge(i, j):
                    G.add_edge(i, j, weight=edge_weights[0, i, j].item())
        
        if pos is None:
            pos = nx.spring_layout(G, seed=42)  # Fix the layout with a fixed seed

        # Get node features for coloring
        node_color = [G.nodes[i]['feature'] for i in G.nodes]
        edge_width = [G[u][v]['weight'] * 5 for u, v in G.edges]
        edge_color = [G[u][v]['weight'] for u, v in G.edges]
        
        nx.draw_networkx_nodes(G, pos, node_size=500, cmap=plt.cm.Blues, node_color=node_color, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=edge_width, edge_color=edge_color, edge_cmap=plt.cm.Reds, alpha=0.7, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)

        edge_labels = {(i, j): f"{G[i][j]['weight']:.2f}" for i, j in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax)

        ax.set_title(f"Time step {t}")
        ax.axis('off')

    if T % 2 != 0:
        axes[-1].axis('off')  # Hide the last subplot
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"graph_visualization_{iteration}.png"))
    plt.close(fig)


def visualize_loss(save_dir: str, loss_history: list):
    """Visualize the energy loss over training iterations (log scale)."""
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Energy Loss")
    plt.title("Energy Loss during Training (Log Scale)")
    plt.xlabel("Iterations")
    plt.ylabel("Energy Loss")
    plt.yscale('log')  # Use log scale for the y-axis
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'energy_loss_log.png'))
    plt.close()


def visualize_lr(save_dir: str, lr_history: list):
    """Visualize the learning rate schedule over training iterations (log scale)."""
    plt.figure(figsize=(8, 5))
    plt.plot(lr_history, label="Learning Rate")
    plt.title("Learning Rate Schedule (Log Scale)")
    plt.xlabel("Iterations")
    plt.ylabel("Learning Rate")
    plt.yscale('log')  # Use log scale for the y-axis
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'learning_rate_log.png'))
    plt.close()


def visualize_energy(save_dir: str, E_t: torch.Tensor):
    """Visualize the energy across time steps."""
    plt.figure(figsize=(8, 5))
    plt.plot(E_t.detach().cpu().numpy(), marker='o', label="Energy")
    plt.title("Energy-t")
    plt.xlabel("Time step")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'Energy-t.png'))
    plt.show()


def visualize_energy_training(save_dir: str, E_t: torch.Tensor, iteration: int):
    """Visualize the energy at each time step during training."""
    plt.figure(figsize=(8, 5))
    plt.plot(E_t.detach().cpu().numpy(), marker='o', label="Energy")
    plt.title(f"Energy-t-iteration-{iteration}")
    plt.xlabel("Time step")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, f'Energy-t-iteration-{iteration}.png'))
    plt.show()


def save_hyperparameters(log_file: str, params: dict):
    """Save hyperparameters to the beginning of the log file."""
    with open(log_file, 'w') as log_f:
        log_f.write("Hyperparameters:\n")
        for key, value in params.items():
            log_f.write(f"{key}: {value}\n")
        log_f.write("\n")


def save_model(save_dir: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int):
    """
    Save the model and optimizer state, using the iteration number in the filename.
    
    Args:
        save_dir (str): Directory to save the model checkpoint.
        model (torch.nn.Module): The PyTorch model to save.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        iteration (int): The current training iteration, used to name the file.
    """
    os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists

    save_path = os.path.join(save_dir, f"model_checkpoint_iter_{iteration}.pth")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")
