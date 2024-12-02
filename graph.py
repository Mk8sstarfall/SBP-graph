import torch

def grad(phi, weight):
    """
    Compute the gradient.
    
    Parameters:
        phi (Tensor): Shape (B, T, N)
        weight (Tensor): Shape (B, T, N, N)
        
    Returns:
        Tensor: Shape (B, T, N, N)
    """
    phi_i = phi.unsqueeze(-1)
    phi_j = phi.unsqueeze(-2)
    return torch.sqrt(weight) * (phi_i - phi_j)

def div(m, weight):
    """
    Compute the divergence.
    
    Parameters:
        m (Tensor): Shape (B, T, N, N)
        weight (Tensor): Shape (B, T, N, N)
        
    Returns:
        Tensor: Shape (B, T, N)
    """
    return -torch.sum(m * torch.sqrt(weight), dim=-1)

def degree(weight):
    """
    Compute the node degree and return normalized degrees.
    
    Parameters:
        weight (Tensor): Shape (B, T, N, N)
        
    Returns:
        Tensor: Shape (B, T, N)
    """
    degree = torch.sum(weight, dim=-1)
    return degree / torch.sum(degree, dim=-1, keepdim=True)

def node_interpolation(rho, weight, eps=1e-8):
    """
    Compute the interpolation of rho.
    
    Parameters:
        rho (Tensor): Shape (B, T, N)
        weight (Tensor): Shape (B, T, N, N)
        eps (float): A small value to prevent division by zero
        
    Returns:
        Tensor: Shape (B, T, N, N)
    """
    d = degree(weight)
    rho = rho / (d + eps)  # Use eps to avoid division by zero
    rho_i = rho.unsqueeze(-1)
    rho_j = rho.unsqueeze(-2)
    theta = 0.5 * (rho_i + rho_j)
    return theta

def inner_product(v1, v2, rho, weight):
    """
    Compute the inner product of v1 and v2 given rho and weight.
    
    Parameters:
        v1 (Tensor): Shape (B, T, N, N)
        v2 (Tensor): Shape (B, T, N, N)
        rho (Tensor): Shape (B, T)
        weight (Tensor): Shape (B, T, N, N)
        
    Returns:
        Tensor: Shape (B, T)
    """
    theta = node_interpolation(rho, weight)
    return torch.sum(v1 * v2 * theta, dim=(-2, -1)) / 2
