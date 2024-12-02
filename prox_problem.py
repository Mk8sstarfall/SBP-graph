import torch
from graph import inner_product, grad, div, interpolation

def energy(rho, b, weight):
    # rho: (B, T, N)
    # b: (B, T, N, N)
    # weight: (B, T, N, N)
    # return: (B, )
    rho_mid = (rho[:, 1:, :] + rho[:, :-1, :]) / 2
    weight_mid = (weight[:, 1:, :, :] + weight[:, :-1, :, :]) / 2
    b_mid = (b[:, 1:, :, :] + b[:, :-1, :, :]) / 2
    return torch.mean(inner_product(b_mid, b_mid, rho_mid, weight_mid), dim=1)

def energy_t(rho, b, weight):
    # rho: (B, T, N)
    # b: (B, T, N, N)
    # weight: (B, T, N, N)
    # return: (B, T)
    rho_mid = (rho[:, 1:, :] + rho[:, :-1, :]) / 2
    weight_mid = (weight[:, 1:, :, :] + weight[:, :-1, :, :]) / 2
    b_mid = (b[:, 1:, :, :] + b[:, :-1, :, :]) / 2
    return inner_product(b_mid, b_mid, rho_mid, weight_mid)

def constraint(rho, b, weight, beta):
    # rho: (B, T, N)
    # b: (B, T-1, N, N)
    # weight: (B, T, N, N)
    # beta: constant
    # return: (B, T-1, N)
    dt = 1.0 / (rho.shape[1] - 1)
    rho_mid = (rho[:, 1:, :] + rho[:, :-1, :]) / 2
    weight_mid = (weight[:, 1:, :, :] + weight[:, :-1, :, :]) / 2
    b_mid = (b[:, 1:, :, :] + b[:, :-1, :, :]) / 2
    drho = rho[:, 1:, :] - rho[:, :-1, :]
    theta = interpolation(rho_mid, weight_mid)
    flux = theta * (b_mid - beta * grad(torch.log(rho_mid), weight_mid))
    constraint = drho / dt + div(flux, weight_mid)
    return constraint
