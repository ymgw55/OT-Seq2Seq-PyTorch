import torch
from torch import Tensor


def ipot_distance(C: Tensor,
                   device,
                   beta=1, t_steps=10, k_steps=1) -> Tensor:
    b, n, m = C.shape
    sigma = (torch.ones([b, m, 1]) / m).to(device)  # [b, m, 1]
    T = torch.ones([b, n, m]).to(device)
    A = torch.exp(-C / beta).to(device)  # [b, n, m]
    for _ in range(t_steps):
        Q = A * T  # [b, n, m]
        for _ in range(k_steps):
            delta = 1 / (n * torch.matmul(Q, sigma))  # [b, n, 1]
            sigma = 1 / (m * torch.matmul(torch.transpose(Q, 1, 2),
                         delta))  # [b, m, 1]
        T = delta * Q * torch.transpose(sigma, 1, 2)  # [b, n, m]

    distance = torch.diagonal(
        torch.matmul(torch.transpose(C, 1, 2), T), dim1=-2, dim2=-1).sum(-1)
    return distance
