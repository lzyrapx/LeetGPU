import torch

# logits, true_labels, loss are tensors on the GPU
def solve(logits: torch.Tensor, true_labels: torch.Tensor, loss: torch.Tensor, N: int, C: int):
    loss[:] = torch.nn.functional.cross_entropy(input=logits, target=true_labels.long())