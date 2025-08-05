import torch

# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor,
          M: int, N: int, d: int):
      attn_scores = Q @ K.T
      attn_values = attn_scores / (d ** 0.5)
      attn_values = torch.softmax(attn_values, dim=1) 
      attn_output = attn_values @ V
      output[:] = attn_output.reshape(-1)