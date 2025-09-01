import torch

# (Optional) speed knobs on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

def solve(A, B, C, M, N, K,
          scale_A, scale_B, scale_C,
          zero_point_A, zero_point_B, zero_point_C):

    # Expect A, B, C already allocated on the desired device (e.g., CUDA)
    A = A.view(M, K)
    B = B.view(K, N)
    C = C.view(M, N)

    device = A.device
    assert B.device == device and C.device == device, "A, B, C must be on the same device"

    # Quant offsets (int32) on-device
    zpA = int(zero_point_A)
    zpB = int(zero_point_B)
    zpC = int(zero_point_C)

    # Shift to symmetric int domain (still tensor ops on-device)
    A_int = A.to(torch.int32) - zpA
    B_int = B.to(torch.int32) - zpB

    # Compute integer matmul in float on GPU (fast and exact for 32-bit integer sums)
    # NOTE: Direct int32/int8 matmul isn't implemented/perf on CUDA in PyTorch;
    # doing it in float32 on-GPU avoids CPU fallbacks.
    acc_float = torch.matmul(A_int.to(torch.float32), B_int.to(torch.float32))

    # Requantize
    scale_factor = float(scale_A) * float(scale_B) / float(scale_C)
    acc_float = acc_float * scale_factor

    # Round-to-nearest-even like torch.round, shift by output zero-point, clamp to int8
    rounded = torch.round(acc_float)
    shifted = rounded + float(zpC)
    clamped = torch.clamp(shifted, min=-128.0, max=127.0).to(torch.int8)

    # Write into provided C tensor
    C.copy_(clamped)