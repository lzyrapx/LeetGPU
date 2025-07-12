import torch

# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    img = image.to(torch.int16)
    img.resize_(height, width, 4)
    img[..., :-1] -= 255
    img = torch.abs(img)
    image.copy_(img)