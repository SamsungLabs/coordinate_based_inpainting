import torch
import torch.utils.data


def itt(img):
    tensor = torch.FloatTensor(img)  #
    if len(tensor.shape) == 3:
        tensor = tensor.permute(2, 0, 1)
    else:
        tensor = tensor.unsqueeze(0)
    return tensor


def tti(tensor):
    tensor = tensor.detach().cpu()
    tensor = tensor[0].permute(1, 2, 0)
    image = tensor.numpy()
    if image.shape[-1] == 1:
        image = image[..., 0]
    return image

