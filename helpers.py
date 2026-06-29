import torch


def get_device() -> torch.device:
    """
    Get the device to use for training and inference.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
