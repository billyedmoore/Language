import torch


def get_torch_device(silent=False):
    """
    Get the torch device with preference for cuda if available.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if not silent:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if not silent:
            print("CUDA not available, using CPU.")
    return device
