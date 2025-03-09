import torch


def auto_device():
    """
    Automatically selects the device for PyTorch based on availability of CUDA or MPS.
    Returns:
        torch.device: The selected device (either "cuda", "mps", or "cpu").
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")
    return device
