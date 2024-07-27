import torch
if torch.cuda.is_available():
    device = torch.device("gpu")
    print("Using gpu")
else:
    device = torch.device("cpu")
    print("Using cpu")