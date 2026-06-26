import sys
import torch
import numpy as np

def main():
    print("Python:", sys.version)
    print("NumPy:", np.__version__)
    print("Torch:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        x = torch.randn(2048, 2048, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print("GPU matrix test:", y.shape, y.device)
    else:
        raise RuntimeError("PyTorch 没有检测到 CUDA GPU")

if __name__ == "__main__":
    main()
