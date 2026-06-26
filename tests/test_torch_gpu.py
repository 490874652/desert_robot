import torch

def main():
    print("Torch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    assert torch.cuda.is_available(), "CUDA 不可用，PyTorch 没有识别到 GPU"

    device = torch.device("cuda")
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    c = a @ b

    print("GPU:", torch.cuda.get_device_name(0))
    print("Result:", c.mean().item())
    print("OK")

if __name__ == "__main__":
    main()
