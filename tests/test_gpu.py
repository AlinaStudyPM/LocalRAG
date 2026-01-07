# tests/test_gpu.py
import torch

if __name__ == "__main__":
    answer ="cuda" if torch.cuda.is_available() else "cpu"
    print(answer)