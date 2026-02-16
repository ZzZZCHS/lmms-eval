import torch
import time

x = torch.randn(1024, 1024, device="cuda")

while True:
    x = x @ x   # tiny matmul
    torch.cuda.synchronize()
    time.sleep(60)
