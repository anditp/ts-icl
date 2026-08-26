# test_torch_cuda.py
import os, sys, torch, traceback
local = os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "unset"))
print(f"ENV: LOCAL_RANK={os.environ.get('LOCAL_RANK')}, RANK={os.environ.get('RANK')}, "
      f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
try:
    lr = int(os.environ.get("LOCAL_RANK", 0))
    print("attempting torch.cuda.set_device(", lr, ")")
    torch.cuda.set_device(lr)
    print("current_device:", torch.cuda.current_device())
    print("device_name:", torch.cuda.get_device_name(torch.cuda.current_device()))
except Exception:
    print("ERROR setting device:")
    traceback.print_exc()