
from torch.cuda import is_available as check_cuda

def runtime():
    if check_cuda(): return "cuda"
    else: return "cpu"