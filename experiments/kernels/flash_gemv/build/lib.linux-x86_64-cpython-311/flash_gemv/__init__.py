import importlib
import os.path as osp
import torch
from .flash_gemv import fuse_gemv_cmp, fuse_gemv_flag, fuse_gemv_flag_gemv
__version__ = '0.0.1'

library = '_C'
spec = importlib.machinery.PathFinder().find_spec(
    library, [osp.dirname(__file__)])
if spec is not None:
    torch.ops.load_library(spec.origin)
else:
    raise ImportError(f"Could not find module '{library}' in "
                      f'{osp.dirname(__file__)}')

__all__ = [fuse_gemv_cmp, fuse_gemv_flag, fuse_gemv_flag_gemv]
