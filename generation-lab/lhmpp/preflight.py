"""Exercise real CUDA extensions without constructing the reconstruction model."""
import importlib
import json
from pathlib import Path
import sys

import torch

result = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
          "gpu": torch.cuda.get_device_name(), "abi": torch._C._GLIBCXX_USE_CXX11_ABI, "modules": {}}
for name in ("pointops_cuda", "spconv.pytorch", "torch_scatter", "pytorch3d._C", "diff_gaussian_rasterization",
             "simple_knn._C", "gsplat.cuda._backend", "flash_attn"):
    module = importlib.import_module(name)
    result["modules"][name] = getattr(module, "__file__", "loaded")
from flash_attn import flash_attn_func
query = torch.randn(1, 32, 4, 32, device="cuda", dtype=torch.float16)
result["flashFinite"] = bool(torch.isfinite(flash_attn_func(query, query, query)).all())
from torch_scatter import scatter_sum
result["scatterCorrect"] = scatter_sum(torch.ones(4, device="cuda"), torch.tensor([0, 0, 1, 1], device="cuda")).tolist() == [2., 2.]
torch.cuda.synchronize()
result["pass"] = result["flashFinite"] and result["scatterCorrect"]
Path(sys.argv[1]).write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
if not result["pass"]:
    raise SystemExit(1)
