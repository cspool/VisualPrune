from __future__ import annotations

import torch


def _stats(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu()
    shape = tuple(data.shape)
    dtype = str(data.dtype).replace("torch.", "")
    numel = data.numel()
    if numel == 0:
        return f"shape={shape} dtype={dtype} empty"

    zero_fraction = (data == 0).to(torch.float32).mean().item()
    finite_fraction = torch.isfinite(data.to(torch.float32)).to(torch.float32).mean().item()

    if data.dtype == torch.bool:
        true_fraction = data.to(torch.float32).mean().item()
        return (
            f"shape={shape} dtype={dtype} true={true_fraction:.4f} "
            f"zero={zero_fraction:.4f}"
        )

    numeric = data.to(torch.float32)
    return (
        f"shape={shape} dtype={dtype} "
        f"mean={numeric.mean().item():+.6f} std={numeric.std(unbiased=False).item():.6f} "
        f"min={numeric.min().item():+.6f} max={numeric.max().item():+.6f} "
        f"zero={zero_fraction:.4f} finite={finite_fraction:.4f}"
    )


def log_section(title: str) -> None:
    print(f"\n{title}")


def log_tensor(name: str, tensor: torch.Tensor, note: str = "") -> None:
    suffix = f"  # {note}" if note else ""
    print(f"  {name:<28} {_stats(tensor)}{suffix}")


def log_tensors(title: str, tensors: dict[str, torch.Tensor]) -> None:
    log_section(title)
    for name, tensor in tensors.items():
        log_tensor(name, tensor)
