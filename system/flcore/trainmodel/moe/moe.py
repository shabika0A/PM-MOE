import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.trainmodel.moe.gate import Gating, CNNGating
from flcore.trainmodel.models import fastText


def _apply_expert_weights(final_results: torch.Tensor, weights_vec: torch.Tensor) -> torch.Tensor:
    """Combine expert outputs using per-sample weights.
    final_results: [B, k, C] or [B, C]
    weights_vec:   [B, k] or [k]
    Returns: [B, C]
    """
    if not hasattr(final_results, "dim"):
        return final_results
    if final_results.dim() == 2:  # [B, C]
        return final_results

    # [B, k, C] expected
    assert final_results.dim() == 3, f"expected [B,k,C], got {tuple(final_results.shape)}"

    if not hasattr(weights_vec, "dim"):
        raise TypeError("weights_vec must be a tensor")

    if weights_vec.dim() == 1:  # [k] → [1,k] → [B,k]
        weights_vec = weights_vec.unsqueeze(0).expand(final_results.shape[0], -1)
    elif weights_vec.dim() == 2:  # [B,k]
        if weights_vec.size(0) != final_results.size(0):
            raise ValueError("weights_vec batch size mismatch with final_results")
    else:
        raise ValueError("weights_vec must be [k] or [B,k]")

    weights_vec = weights_vec.to(final_results.dtype)
    return torch.sum(final_results * weights_vec.unsqueeze(-1), dim=1)


class MoE(nn.Module):
    """Dense MoE over trained_experts with a learned gating over the flattened input.
    """
    def __init__(self, trained_experts: list[nn.Module]):
        super().__init__()
        self.experts = nn.ModuleList(trained_experts)
        num_experts = len(trained_experts)
        # infer an input dimension for gating using a small conv rule-of-thumb; fallback to last expert attr
        in_feat = getattr(trained_experts[0], 'in_features', None)
        if in_feat is None:
            in_feat = getattr(trained_experts[0], 'in_channels', 1)
        self.gating = Gating(in_feat, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            B = x.size(0)
            weights = self.gating(x.flatten(1))  # [B, E]
            outs = []
            for e in self.experts:
                y = e(x)
                if y.dim() > 2:
                    y = y.view(B, -1)
                outs.append(y)
            stacked = torch.stack(outs, dim=1)  # [B, E, C]
            return torch.sum(stacked * weights.unsqueeze(-1), dim=1)  # [B, C]


class ToPMoE(nn.Module):
    """Top‑k MoE with optional args.topk and optional args.id (unused for syntax safety)."""
    def __init__(self, trained_experts: list[nn.Module], gate_input_dim: int, args):
        super().__init__()
        self.experts = nn.ModuleList(trained_experts)
        self.num_experts = len(trained_experts)
        self.gating = Gating(gate_input_dim, self.num_experts)
        self.k = int(getattr(args, 'topk', max(1, self.num_experts)))
        self.args = args
        self.energy_T = float(getattr(args, 'energy_T', 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        k = min(self.k, self.num_experts)
        weights = self.gating(x.flatten(1))               # [B,E]
        wvals, idxs = torch.topk(weights, k, dim=-1)      # [B,k]

        results = []
        for i in range(B):
            expert_outs = []
            for idx in idxs[i]:
                yi = self.experts[int(idx)](x[i].unsqueeze(0))  # [1,C] or [1,*]
                yi = yi.view(yi.size(0), -1).squeeze(0)         # [C]
                expert_outs.append(yi)
            results.append(torch.stack(expert_outs, dim=0))     # [k,C]

        final_results = torch.stack(results, dim=0)             # [B,k,C]
        return _apply_expert_weights(final_results, wvals)


class NormalToPMoE(nn.Module):
    def __init__(self, trained_experts: list[nn.Module], gate_input_dim: int, args):
        super().__init__()
        self.experts = nn.ModuleList(trained_experts)
        self.num_experts = len(trained_experts)
        self.gating = Gating(gate_input_dim, self.num_experts)
        self.k = int(getattr(args, 'topk', max(1, self.num_experts)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        k = min(self.k, self.num_experts)
        weights = self.gating(x.flatten(1))
        wvals, idxs = torch.topk(weights, k, dim=-1)

        results = []
        for i in range(B):
            expert_outs = []
            for idx in idxs[i]:
                yi = self.experts[int(idx)](x[i].unsqueeze(0))
                yi = yi.view(yi.size(0), -1).squeeze(0)
                expert_outs.append(yi)
            results.append(torch.stack(expert_outs, dim=0))

        final_results = torch.stack(results, dim=0)
        return _apply_expert_weights(final_results, wvals)


class ExtractorToPMoE(nn.Module):
    def __init__(self, trained_experts: list[nn.Module], gate_input_dim: int, args):
        super().__init__()
        self.experts = nn.ModuleList(trained_experts)
        self.num_experts = len(trained_experts)
        self.gating = Gating(gate_input_dim, self.num_experts)
        self.k = int(getattr(args, 'topk', max(1, self.num_experts)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        k = min(self.k, self.num_experts)
        weights = self.gating(x.flatten(1))
        wvals, idxs = torch.topk(weights, k, dim=-1)

        results = []
        if isinstance(self.experts[0], fastText):
            for i in range(B):
                outs = []
                for idx in idxs[i]:
                    ft = self.experts[int(idx)]
                    feat = ft.fc1(x[i].unsqueeze(0))
                    logits = ft.fc(feat)
                    out = F.log_softmax(logits, dim=1).flatten(0)
                    outs.append(out)
                results.append(torch.stack(outs, dim=0))
        else:
            for i in range(B):
                outs = []
                for idx in idxs[i]:
                    yi = self.experts[int(idx)](x[i].unsqueeze(0)).flatten(0)
                    outs.append(yi)
                results.append(torch.stack(outs, dim=0))

        final_results = torch.stack(results, dim=0)
        return _apply_expert_weights(final_results, wvals)


class ParamToPMoE(nn.Module):
    def __init__(self, trained_experts: list[torch.Tensor], args):
        super().__init__()
        self.experts = trained_experts  # list of parameter tensors [C]
        self.num_experts = len(trained_experts)
        exp_dim = trained_experts[0].shape[0]
        self.gating = Gating(exp_dim, self.num_experts)
        self.k = int(getattr(args, 'topk', max(1, self.num_experts)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        k = min(self.k, self.num_experts)
        weights = self.gating(x)  # [B,E]
        wvals, idxs = torch.topk(weights, k, dim=-1)
        results = []
        for i in range(B):
            outs = [self.experts[int(idx)] for idx in idxs[i]]  # list of [C]
            results.append(torch.stack(outs, dim=0))            # [k,C]
        final_results = torch.stack(results, dim=0)              # [B,k,C]
        return _apply_expert_weights(final_results, wvals)


class PatchMoE(nn.Module):
    def __init__(self, trained_experts: list[nn.Module], data_type: str = "cifar10"):
        super().__init__()
        self.data_type = data_type
        self.experts = nn.ModuleList(trained_experts)
        self.num_experts = len(trained_experts)
        in_feat = getattr(trained_experts[0], 'in_features', None)
        if in_feat is None:
            in_feat = getattr(trained_experts[0], 'in_channels', 1) * self.num_experts
        self.gating = Gating(in_feat, self.num_experts)
        self.trained_experts = trained_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x_flat = x.view(B, -1)
        weights = self.gating(x_flat)                 # [B,E]
        # split features per expert along last dim
        seg_size = x_flat.size(1) // self.num_experts
        segs = [x_flat[:, i*seg_size:(i+1)*seg_size] for i in range(self.num_experts)]
        outs = [e(seg) for e, seg in zip(self.experts, segs)]
        stacked = torch.stack(outs, dim=1)            # [B,E,C]
        mix = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        if self.data_type == 'cifar10':
            return mix.view(B, 3, 32, 32)
        return mix


class PatchCNNMoE(nn.Module):
    def __init__(self, trained_experts: list[nn.Module], data_type: str = "cifar10"):
        super().__init__()
        self.data_type = data_type
        self.experts = nn.ModuleList(trained_experts)
        self.num_experts = len(trained_experts)
        in_ch = getattr(trained_experts[0], 'in_channels', 1)
        self.gating = CNNGating(in_ch, self.num_experts)

    @staticmethod
    def split_tensor(x: torch.Tensor, n: int):
        assert x.dim() == 4, "Input must be [B,C,H,W]"
        B, C, H, W = x.shape
        if H % n == 0:
            return torch.chunk(x, chunks=n, dim=-2)
        if W % n == 0:
            return torch.chunk(x, chunks=n, dim=-1)
        raise ValueError(f"Neither H={H} nor W={W} divisible by n={n}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        splits = self.split_tensor(x, self.num_experts)
        weights = self.gating(x)              # [B,E]
        outs = [e(seg) for e, seg in zip(self.experts, splits)]
        stacked = torch.stack(outs, dim=1)    # [B,E,C]
        mix = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        if self.data_type == 'cifar10':
            return mix.view(B, -1)            # flat features
        return mix
