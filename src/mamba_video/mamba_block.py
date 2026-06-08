"""Pure-PyTorch Mamba SSM block for CPU execution.

Implements the Selective State Space Model (Mamba) from Gu & Dao (2023)
without any CUDA kernels. Uses a sequential scan loop that runs on CPU.

This is intentionally not optimized for speed — correctness and portability
come first. Phase 5 (avx2-kernels) will provide the optimized version.

Reference: https://arxiv.org/abs/2312.00752
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MambaBlock(nn.Module):
    """Selective State Space Model block (Mamba).

    Drop-in replacement for a transformer attention block. Takes the same
    input/output shape: (batch, seq_len, d_model) → (batch, seq_len, d_model).

    Architecture:
        x → Linear(d_model → 2*d_inner) → split → [z, x']
        x' → Conv1D → SiLU → SSM scan → * SiLU(z) → Linear(d_inner → d_model) → y

    Args:
        d_model: Input/output dimension (matches attention hidden dim).
        d_state: SSM state dimension (default: 16).
        d_conv: Local convolution kernel width (default: 4).
        expand: Expansion factor for inner dimension (default: 2).
        dt_min: Minimum delta clamp value (default: 1e-4).
        dt_max: Maximum delta clamp value (default: 0.1).
        bias: Use bias in linear projections (default: False).
        conv_bias: Use bias in Conv1D (default: True).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 1e-4,
        dt_max: float = 0.1,
        bias: bool = False,
        conv_bias: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        self.dt_min = dt_min
        self.dt_max = dt_max

        # Input projection: project to 2 * d_inner (for x and gate z)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)

        # Local convolution on x branch
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias,
        )

        # SSM parameters
        # B and C are input-dependent (selectivity)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        # The +1 is for delta (Δ)

        # A is a learned diagonal matrix (log-space for stability)
        # Initialize as negative values so exp(A) < 1 (stable dynamics)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(self.d_inner, -1))

        # D is a skip connection parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Delta (Δ) projection for input-dependent discretization step
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        # Layer norm (optional, helps stability)
        self.norm = nn.LayerNorm(d_model)

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stability."""
        # Small initialization for projections
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.x_proj.weight, gain=0.1)

        # Delta projection bias: initialize to produce small positive values
        with torch.no_grad():
            self.dt_proj.bias.uniform_(0.001, 0.01)

    def _discretize(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        delta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Discretize continuous SSM parameters using Zero-Order Hold (ZOH).

        Converts continuous-time (A, B) to discrete-time (Ā, B̄) using:
            Ā = exp(Δ · A)
            B̄ = (exp(Δ · A) - I) · A⁻¹ · Δ · B ≈ Δ · B  (simplified)

        Uses float32 for numerical stability.

        Args:
            A: Continuous state matrix, shape (d_inner, d_state).
            B: Input-dependent B matrix, shape (batch, seq_len, d_state).
            delta: Discretization step, shape (batch, seq_len, d_inner).

        Returns:
            Tuple of (A_bar, B_bar) in discrete time.
        """
        # Compute in float32 for stability
        orig_dtype = delta.dtype
        A = A.float()
        B = B.float()
        delta = delta.float()

        # Ā = exp(Δ · A)
        # delta: (batch, seq, d_inner) → (batch, seq, d_inner, 1)
        # A: (d_inner, d_state) → (1, 1, d_inner, d_state)
        delta_A = delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        A_bar = torch.exp(delta_A)  # (batch, seq, d_inner, d_state)

        # B̄ ≈ Δ · B (first-order approximation)
        # delta: (batch, seq, d_inner) → (batch, seq, d_inner, 1)
        # B: (batch, seq, d_state) → (batch, seq, 1, d_state)
        B_bar = delta.unsqueeze(-1) * B.unsqueeze(-2)  # (batch, seq, d_inner, d_state)

        return A_bar.to(orig_dtype), B_bar.to(orig_dtype)

    def _selective_scan(
        self,
        x: torch.Tensor,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Run the selective scan (recurrent SSM) sequentially.

        This is the core SSM computation:
            h[t] = Ā[t] * h[t-1] + B̄[t] * x[t]
            y[t] = C[t] · h[t]

        Runs as a Python for-loop over timesteps. Not optimized —
        Phase 5 will replace this with AVX2 kernels.

        Args:
            x: Input sequence, shape (batch, seq_len, d_inner).
            A_bar: Discrete state matrix, shape (batch, seq_len, d_inner, d_state).
            B_bar: Discrete input matrix, shape (batch, seq_len, d_inner, d_state).
            C: Output matrix, shape (batch, seq_len, d_state).

        Returns:
            Output sequence, shape (batch, seq_len, d_inner).
        """
        batch, seq_len, d_inner = x.shape
        d_state = A_bar.shape[-1]

        # Initialize hidden state
        h = torch.zeros(batch, d_inner, d_state, dtype=x.dtype, device=x.device)

        outputs = []
        for t in range(seq_len):
            # h[t] = A_bar[t] * h[t-1] + B_bar[t] * x[t]
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)

            # y[t] = C[t] · h[t] — sum over state dimension
            # C: (batch, d_state) at time t
            # h: (batch, d_inner, d_state)
            y_t = torch.einsum("bn,bdn->bd", C[:, t], h)
            outputs.append(y_t)

        # Stack: list of (batch, d_inner) → (batch, seq_len, d_inner)
        y = torch.stack(outputs, dim=1)
        return y

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the Mamba block.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).
            residual: Optional residual connection input.

        Returns:
            Output tensor, shape (batch, seq_len, d_model).
        """
        # Normalize input
        x_norm = self.norm(x)

        # Input projection → split into x_branch and gate z
        xz = self.in_proj(x_norm)  # (batch, seq, 2 * d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # each (batch, seq, d_inner)

        # Conv1D on x branch
        # Conv1D expects (batch, channels, seq_len)
        x_conv = x_branch.transpose(1, 2)  # (batch, d_inner, seq)
        x_conv = self.conv1d(x_conv)[:, :, :x_branch.shape[1]]  # trim padding
        x_conv = x_conv.transpose(1, 2)  # (batch, seq, d_inner)

        # Activation
        x_conv = F.silu(x_conv)

        # SSM parameter projection
        x_proj_out = self.x_proj(x_conv)  # (batch, seq, d_state*2 + 1)

        # Split into B, C, and delta_raw
        B = x_proj_out[:, :, :self.d_state]           # (batch, seq, d_state)
        C = x_proj_out[:, :, self.d_state:2*self.d_state]  # (batch, seq, d_state)
        delta_raw = x_proj_out[:, :, -1:]             # (batch, seq, 1)

        # Delta projection and clamping
        delta = self.dt_proj(delta_raw)               # (batch, seq, d_inner)
        delta = F.softplus(delta)                     # Ensure positive
        delta = delta.clamp(min=self.dt_min, max=self.dt_max)

        # Get A from log-space
        A = -torch.exp(self.A_log)  # (d_inner, d_state), negative for stability

        # Discretize
        A_bar, B_bar = self._discretize(A, B, delta)

        # Selective scan
        y = self._selective_scan(x_conv, A_bar, B_bar, C)

        # Skip connection with D
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        # Gate with z
        y = y * F.silu(z)

        # Output projection
        y = self.out_proj(y)

        # Residual connection
        if residual is not None:
            y = y + residual
        else:
            y = y + x

        return y

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return (
            f"d_model={self.d_model}, d_state={self.d_state}, "
            f"d_conv={self.d_conv}, expand={self.expand}, "
            f"d_inner={self.d_inner}"
        )
