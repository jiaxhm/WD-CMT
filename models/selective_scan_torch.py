"""Real grouped selective recurrence in PyTorch, with chunk checkpointing.

Matches the real-valued grouped-B/C equations in state-spaces/mamba's
selective_scan_ref (mamba_ssm/ops/selective_scan_interface.py).
No state is detached at chunk boundaries. Slower than the CUDA extension.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk(state, u, delta, A, B, C):
    # Grouped layout avoids materializing B,C repeated across channels.
    outputs = []
    for t in range(u.shape[-1]):
        dt = delta[..., t, None]
        state = torch.exp(dt * A[None]) * state + dt * B[:, :, None, :, t] * u[..., t, None]
        outputs.append((state * C[:, :, None, :, t]).sum(-1))
    return state, torch.stack(outputs, dim=-1)


def selective_scan_torch(u, delta, A, B, C, D=None, delta_bias=None,
                         delta_softplus=True, chunk_size=64):
    if B.ndim != 4 or C.shape != B.shape or A.is_complex():
        raise ValueError('This backend requires real A and grouped B,C [batch,groups,state,length].')
    batch, channels, length = u.shape
    groups = B.shape[1]
    if channels % groups or chunk_size < 1:
        raise ValueError('Invalid scan groups or chunk size.')
    dtype = u.dtype
    # Keep doubles for numerical gradient tests; production inputs are float32.
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    u, delta, A, B, C = [x.to(work_dtype) for x in (u,delta,A,B,C)]
    if delta_bias is not None:
        delta = delta + delta_bias.to(work_dtype)[None,:,None]
    if delta_softplus:
        delta = F.softplus(delta)
    width, states = channels // groups, A.shape[-1]
    u = u.reshape(batch,groups,width,length)
    delta = delta.reshape_as(u)
    A = A.reshape(groups,width,states)
    state = u.new_zeros(batch,groups,width,states)
    output = []
    for start in range(0,length,chunk_size):
        end = min(start+chunk_size,length)
        args = (state,u[...,start:end],delta[...,start:end],A,B[...,start:end],C[...,start:end])
        if torch.is_grad_enabled() and any(x.requires_grad for x in args):
            state, y = checkpoint(_chunk,*args,use_reentrant=False)
        else:
            state, y = _chunk(*args)
        output.append(y)
    y = torch.cat(output,dim=-1).reshape(batch,channels,length)
    if D is not None:
        y = y + u.reshape(batch,channels,length)*D.to(work_dtype)[None,:,None]
    return y.to(dtype)
