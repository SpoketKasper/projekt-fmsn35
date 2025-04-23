import torch
import torch.nn as nn
import math

def dastft_window_transform(w_in, win_min, win_max, dtype, device):
    w_out = torch.minimum(
        torch.maximum(
            w_in,
            torch.full_like(w_in, win_min, dtype=dtype, device=device),
        ),
        torch.full_like(w_in, win_max, dtype=dtype, device=device),
    )
    return w_out

def dastft_stride_transform(s_in, stride_min, stride_max, dtype, device):
    s_out = torch.minimum(
        torch.maximum(
            s_in,
            torch.full_like(s_in, stride_min, dtype=dtype, device=device),
        ),
        torch.full_like(s_in, stride_max, dtype=dtype, device=device),
    )
    return s_out

def dastft_window_function(direction, idx_frac, N, F, T, actual_win_length, dtype, device):
    base = torch.arange(
        0, N, 1, dtype=dtype, device=device,
    )[:, None, None].expand([-1, F, T])
    base = base - idx_frac
    
    mask1 = base.ge(torch.ceil((N - 1 + actual_win_length) / 2))
    mask2 = base.le(torch.floor((N - 1 - actual_win_length) / 2))
    
    if direction == "forward":
        tap_win = 0.5 - 0.5 * torch.cos(
            2
            * math.pi
            * (base + (actual_win_length - N + 1) / 2)
            / actual_win_length
        )
        tap_win[mask1] = 0
        tap_win[mask2] = 0
        tap_win = tap_win / N * 2
        return tap_win
    
    elif direction == "backward":
        f = torch.sin(
            2
            * math.pi
            * (base + (actual_win_length - N + 1) / 2)
            / actual_win_length
        )
        d_tap_win = (
            -math.pi
            / actual_win_length.pow(2)
            * ((N - 1) / 2 - base)
            * f
        )
        d_tap_win[mask1] = 0
        d_tap_win[mask2] = 0
        d_tap_win = d_tap_win / N * 2
        return d_tap_win
    return None

def dastft_unfold(x, frames, N, L, device):
    idx_floor = frames.floor()
    idx_frac = frames - idx_floor
    idx_floor = idx_floor.long()[:, None].expand(
        (idx_floor.shape[0], N)
    ) + torch.arange(0, N, device=device)
    
    idx_floor[idx_floor >= L] = -1
    folded_x = x[:, idx_floor]
    folded_x[:, idx_floor < 0] = 0
    
    return folded_x, idx_frac

def dastft_compute(
    x, 
    win_length_param, 
    strides_param, 
    N, 
    F, 
    T, 
    pow=1.0, 
    tapering_function="hann",
    direction="forward"
):
    device = x.device
    dtype = x.dtype
    L = x.shape[-1]
    
    actual_win_length = dastft_window_transform(
        win_length_param, N/20, N, dtype, device
    )
    actual_strides = dastft_stride_transform(
        strides_param, 0, max(N, strides_param.abs().max()), dtype, device
    )
    
    expanded_stride = actual_strides.expand((T,))
    frames = torch.zeros_like(expanded_stride)
    frames[1:] = frames[0] + expanded_stride[1:].cumsum(dim=0)
    
    folded_x, idx_frac = dastft_unfold(x, frames, N, L, device)
    
    tap_win = dastft_window_function(
        direction, idx_frac, N, F, T, actual_win_length, dtype, device
    ).permute(2, 1, 0)  # T, N, N
    
    shift = torch.arange(
        end=F,
        device=device,
        dtype=dtype,
        requires_grad=False,
    )
    shift = idx_frac[:, None] * shift[None, :]  # T, N
    
    folded_x = folded_x[:, :, None, :]  # B, T, 1, N
    tap_win = tap_win[None, :, :, :]  # 1, T, F, 1
    shift = torch.exp(2j * math.pi * shift / N)[None, :, :, None]  # 1, T, N, 1
    tapered_x = folded_x * tap_win * shift  # B, T, F, N
    
    coeff = torch.arange(
        end=N,
        device=device,
        dtype=dtype,
        requires_grad=False,
    )
    coeff = coeff[:F, None] @ coeff[None, :]
    coeff = torch.exp(-2j * math.pi * coeff / N)  # N, N
    
    coeff = coeff[None, None, :, :]  # 1, 1, N, N
    stft = (tapered_x * coeff).sum(dim=-1)
    stft = stft.permute(0, 2, 1)
    
    if direction == "forward":
        spec = stft.abs().pow(pow)[:, :F] + torch.finfo(dtype).eps
        return spec, stft
    return stft