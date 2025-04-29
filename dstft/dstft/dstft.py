#!/usr/bin/env python

"""Module docstring goes here."""

from __future__ import annotations

from math import pi
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn

import time
tic_time = None
def tic():
    global tic_time
    tic_time = time.time()
def toc(print_elapsed=True):
    global tic_time
    elapsed = time.time() - tic_time
    if print_elapsed:
        print(f"Elapsed time: {elapsed:.6f} seconds")
    return elapsed

import torch.utils.bottleneck as bottleneck
import numpy as np
from scipy.signal import stft
from scipy.stats import beta
from scipy.interpolate import BSpline
import torch.nn.functional as F
import cv2
class fastDSTFT(nn.Module):
    def __init__(self, x,
                    initial_win_length: float,   
                    support: int,                
                    stride: int, 
                    #warm_start: bool = False,
                    window_function: str = 'beta',
                    mel_filter: bool = False,
                    n_mels: int = None,
                    spline_density: int = 1,  # number of splines per stride
                    spline_degree: int = 3,  # degree of the spline
                    #win_p: str = None,
                    #stride_p: str = None,
                    #win_requires_grad: bool = True,
                    #stride_requires_grad: bool = True,
                    win_min = None,
                    win_max = None,
                    #stride_min = None,
                    #stride_max = None,    
                    #tapering_function: str = 'hann',
                    sr : int = 16_000,
                    ):
        super().__init__()     
        
        # Constants and hyperparameters
        self.x = x
        self.sr = sr
        self.eps = torch.finfo(x.dtype).eps
        self.device = x.device
        self.dtype = x.dtype

        self.stride = stride
        self.N = support                                                                    # support size
        self.support = support
        self.T = int(1 + torch.div(x.shape[-1] - self.N, stride, rounding_mode='floor'))    # time steps       
        self.F = int(1 + self.N/2)                                                          # nb of frequencies
        #self.L = x.shape[-1]                                                                # signal length
        #self.B = x.shape[0]                                                                 # batch size
        
        self.frame_starts = torch.arange(self.T) * self.stride  # Starting indices of the frames
        self.frame_centers = self.frame_starts + self.support // 2  # Centers of the frames
        self.frame_ends = self.frame_starts + self.support  # Ending indices of the frames
        self.spline_centers = None
        self.spline_map = None
        self.spline_degree = spline_degree
        self.spline_density = spline_density
        if win_min is None: self.min_length = 100
        if win_max is None: self.max_length = 1000
        self.window_function = window_function
        self.precomputed = False
        #self.warm_start = warm_start
        #self.times1 = []
        #self.times2 = []

        # window_lengths
        if mel_filter:
            win_length_size = (n_mels, self.T)
        else: 
            win_length_size = (self.F, self.T)
        self.window_lengths = nn.Parameter(torch.full(win_length_size, initial_win_length, dtype=self.dtype, device=self.device), requires_grad=True)

        # stft values of splines
        self.spline_stfts = None

    # Precomputation of STFT with spline windows
    def generate_spline(self):
        # Generate a B-spline of given degree over a support
        num_knots = self.spline_degree +2
        spline_support = int((self.spline_degree+1)*self.stride/self.spline_density)
        knots = np.linspace(0, spline_support, num_knots)
        #print(knots)
        knots = knots[1:-1]  # remove the endpoints
        knots = np.concatenate((
            [0] * (self.spline_degree+1),  #[knots[0]] * (spline_degree+1),  # left clamped
            knots,
            [spline_support] * (self.spline_degree+1)#[knots[-1]] * (spline_degree+1)  # right clamped
        ))
        num_basis = len(knots) - self.spline_degree  - 1
        t = np.linspace(0, spline_support, spline_support)
        coefs = np.zeros(num_basis)
        coefs[num_basis//2] = 1
        bspline = BSpline(knots, coefs, self.spline_degree)(t)

        return bspline
    def zero_pad(self, arr, target_length):
        # Ensure arr is 1D
        if arr.ndim == 2:
            arr = arr.squeeze()
        assert arr.ndim == 1, f"Expected 1D array after squeeze, got shape {arr.shape}"

        pad_len = target_length - arr.size(0)
        if pad_len < 0:
            raise ValueError(f"Target length {target_length} smaller than array length {arr.size(0)}")

        pad_left = pad_len // 2
        pad_right = pad_len - pad_left

        return F.pad(arr, (pad_left, pad_right), mode='constant', value=0)
    def precompute(self):  # calculate the stft with spline windows to later be combined 
        # Generate spline basis
        spline_basis = self.generate_spline()

        # Length of the B-spline basis function
        basis_length = len(spline_basis)

        # Compute spline centers based on stride, density, and spline basis length
        # Adjust the center so that it accounts for the length of the B-spline
        self.n_splines = int(1 + (len(self.x[0])-basis_length) * self.spline_density / self.stride)
        self.spline_vals_size = (self.F, self.n_splines)
        self.spline_stfts = torch.zeros((self.spline_vals_size[0], self.spline_vals_size[1]), dtype=torch.complex64)
        self.spline_centers = (torch.arange(self.n_splines) * self.stride // self.spline_density) + (basis_length // 2)

        # Calculate spline start and end indices based on the centers
        spline_support = int((self.spline_degree+1)*self.stride/self.spline_density)
        spline_starts = self.spline_centers - spline_support // 2
        spline_ends = self.spline_centers + spline_support // 2

        # Precompute frame starts and ends
        frame_starts = self.frame_starts.unsqueeze(0)  # Shape: (1, T)
        frame_ends = self.frame_ends.unsqueeze(0)  # Shape: (1, T)

        # Create an overlap mask for which splines overlap with which frames
        containment_mask = (spline_starts.unsqueeze(1) >= frame_starts) & (spline_ends.unsqueeze(1) <= frame_ends)
        
        # Store STFT results and prepare spline map
        self.spline_map = {frame_idx: [] for frame_idx in range(self.T)}

        for i, center in enumerate(self.spline_centers):
            # Extract and pad spline window
            start, end = spline_starts[i], spline_ends[i]
            spline_window = self.x[:, start:end] * spline_basis
            spline_window_padded = self.zero_pad(spline_window, self.N)

            # Compute FFT
            Zxx = torch.fft.fft(spline_window_padded)
        
            # Store STFT in self.spline_stfts
            self.spline_stfts[:, i] = Zxx[:self.F]

            # Update the spline map with relevant frame indices
            relevant_frames = containment_mask[i].nonzero().flatten()  # Indices where the spline is contained within with a frame
            for frame_idx in relevant_frames:
                self.spline_map[frame_idx.item()].append(i)

        # Calculate spline offsets in windows and prepare for beta coefficients
        self.offsets = self.spline_centers[self.spline_map.get(0, [])] - self.frame_centers[0]  # (n_relevant_splines,)
        #self.prepare_for_beta_coefficients(self.offsets)
        self.precomputed = True

    # Warm start, not super reliable or well motivated right now
    def enforce_ordering(self, value):  # helper function for warm start
        stride = self.stride
        margin = 0  # [0, stride)
        for i in range(value.shape[0]):  # Iterate over rows
            # too large from the left
            last_left_edge = -torch.inf
            for j in range(value.shape[1]):  # Go right over columns
                if stride*j -value[i, j]/2 -margin < last_left_edge:
                    value[i, j] = 2*(stride*j -last_left_edge -margin)
                last_left_edge = stride*j -value[i, j]/2
            # too large from the right
            last_right_edge = torch.inf
            for j in range(value.shape[1] - 1, -1, -1): # Go left over columns
                if stride*j +value[i, j]/2 +margin > last_right_edge:
                    value[i, j] = 2*(last_right_edge -margin -stride*j)
                last_right_edge = stride*j +value[i, j]/2
        return value
    def warm_start(self):
        with torch.no_grad():
            self.window_lengths.data.fill_(self.min_length)
        image = np.array(self().squeeze(0).detach())
        #plt.figure(figsize=[7, 5])
        #plt.imshow(image, cmap='jet', aspect='auto', interpolation='nearest')
        #plt.colorbar()
        #plt.show()

        kernel_size = (0, 0)
        sigma = 2
        smoothed_image = cv2.GaussianBlur(image, kernel_size, sigmaX=sigma, sigmaY=0)
        left_right_derivative = np.diff(smoothed_image, axis=1, append=smoothed_image[:, -1:])
        #left_right_2nd_derivative = np.abs(np.diff(left_right_derivative, axis=1, append=left_right_derivative[:, -1:]))
        #derivsum = np.abs(left_right_derivative)+np.abs(left_right_2nd_derivative)
        good_window_length = 1/(np.abs(left_right_derivative)+1e-7)
        good_window_length = 200*good_window_length
        good_window_length[good_window_length < 100] = 100
        good_window_length[good_window_length > 1000] = 1000
        sigma = 1
        smoothed_window_length = cv2.GaussianBlur(good_window_length, kernel_size, sigma)
        with torch.no_grad():
            smoothed_window_length = self.enforce_ordering(smoothed_window_length)
            self.window_lengths.data.copy_(torch.tensor(smoothed_window_length))
    
    # Spline coefficients for building larger windows
    def coefficients(self, lambda_):  # get the spline coefficients
        normalized_offsets = self.offsets / self.N
        # (n_relevant_splines)

        # The Greville abscissae are the centers of the splines! Kind of, boundary effect?
        if self.window_function == 'beta':
            a = 1/2* ( (3000/lambda_)**2 -1 ).unsqueeze(1)

            x = 0.5+normalized_offsets
            x_1mx = x*(1-x).unsqueeze(0)
            if len(a.shape)==3:  # if a has all time steps, we need to add a dimension
                x_1mx = x_1mx.unsqueeze(2)
        
            # going through logs to avoid numerical issues
            log_x_1mx = torch.log(x_1mx)
            log_pdf_ish = (a - 1) * log_x_1mx
            log_pdf = log_pdf_ish -log_pdf_ish.logsumexp(dim=1, keepdim=True)
            pdf = torch.exp(log_pdf)
            return pdf
        elif self.window_function == 'hann':
            half_width = (lambda_/(2 * self.N)).unsqueeze(1)
            # (n_frequencies, 1, T)  for normal forward

            normalized_offsets = normalized_offsets.unsqueeze(0)
            if len(half_width.shape)==3:  # if a has all time steps, we need to add a dimension
                normalized_offsets = normalized_offsets.unsqueeze(2)
            # (1, n_relevant_splines, 1)  for normal forward

            # normalized offsets is already centered
            out = 0.5 * (1 + torch.cos(torch.pi * normalized_offsets / half_width))
            out[torch.abs(normalized_offsets) > half_width] = 0
            out = out / out.sum(dim=1, keepdim=True)
            # (n_frequencies, n_relevant_splines, T)  for normal forward

            return out
        else:
            raise NotImplementedError(f"Window function '{self.window_function}' not implemented.")

    # Fast version with a lot of precomputations (right now the same haha)
    def forward(self):  # uses stft
        stft = self.stft()
        spec = stft.abs() + self.eps
        return spec.unsqueeze(0)
    def stft(self):  # fast, utilizes all precomputations
        if not self.precomputed:
            raise RuntimeError("Precompute the splines first using the precompute() method.")
        
        # Build tensor of relevant spline indices
        relevant_splines = [self.spline_map.get(t, []) for t in range(self.T)]
        relevant_splines = torch.tensor(relevant_splines, dtype=torch.long).T 
        # (n_relevant_splines, T)
        # Fetch relevant spline FFTs for each frame (with cool indexing)
        spline_stfts = self.spline_stfts[:, relevant_splines]
        # (n_frequencies, n_relevant_splines, T)
        
        # Modulate spline FFTs
        modulation_factors = torch.exp(-1j * 2 * torch.pi * torch.arange(self.F).unsqueeze(1) / self.N * self.offsets.unsqueeze(0)).unsqueeze(2)
        # (n_frequencies, n_relevant_splines, 1)
        modulated_stfts = spline_stfts * modulation_factors
        # (n_frequencies, n_relevant_splines, T)
        #self.times1.append(toc(print_elapsed=False))

        #tic()
        # Extract window lengths (lambdas)
        lambdas_ = self.window_lengths
        # (n_frequencies, T)

        # Compute coefficients
        coeffs = self.coefficients(lambdas_)  # self.get_beta_coefficients(lambdas_)  
        # (n_frequencies, n_relevant_splines, T)
    
        # Calculate the STFT components for all frames
        stft_components = coeffs * modulated_stfts
        # (n_frequencies, n_relevant_splines, T)

        # Sum over the relevant splines for each frame
        spectrogram = stft_components.sum(dim=1)  
        # (n_frequencies, T)
        #self.times2.append(toc(print_elapsed=False))

        return spectrogram

    # Low memory version, never stores large arrays (only utilizes spline precomputation)
    def low_memory_forward(self):
        stft = self.low_memory_stft()
        spec = stft.abs() + self.eps
        return spec.unsqueeze(0)
    def low_memory_stft(self):
        if not self.precomputed:
            raise RuntimeError("Precompute the splines first using the precompute() method.")

        # Initialize the spectrogram tensor (n_frames x n_frequencies)
        spectrogram = torch.zeros(self.spline_vals_size[0], self.T, dtype=torch.complex64)

        # Loop over all time steps
        for frame_idx in range(self.T):
            # Get the relevant splines for this frame from the precomputed mapping
            relevant_splines = self.spline_map.get(frame_idx, [])
            # Fetch relevant spline FFTs
            spline_stfts = self.spline_stfts[:, relevant_splines]  
            # (n_frequencies, n_relevant_splines)

            # Modulate the spline FFTs
            modulation_factors = torch.exp(-1j*2*torch.pi* torch.arange(self.F)/self.N *self.offsets.unsqueeze(1)).transpose(0, 1)  
            # (n_frequencies, n_relevant_splines)
            modulated_stfts = spline_stfts* modulation_factors  
            # (n_frequencies, n_relevant_splines)
            
            # Get the coefficient from the window lengths   
            lambdas_ = self.window_lengths[:, frame_idx]
            coeffs = self.coefficients(lambdas_)  # self.get_beta_coefficients(lambdas_)  
            # (n_frequencies, n_relevant_splines)

            # Compute the STFT components for all frequencies
            stft_components = coeffs* modulated_stfts  
            # (n_frequencies, n_relevant_splines)

            # Calculate the spectrogram entries for this time step
            spectrogram[:, frame_idx] += stft_components.sum(dim=1)
        return spectrogram

    # Put window lengths back within the allowed range during optimization
    def put_windows_back(self):
        minimum = self.min_length
        maximum = self.max_length
        with torch.no_grad():
            self.window_lengths.clamp_(min=minimum, max=maximum)

    # Plot the spectrogram and window lengths
    def plot(self, spec, weights: bool = True, title=""):
        plt.figure(figsize=(6.4, 4.8))
        plt.title("Spectrogram "+title)
        ax = plt.subplot()
        im = ax.imshow(spec[0].detach().cpu().log(), 
            aspect="auto", 
            origin="lower", 
            cmap="jet",
            interpolation='nearest',
            extent=[0, spec.shape[-1], 0, spec.shape[-2]])
        plt.ylabel("frequencies")
        plt.xlabel("frames")
        plt.colorbar(im, ax=ax)
        plt.show()

        if weights is True:
            plt.figure(figsize=(6.4, 4.8))
            plt.title("Distribution of window lengths "+title)
            ax = plt.subplot()
            im = ax.imshow(
                self.window_lengths.detach().cpu(),
                aspect="auto",
                origin="lower",
                cmap="jet",
                interpolation='nearest'
            )
            ax.set_ylabel("frequencies")
            ax.set_xlabel("frames")
            plt.colorbar(im, ax=ax)
            #im.set_clim(self.win_min, self.win_max)
            plt.show()
    
    # small gains with these, not worth it against flexibility
    """def prepare_for_beta_coefficients(self, offset):  # Normalized offsets for beta coefficients
        normalized_offset = offset / self.N
        x = 0.5+normalized_offset
        self.x_1mx = x*(1-x).unsqueeze(0)"""
    """def get_beta_coefficients(self, lambda_):
        #x_1mx = self.x_1mx
        normalized_offset = self.offsets / self.N
        x = 0.5+normalized_offset
        x_1mx = x*(1-x).unsqueeze(0)
        a = 1/2* ( (3000/lambda_)**2 -1 ).unsqueeze(1)
        if len(a.shape)==3:  # if a has all time steps, we need to add a dimension
            x_1mx = x_1mx.unsqueeze(2)
    
        # going through logs to avoid numerical issues
        log_x_1mx = torch.log(x_1mx)
        log_pdf_ish = (a - 1) * log_x_1mx
        log_pdf = log_pdf_ish -log_pdf_ish.logsumexp(dim=1, keepdim=True)
        pdf = torch.exp(log_pdf)
        return pdf"""

class DSTFTdev(nn.Module):
    def __init__(self, x,
                    win_length: float,   
                    support: int,                
                    stride: int, 
                    pow: float = 1.0, 
                    win_pow: float = 1.0,
                    win_p: str = None,
                    stride_p: str = None,
                    pow_p: str = None,       
                    win_requires_grad: bool = True,
                    stride_requires_grad: bool = True,
                    pow_requires_grad: bool = False,             
                    #params: str = 'p_tf', # p, p_t, p_f, p_tf
                    win_min = None,
                    win_max = None,
                    stride_min = None,
                    stride_max = None,    
                    
                    tapering_function: str = 'hann',
                    sr : int = 16_000,
                    window_transform = None,
                    stride_transform = None,
                    dynamic_parameter : bool = False,
                    first_frame : bool = False,
                    ):
        super().__init__()     
        
        # Constants and hyperparameters
        self.N = support                        # support size
        self.F = int(1 + self.N/2)              # nb of frequencies
        self.B = x.shape[0]                     # batch size
        self.L = x.shape[-1]                    # signal length
        self.device = x.device
        self.dtype = x.dtype
        
        self.win_requires_grad = win_requires_grad
        self.stride_requires_grad = stride_requires_grad
        self.pow_requires_grad = pow_requires_grad
        self.tapering_function = tapering_function
        self.dynamic_parameter = dynamic_parameter
        self.first_frame = first_frame
        self.sr = sr
        self.pow = pow
        self.tap_win = None
        
        # Register eps and min as a buffer tensor
        self.register_buffer('eps', torch.tensor(torch.finfo(torch.float).eps, dtype=self.dtype, device=self.device))
        self.register_buffer('min', torch.tensor(torch.finfo(torch.float).min, dtype=self.dtype, device=self.device))
        
        # Calculate the number of frames
        self.T = int(1 + torch.div(x.shape[-1] - (self.N - 1) - 1, stride, rounding_mode='floor'))        
        #self.T = int(1 + torch.div(self.L - self.N/2 , stride, rounding_mode='floor'))
        
        if win_min is None: self.win_min = self.N / 20 # 0
        else: self.win_min = win_min        
        if win_max is None: self.win_max = self.N
        else: self.win_max = win_max
        if stride_min is None: self.stride_min = 0
        else: self.stride_min = stride_min        
        if stride_max is None: self.stride_max = max(self.N, abs(stride))
        else: self.stride_max = stride_max
        
        # HOP LENGTH / FRAME INDEX
        if stride_transform is None: self.stride_transform = self.__stride_transform
        else: self.stride_transform = stride_transform
        # Determine the shape of the stride/hop-length/ frame index parameters
        if   stride_p is None: stride_size = (1,)
        elif stride_p == 't':  stride_size = (self.T,)   
        else: print('stride_p error', stride_p)
        # Create the window length parameter and assign it the appropriate shape
        self.strides = nn.Parameter(torch.full(stride_size, abs(stride), dtype=self.dtype, device=self.device), requires_grad=self.stride_requires_grad)     
        #print(self.strides)
        #self.stride = stride       
        
        # WIN LENGTH
        # win length constraints
        if window_transform is None: self.window_transform = self.__window_transform
        else: self.window_transform = window_transform
        # Determine the shape of the window length parameters
        if   win_p is None: win_length_size = (1, 1)
        elif win_p == 't':  win_length_size = (1, self.T)
        elif win_p == 'f':  win_length_size = (self.N, 1)
        elif win_p == 'tf': win_length_size = (self.N, self.T)
        else: print('win_p error', win_p)
        # Create the window length parameter and assign it the appropriate shape
        self.win_length = nn.Parameter(torch.full(win_length_size, abs(win_length), dtype=self.dtype, device=self.device), requires_grad=self.win_requires_grad)
        
        # WIN POW
        if   pow_p is None: win_pow_size = (1, 1)
        elif pow_p == 't':  win_pow_size = (1, self.T)
        elif pow_p == 'f':  win_pow_size = (self.N, 1)
        elif pow_p == 'tf': win_pow_size = (self.N, self.T)
        else: print('pow_p error', pow_p)
        self.win_pow = nn.Parameter(torch.full(win_pow_size, abs(win_pow), dtype=self.dtype, device=self.device), requires_grad=self.pow_requires_grad)


    def __window_transform(self, w_in): #
        w_out = torch.minimum(torch.maximum(w_in, torch.full_like(w_in, self.win_min, dtype=self.dtype, device=self.device)), torch.full_like(w_in, self.win_max, dtype=self.dtype, device=self.device))
        return w_out
    
    def __stride_transform(self, s_in): # born stride entre 0 et 2N 
        s_out = torch.minimum(torch.maximum(s_in, torch.full_like(s_in, self.stride_min, dtype=self.dtype, device=self.device)), torch.full_like(s_in, self.stride_max, dtype=self.dtype, device=self.device))
        return s_out 
    
    @property 
    def actual_win_length(self): # contraints
        return self.window_transform(self.win_length)
    
    @property 
    def actual_strides(self): # stride contraints, actual stride between frames
        return self.stride_transform(self.strides)
    
    @property
    def frames(self):
        # Compute the temporal position (indices) of frames (support)
        expanded_stride = self.actual_strides.expand((self.T,))    
        frames = torch.zeros_like(expanded_stride)
        if self.first_frame:
            frames[0] = (self.actual_win_length.expand((self.N, self.T))[:, 0].max(dim=0, keepdim=False)[0] - self.N)/2
        frames[1:] = frames[0] + expanded_stride[1:].cumsum(dim=0)        
        
        #frames = torch.cumsum(self.actual_strides, dim=0)
        #print(frames)
        return frames
    
    @property 
    def effective_strides(self):
        # Compute the strides between window (and not frames)
        expanded_stride = self.actual_strides.expand((self.T,))    
        effective_strides = torch.zeros_like(expanded_stride)
        effective_strides[1:] = expanded_stride[1:]
        cat = torch.cat((torch.tensor([self.N], dtype=self.dtype, device=self.device), self.actual_win_length.expand((self.N, self.T)).max(dim=0, keepdim=False)[0]), dim=0).diff()/2        
        effective_strides = effective_strides - cat
        return effective_strides
    
    def forward(self, x):
        # Perform the forward STFT and extract the magnitude, phase, real, and imaginary parts
        stft = self.stft(x, 'forward')
        real, imag, spec, phase = stft.real, stft.imag, stft.abs().pow(self.pow)[:, :self.F], stft.angle()[:, :self.F]
        spec = spec + torch.finfo(x.dtype).eps
        return spec, stft, real, imag, phase
    
    def backward(self, x, dl_ds):
        # Compute the gradient of the loss w.r.t. window length parameter with the chain rule
        dstft_dp = self.stft(x, 'backward')
        dl_dp = (torch.conj(dl_ds) * dstft_dp).sum().real #dl_dp = (dl_ds.real * dstft_dp.real + dl_ds.imag * dstft_dp.imag).sum()        
        return dl_dp.unsqueeze(0)
    
    def stft(self, x: torch.tensor, direction: str):
        #batch_size, length, device, dtype = x.shape[0], x.shape[-1], x.device, x.dtype     
        
        # Generate strided signal and shift idx_frac
        strided_x, idx_frac = self.stride(x) # B, T, N; T

        # Generate the tapering window function for the STFT
        self.tap_win = self.window_function(direction=direction, idx_frac=idx_frac).permute(2, 1, 0)  # T, N, N
        
        # Generate tapering function shift   
        shift = torch.arange(end=self.N, device=self.device, dtype=self.dtype, requires_grad=False)
        shift = idx_frac[:, None] * shift[None, :] # T, N
        
        # Compute tapered x
        strided_x = strided_x[:, :, None, :] # B, T, 1, N
        self.tap_win = self.tap_win[None, :, :, :] # 1, T, N, 1
        shift = torch.exp( 2j * pi * shift / self.N)[None, :, :, None] # 1, T, N, 1
        tapered_x = strided_x * self.tap_win * shift # B, T, N, N
        
        # Generate Fourier coefficients    
        coeff = torch.arange(end=self.N, device=self.device, dtype=self.dtype, requires_grad=False) 
        coeff = coeff[:, None] @ coeff[None, :]
        coeff = torch.exp(- 2j * pi * coeff / self.N) # N, N
        
        # Perform the STFT
        coeff = coeff[None, None, :, :]  # 1, 1, N, N     
        stft = (tapered_x * coeff).sum(dim=-1)  # B, T, N        #stft = torch.einsum('...ij,...jk->...ik', tapered_x, coeff)
        stft = stft.permute(0, 2, 1)         # B, N, T           #stft = stft.transpose(-1, -2)  

            
        return stft
    
    def stride(self, x) -> torch.tensor:                
        # frames index and strided x
        idx_floor = self.frames.floor()
        idx_frac = self.frames - idx_floor
        idx_floor = idx_floor.long()[:, None].expand((self.T, self.N)) + torch.arange(0, self.N, device=self.device)
        idx_floor[idx_floor >= self.L] = -1
        strided_x = x[:, idx_floor]
        strided_x[:, idx_floor < 0] = 0
        return strided_x, idx_frac
    
    def window_function(self, direction: str, idx_frac) -> torch.tensor:
        if self.tapering_function not in {'hann', 'hanning',}:
            raise ValueError(f"tapering_function must be one of '{'hann', 'hanning',}', but got padding_mode='{self.tapering_function}'")
        else:
            # Create an array of indices to use as the base for the window function
            base = torch.arange(0, self.N, 1, dtype=self.dtype, device=self.device)[:, None, None].expand([-1, self.N, self.T])   
            base = base - idx_frac
            # Expand the win_length parameter to match the shape of the base array         
            #if self.actual_win_length.dim() == 3:
            #    self.expanded_win_length = self.actual_win_length.expand([self.N, self.N, self.T])
            #elif self.actual_win_length.dim() == 1:
            #    self.expanded_win_length = self.actual_win_length[:, None, None].expand([self.N, self.N, self.T])
            #elif self.actual_win_length.dim() == 2 and self.actual_win_length.shape[-1] == self.T:
            #    self.expanded_win_length = self.actual_win_length[:, None, :].expand([self.N, self.N, self.T])
            #elif self.actual_win_length.dim() == 2 and self.actual_win_length.shape[-1] == self.N: 
            #    self.expanded_win_length = self.actual_win_length[:, :, None].expand([self.N, self.N, self.T])          
        
        # calculate the tapering function and its derivate w.r.t. window length
        if self.tapering_function == 'hann' or self.tapering_function == 'hanning':
            if direction == 'forward':
                self.tap_win = 0.5 - 0.5 * torch.cos(2 * pi * (base + (self.actual_win_length-self.N+1)/2) / self.actual_win_length )                
                mask1 = base.ge(torch.ceil( (self.N-1+self.actual_win_length)/2))
                mask2 = base.le(torch.floor((self.N-1-self.actual_win_length)/2))            
                self.tap_win[mask1] = 0
                self.tap_win[mask2] = 0
                self.tap_win = self.tap_win / self.tap_win.sum(dim=0, keepdim=True) 
                return self.tap_win.pow(self.win_pow)
            
            elif direction == 'backward':
                f = torch.sin(2 * pi * (base - (self.N-1)/2) / self.actual_win_length)            
                d_tap_win = - pi / self.actual_win_length * ((self.N-1)/2 - base) * f
                d_tap_win[mask1] = 0
                d_tap_win[mask2] = 0
                return d_tap_win
            
    def coverage(self): # in [0, 1]    
        # compute coverage
        expanded_win, _ = self.actual_win_length.expand((self.N, self.T)).min(dim=0, keepdim=False)
        cov = expanded_win[0]
        maxi = self.frames[0] + self.N/2 + expanded_win[0]/2        
        for i in range(1, self.T):
            start = torch.min(self.L*torch.ones_like(expanded_win[i]), torch.max(torch.zeros_like(expanded_win[i]), self.frames[i] + self.N/2 - expanded_win[i]/2))
            end = torch.min(self.L*torch.ones_like(expanded_win[i]), torch.max(torch.zeros_like(expanded_win[i]), self.frames[i] + self.N/2 + expanded_win[i]/2))
            if end > maxi:
                cov += end - torch.max(start, maxi)
                maxi = end
        cov /= self.L
        return cov

    def print(self, spec, x=None, marklist=None, weights=True, wins=True, bar=False):
        plt.figure()
        plt.title('Spectrogram')
        ax = plt.subplot()
        im = ax.imshow(spec[0].detach().cpu().log(), aspect='auto', origin='lower', cmap='jet', extent=[0,spec.shape[-1], 0, spec.shape[-2]])
        plt.ylabel('frequencies')
        plt.xlabel('frames')
        if bar == True: plt.colorbar(im, ax=ax)
        plt.show()

        if weights == True:
            plt.figure()
            plt.title('Distribution of window lengths')
            ax = plt.subplot()
            im = ax.imshow(self.actual_win_length[:self.F].detach().cpu(), aspect='auto', origin='lower', cmap='jet')
            ax.set_ylabel('frequencies')
            ax.set_xlabel('frames')
            if bar == True : 
                plt.colorbar(im, ax=ax)
                im.set_clim(self.win_min, self.win_max)   
            plt.show()   

        if self.tap_win is not None and wins == True:
            fig, ax = plt.subplots()
            ax.plot(self.T + .5 + x.squeeze().cpu().numpy(), linewidth=1,)
            for i, start in enumerate(self.frames.detach().cpu()):
                ax.plot(range(int(start.floor().item()), int(start.floor().item()+self.N)), self.T-i-1.3 + 150 * self.tap_win[:, i, :, :].mean(dim=1).squeeze().detach().cpu(), c='#1f77b4')

            if marklist is not None:
                for elem in marklist:
                    plt.axvline(elem, 0, self.T, c='gray')
            else:
                ax.axvline(x=0, ymin=0, ymax=self.T, c='gray')
                ax.axvline(x=x.shape[-1], ymin=0, ymax=self.T, c='gray')
            plt.show()

class DSTFT(nn.Module):
    """Differentiable short-time Fourier transform (DSTFT) module.

    Args:
    ----
        nn (_type_): _description_

    """

    def __init__(
        self: DSTFT,
        x: torch.tensor,
        win_length: float,
        support: int,
        stride: int,
        pow: float = 1.0,
        win_pow: float = 1.0,
        win_p: str | None = None,
        stride_p: str | None = None,
        pow_p: str | None = None,
        win_requires_grad=True,
        stride_requires_grad: bool = True,
        pow_requires_grad: bool = False,
        # params: str = 'p_tf', # p, p_t, p_f, p_tf
        win_min: float | None = None,
        win_max: float | None = None,
        stride_min: float | None = None,
        stride_max: float | None = None,
        pow_min: float | None = None,
        pow_max: float | None = None,
        tapering_function: str = "hann",
        sr: int = 16_000,
        window_transform=None,
        stride_transform=None,
        dynamic_parameter: bool = False,
        first_frame: bool = False,
    ):
        super().__init__()

        # Constants and hyperparameters
        self.N = support  # support size
        self.F = int(1 + self.N / 2)  # nb of frequencies
        self.B = x.shape[0]  # batch size
        self.L = x.shape[-1]  # signal length
        self.device = x.device
        self.dtype = x.dtype

        self.win_requires_grad = win_requires_grad
        self.stride_requires_grad = stride_requires_grad
        self.pow_requires_grad = pow_requires_grad
        self.tapering_function = tapering_function
        self.dynamic_parameter = dynamic_parameter
        self.first_frame = first_frame
        self.sr = sr
        self.pow = pow
        self.tap_win = None

        # Register eps and min as a buffer tensor
        self.register_buffer(
            "eps",
            torch.tensor(
                torch.finfo(torch.float).eps,
                dtype=self.dtype,
                device=self.device,
            ),
        )
        self.register_buffer(
            "min",
            torch.tensor(
                torch.finfo(torch.float).min,
                dtype=self.dtype,
                device=self.device,
            ),
        )

        # Calculate the number of frames
        # self.T = int(
        #    1
        #    + torch.div(
        #        x.shape[-1] - (self.N - 1) - 1, stride, rounding_mode="floor",
        #    ),
        # )
        self.T = 1 + int(torch.div(x.shape[-1], stride, rounding_mode="floor"))

        if win_min is None:
            self.win_min = self.N / 20
        else:
            self.win_min = win_min
        if win_max is None:
            self.win_max = self.N
        else:
            self.win_max = win_max
        if stride_min is None:
            self.stride_min = 0
        else:
            self.stride_min = stride_min
        if stride_max is None:
            self.stride_max = max(self.N, abs(stride))
        else:
            self.stride_max = stride_max
        if pow_min is None:
            self.pow_min = 0.001
        else:
            self.pow_min = pow_min
        if pow_max is None:
            self.pow_max = 1000
        else:
            self.pow_max = pow_max

        # HOP LENGTH / FRAME INDEX
        # hop length constraints
        if stride_transform is None:
            self.stride_transform = self.__stride_transform
        else:
            self.stride_transform = stride_transform

        # Determine the shape of the stride/hop-length/ frame index parameters
        if stride_p is None:
            stride_size = (1,)
        elif stride_p == "t":
            stride_size = (self.T,)
        else:
            raise ValueError(f"stride_p error {stride_p}")
        # Create the window length parameter and assign it the appropriate shape
        self.init_stride = abs(stride)
        self.strides = nn.Parameter(
            torch.full(
                stride_size,
                self.init_stride,
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.stride_requires_grad,
        )

        # WIN LENGTH
        # win length constraints
        if window_transform is None:
            self.window_transform = self.__window_transform
        else:
            self.window_transform = window_transform

        # Determine the shape of the window length parameters
        if win_p is None:
            win_length_size = (1, 1)
        elif win_p == "t":
            win_length_size = (1, self.T)
        else:
            raise ValueError(
                f"win_p error {win_p}, maybe use ADSTFT instead (frequency varying windows)",
            )
        # Create the window length parameter and assign it the appropriate shape
        self.win_length = nn.Parameter(
            torch.full(
                win_length_size,
                abs(win_length),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.win_requires_grad,
        )

        # WIN POW
        if pow_p is None:
            win_pow_size = (1, 1)
        elif pow_p == "t":
            win_pow_size = (1, self.T)
        else:
            print("pow_p error", pow_p)
        self.win_pow = nn.Parameter(
            torch.full(
                win_pow_size,
                abs(win_pow),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.pow_requires_grad,
        )

    def __window_transform(self: DSTFT, w_in):
        w_out = torch.minimum(
            torch.maximum(
                w_in,
                torch.full_like(
                    w_in, self.win_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                w_in, self.win_max, dtype=self.dtype, device=self.device,
            ),
        )
        return w_out

    def __stride_transform(
        self: DSTFT, s_in: torch.Tensor,
    ):  # born stride entre 0 et 2N
        s_out = torch.minimum(
            torch.maximum(
                s_in,
                torch.full_like(
                    s_in, self.stride_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                s_in, self.stride_max, dtype=self.dtype, device=self.device,
            ),
        )
        return s_out

    def __pow_transform(
        self: DSTFT, p_in: torch.Tensor,
    ):  # born stride entre 0 et 2N
        p_out = torch.minimum(
            torch.maximum(
                p_in,
                torch.full_like(
                    p_in, self.pow_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                p_in, self.pow_max, dtype=self.dtype, device=self.device,
            ),
        )
        return p_out

    @property
    def actual_win_length(self: DSTFT):  # contraints
        return self.window_transform(self.win_length)

    @property
    def actual_strides(
        self,
    ):  # stride contraints, actual stride between frames
        return self.stride_transform(self.strides)

    @property
    def actual_pow(self: DSTFT):  # pow contraints
        return self.pow_transform(self.win_pow)

    @property
    def frames(self: DSTFT):
        # Compute the temporal position (indices) of frames (support)
        expanded_stride = self.actual_strides.expand((self.T,))
        frames = torch.zeros_like(expanded_stride)
        # frames[0] = - self.N / 2
        # if self.first_frame:
        #    frames[0] = (
        #        self.actual_win_length.expand((self.N, self.T))[:, 0].max(
        #            dim=0, keepdim=False,
        #        )[0]
        #        - self.N
        #    ) / 2
        frames -= self.N / 2 + self.init_stride
        frames += expanded_stride.cumsum(dim=0)

        return frames

    @property
    def effective_strides(self: DSTFT):
        # Compute the strides between window (and not frames)
        expanded_stride = self.actual_strides.expand((self.T,))
        effective_strides = torch.zeros_like(expanded_stride)
        effective_strides[1:] = expanded_stride[1:]
        cat = (
            torch.cat(
                (
                    torch.tensor(
                        [self.N], dtype=self.dtype, device=self.device,
                    ),
                    self.actual_win_length.expand((self.N, self.T)).max(
                        dim=0, keepdim=False,
                    )[0],
                ),
                dim=0,
            ).diff()
            / 2
        )
        effective_strides = effective_strides - cat
        return effective_strides

    def forward(self: DSTFT, x: torch.tensor) -> tuple:
        # Perform the forward STFT and extract the magnitude, phase, real, and imaginary parts
        stft = self.stft(x, "forward")
        spec = stft.abs().pow(self.pow)[:, : self.F] + torch.finfo(x.dtype).eps
        return spec, stft  # , real, imag, phase

    def backward(
        self: DSTFT, x: torch.tensor, dl_ds: torch.tensor,
    ) -> torch.tensor:
        # Compute the gradient of the loss w.r.t. window length parameter with the chain rule
        dstft_dp = self.stft(x, "backward")
        dl_dp = torch.conj(dl_ds) * dstft_dp
        dl_dp = dl_dp.sum().real.expand(self.win_length.shape)
        return dl_dp

    def stft(self: DSTFT, x: torch.tensor, direction: str):
        # batch_size, length, device, dtype = x.shape[0], x.shape[-1], x.device, x.dtype

        # Generate strided signal and shift idx_frac
        folded_x, idx_frac = self.unfold(x)  # B, T, N; T

        # Generate the tapering window function for the STFT
        self.tap_win = self.window_function(
            direction=direction, idx_frac=idx_frac,
        ).permute(1, 0)  # T, N

        # Compute tapered x
        self.folded_x = folded_x[:, :, :]  # B, T, N
        self.tap_win = self.tap_win[None, :, :]  # 1, T, 1
        self.tapered_x = self.folded_x * self.tap_win  # B, T, N,

        spectr = torch.fft.rfft(self.tapered_x)

        shift = torch.arange(
            end=self.F,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        shift = idx_frac[:, None] * shift[None, :]  # T, N
        shift = torch.exp(2j * pi * shift / self.N)[None, ...]  # 1, T, N

        stft = spectr * shift
        return stft.permute(0, 2, 1)

    def inverse_dstft(self: DSTFT, stft: torch.Tensor) -> torch.tensor:
        """Compute inverse differentiable short-time Fourier transform (IDSTFT).

        Args:
        ----
            self (DSTFT): _description_
            stft (torch.Tensor): _description_

        Returns:
        -------
            torch.tensor: _description_

        """
        # shift
        # shift = torch.arange(
        #     end=self.F,
        #     device=self.device,
        #     dtype=self.dtype,
        #     requires_grad=False,
        # )
        # shift = idx_frac[:, None] * shift[None, :]  # T, N
        # stft = stft * torch.exp(2j * pi * shift / self.N)[None, ...]  # 1, T, N
        # print(stft.shape)

        # inverse
        # print(stft.shape, stft.dtype)
        ifft = torch.fft.irfft(stft, n=self.N, dim=-2)
        # print(ifft.shape, self.tap_win.sum(-1, keepdim=True).shape)
        
        # add shift
        self.itap_win = self.synt_win(None, None)
        ifft = ifft.permute(0, -1, -2) * self.itap_win
        
        # fold
        x_hat = self.fold(ifft)

        return x_hat

    def unfold(self: DSTFT, x: torch.tensor) -> torch.tensor:
        # frames index and strided x
        idx_floor = self.frames.floor()
        # print(self.frames.shape, self.frames)
        idx_frac = self.frames - idx_floor
        idx_floor = idx_floor.long()[:, None].expand((
            self.T,
            self.N,
        )) + torch.arange(0, self.N, device=self.device)
        idx_floor[idx_floor >= self.L] = -1
        # print(self.B, idx_floor.shape, x.shape)
        folded_x = x[:, idx_floor]
        folded_x[:, idx_floor < 0] = 0
        return folded_x, idx_frac

    def fold(self: DSTFT, folded_x: torch.tensor) -> torch.tensor:
        x_hat = torch.zeros(
            self.B, self.L, device=self.device, dtype=self.dtype,
        )
        # print(x_hat.shape, self.B, self.L)
        #print(folded_x.shape)
        for t in range(self.T):
            start_idx = max(0, int(self.frames[t]))
            end_idx = min(self.L - 1, int(self.frames[t]) + self.N)
            start_dec = start_idx - int(self.frames[t])
            end_dec = end_idx - int(self.frames[t])
            x_hat[:, start_idx:end_idx] += folded_x[:, t, start_dec:end_dec]
        return x_hat

    def window_function(self: DSTFT, direction: str, idx_frac) -> torch.tensor:
        if self.tapering_function not in {"hann", "hanning"}:
            raise ValueError(
                f"tapering_function must be one of '{('hann', 'hanning')}', but got padding_mode='{self.tapering_function}'",
            )
        else:
            # Create an array of indices to use as the base for the window function
            base = torch.arange(
                0, self.N, 1, dtype=self.dtype, device=self.device,
            )[:, None].expand([-1, self.T])
            base = base - idx_frac
            # Expand the win_length parameter to match the shape of the base array

        # calculate the tapering function and its derivate w.r.t. window length
        mask1 = base.ge(torch.ceil((self.N - 1 + self.actual_win_length) / 2))
        mask2 = base.le(torch.floor((self.N - 1 - self.actual_win_length) / 2))
        if (
            self.tapering_function == "hann"
            or self.tapering_function == "hanning"
        ):
            if direction == "forward":
                self.tap_win = 0.5 - 0.5 * torch.cos(
                    2
                    * pi
                    * (base + (self.actual_win_length - self.N + 1) / 2)
                    / self.actual_win_length,
                )
                self.tap_win[mask1] = 0
                self.tap_win[mask2] = 0
                #print(self.tap_win.max())
                #self.tap_win = self.N* self.tap_win / self.tap_win.sum(dim=0, keepdim=True) # self.N* 
                #print(self.tap_win.max())
                #print(self.tap_win.shape)
                #print(self.N)
                #s = torch.sqrt(self.N /torch.sum(self.tap_win ** 2, dim=0))
                #print(s.shape)
                #self.tap_win = s* self.tap_win
                return self.tap_win.pow(self.win_pow)

            elif direction == "backward":
                f = torch.sin(
                    2
                    * pi
                    * (base + (self.actual_win_length - self.N + 1) / 2)
                    / self.actual_win_length,
                )
                d_tap_win = (
                    -pi
                    / self.actual_win_length.pow(2)
                    * ((self.N - 1) / 2 - base)
                    * f
                )
                d_tap_win[mask1] = 0
                d_tap_win[mask2] = 0
                d_tap_win = d_tap_win / self.N * 2
                return d_tap_win
        return None

    def synt_win(self: DSTFT, direction: str, idx_frac) -> torch.tensor:
        
        wins = torch.zeros(self.L)        
        for t in range(self.T):
            start_idx = max(0, int(self.frames[t]))
            end_idx = min(self.L, int(self.frames[t]) + self.N)
            start_dec = start_idx - int(self.frames[t])
            end_dec = end_idx - int(self.frames[t])
            wins[start_idx:end_idx] += (
                self.tap_win[:, t, start_dec:end_dec].squeeze().detach().cpu()
            )
        self.wins = wins
        self.iwins = torch.zeros(self.L)
        self.iwins[ self.wins > 0 ] = 1 / self.wins[self.wins > 0]
        
        plt.plot(self.wins, label="wins")
        plt.plot(self.iwins, label='iwins')
        plt.plot(self.iwins * self.wins)
        plt.legend()
        
        itap_win = torch.zeros_like(self.tap_win)
        for t in range(self.T):
            start_idx = max(0, int(self.frames[t]))
            end_idx = min(self.L, int(self.frames[t]) + self.N)
            start_dec = start_idx - int(self.frames[t])
            end_dec = end_idx - int(self.frames[t])
            itap_win[:, t, start_dec:end_dec] = (
            #self.tap_win[:, t, start_dec:end_dec]
            #/ wins[start_idx:end_idx] * 
            self.iwins[start_idx:end_idx]
        )
        return itap_win

    def coverage(self: DSTFT):  # in [0, 1]
        # compute coverage
        expanded_win, _ = self.actual_win_length.expand((self.N, self.T)).min(
            dim=0, keepdim=False,
        )
        cov = expanded_win[0]
        maxi = self.frames[0] + self.N / 2 + expanded_win[0] / 2
        for i in range(1, self.T):
            start = torch.min(
                self.L * torch.ones_like(expanded_win[i]),
                torch.max(
                    torch.zeros_like(expanded_win[i]),
                    self.frames[i] + self.N / 2 - expanded_win[i] / 2,
                ),
            )
            end = torch.min(
                self.L * torch.ones_like(expanded_win[i]),
                torch.max(
                    torch.zeros_like(expanded_win[i]),
                    self.frames[i] + self.N / 2 + expanded_win[i] / 2,
                ),
            )
            if end > maxi:
                cov += end - torch.max(start, maxi)
                maxi = end
        cov /= self.L
        return cov

    def plot(
        self: DSTFT,
        spec: torch.Tensor,
        x: torch.Tensor | None = None,
        marklist: Optional[List[Any]] = None,
        figsize=(6.4, 4.8),
        f_hat=None,
        fs=None,
        *,
        weights: bool = True,
        wins: bool = True,
        bar: bool = False,
        cmap: float = "jet",
        ylabel: float = "frequencies",
        xlabel: float = "frames",
    ):
        f_max = spec.shape[-2] if fs is None else fs / 2
        plt.figure(figsize=figsize)
        plt.title("Spectrogram")
        ax = plt.subplot()
        im = ax.imshow(
            spec[0].detach().cpu().log(),
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=[0, spec.shape[-1], 0, f_max],
        )
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if bar == True:
            plt.colorbar(im, ax=ax)
        if f_hat is not None:
            for f in f_hat:
                plt.plot(f, linewidth=0.5, c="k", alpha=0.7)
        plt.show()

        if weights == True:
            plt.figure(figsize=figsize)
            plt.title("Distribution of window lengths")
            ax = plt.subplot()
            im = ax.imshow(
                self.actual_win_length[: self.F].detach().cpu(),
                aspect="auto",
                origin="lower",
                cmap=cmap,
            )
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            if bar == True:
                plt.colorbar(im, ax=ax)
                im.set_clim(self.win_min, self.win_max)
            plt.show()

        if self.tap_win is not None and wins == True:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(self.T + 0.5 + x.squeeze().cpu().numpy(), linewidth=1)
            for i, start in enumerate(self.frames.detach().cpu()):
                ax.plot(
                    range(
                        int(start.floor().item()),
                        int(start.floor().item() + self.N),
                    ),
                    self.T
                    - i
                    - 1.3
                    + self.tap_win[:, i, :].squeeze().detach().cpu(),
                    c="#1f77b4",
                )

            if marklist is not None:
                for elem in marklist:
                    plt.axvline(elem, 0, self.T, c="gray")
            else:
                ax.axvline(x=0, ymin=0, ymax=self.T, c="gray")
                ax.axvline(x=x.shape[-1], ymin=0, ymax=self.T, c="gray")
            plt.show()

# fixed (one line)
class DSTFTabs(nn.Module):
    """Differentiable short-time Fourier transform (DSTFT) module.

    Args:
    ----
        nn (_type_): _description_

    """

    def __init__(
        self: DSTFT,
        x: torch.tensor,
        win_length: float,
        support: int,
        stride: int,
        pow: float = 1.0,
        win_pow: float = 1.0,
        win_p: str | None = None,
        stride_p: str | None = None,
        pow_p: str | None = None,
        win_requires_grad=True,
        stride_requires_grad: bool = True,
        pow_requires_grad: bool = False,
        # params: str = 'p_tf', # p, p_t, p_f, p_tf
        win_min: float | None = None,
        win_max: float | None = None,
        stride_min: float | None = None,
        stride_max: float | None = None,
        pow_min: float | None = None,
        pow_max: float | None = None,
        tapering_function: str = "hann",
        sr: int = 16_000,
        window_transform=None,
        stride_transform=None,
        dynamic_parameter: bool = False,
        first_frame: bool = False,
    ):
        super().__init__()

        # Constants and hyperparameters
        self.N = support  # support size
        self.F = int(1 + self.N / 2)  # nb of frequencies
        self.B = x.shape[0]  # batch size
        self.L = x.shape[-1]  # signal length
        self.device = x.device
        self.dtype = x.dtype

        self.win_requires_grad = win_requires_grad
        self.stride_requires_grad = stride_requires_grad
        self.pow_requires_grad = pow_requires_grad
        self.tapering_function = tapering_function
        self.dynamic_parameter = dynamic_parameter
        self.first_frame = first_frame
        self.sr = sr
        self.pow = pow
        self.tap_win = None

        # Register eps and min as a buffer tensor
        self.register_buffer(
            "eps",
            torch.tensor(
                torch.finfo(torch.float).eps,
                dtype=self.dtype,
                device=self.device,
            ),
        )
        self.register_buffer(
            "min",
            torch.tensor(
                torch.finfo(torch.float).min,
                dtype=self.dtype,
                device=self.device,
            ),
        )

        # Calculate the number of frames
        # self.T = int(
        #    1
        #    + torch.div(
        #        x.shape[-1] - (self.N - 1) - 1, stride, rounding_mode="floor",
        #    ),
        # )
        self.T = 1 + int(torch.div(x.shape[-1], stride, rounding_mode="floor"))

        if win_min is None:
            self.win_min = self.N / 20
        else:
            self.win_min = win_min
        if win_max is None:
            self.win_max = self.N
        else:
            self.win_max = win_max
        if stride_min is None:
            self.stride_min = 0
        else:
            self.stride_min = stride_min
        if stride_max is None:
            self.stride_max = max(self.N, abs(stride))
        else:
            self.stride_max = stride_max
        if pow_min is None:
            self.pow_min = 0.001
        else:
            self.pow_min = pow_min
        if pow_max is None:
            self.pow_max = 1000
        else:
            self.pow_max = pow_max

        # HOP LENGTH / FRAME INDEX
        # hop length constraints
        if stride_transform is None:
            self.stride_transform = self.__stride_transform
        else:
            self.stride_transform = stride_transform

        # Determine the shape of the stride/hop-length/ frame index parameters
        if stride_p is None:
            stride_size = (1,)
        elif stride_p == "t":
            stride_size = (self.T,)
        else:
            raise ValueError(f"stride_p error {stride_p}")
        # Create the window length parameter and assign it the appropriate shape
        self.init_stride = abs(stride)
        self.strides = nn.Parameter(
            torch.full(
                stride_size,
                self.init_stride,
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.stride_requires_grad,
        )

        # WIN LENGTH
        # win length constraints
        if window_transform is None:
            self.window_transform = self.__window_transform
        else:
            self.window_transform = window_transform

        # Determine the shape of the window length parameters
        if win_p is None:
            win_length_size = (1, 1)
        elif win_p == "t":
            win_length_size = (1, self.T)
        else:
            raise ValueError(
                f"win_p error {win_p}, maybe use ADSTFT instead (frequency varying windows)",
            )
        # Create the window length parameter and assign it the appropriate shape
        self.win_length = nn.Parameter(
            torch.full(
                win_length_size,
                abs(win_length),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.win_requires_grad,
        )

        # WIN POW
        if pow_p is None:
            win_pow_size = (1, 1)
        elif pow_p == "t":
            win_pow_size = (1, self.T)
        else:
            print("pow_p error", pow_p)
        self.win_pow = nn.Parameter(
            torch.full(
                win_pow_size,
                abs(win_pow),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.pow_requires_grad,
        )

    def __window_transform(self: DSTFT, w_in):
        w_out = torch.minimum(
            torch.maximum(
                w_in,
                torch.full_like(
                    w_in, self.win_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                w_in, self.win_max, dtype=self.dtype, device=self.device,
            ),
        )
        return w_out

    def __stride_transform(
        self: DSTFT, s_in: torch.Tensor,
    ):  # born stride entre 0 et 2N
        s_out = torch.minimum(
            torch.maximum(
                s_in,
                torch.full_like(
                    s_in, self.stride_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                s_in, self.stride_max, dtype=self.dtype, device=self.device,
            ),
        )
        return s_out

    def __pow_transform(
        self: DSTFT, p_in: torch.Tensor,
    ):  # born stride entre 0 et 2N
        p_out = torch.minimum(
            torch.maximum(
                p_in,
                torch.full_like(
                    p_in, self.pow_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                p_in, self.pow_max, dtype=self.dtype, device=self.device,
            ),
        )
        return p_out

    @property
    def actual_win_length(self: DSTFT):  # contraints
        return self.window_transform(self.win_length)

    @property
    def actual_strides(
        self,
    ):  # stride contraints, actual stride between frames
        return self.stride_transform(self.strides)

    @property
    def actual_pow(self: DSTFT):  # pow contraints
        return self.pow_transform(self.win_pow)

    @property
    def frames(self: DSTFT):
        # Compute the temporal position (indices) of frames (support)
        expanded_stride = self.actual_strides.expand((self.T,))
        frames = torch.zeros_like(expanded_stride)
        # frames[0] = - self.N / 2
        # if self.first_frame:
        #    frames[0] = (
        #        self.actual_win_length.expand((self.N, self.T))[:, 0].max(
        #            dim=0, keepdim=False,
        #        )[0]
        #        - self.N
        #    ) / 2
        frames -= self.N / 2 + self.init_stride
        frames += expanded_stride.cumsum(dim=0)

        return frames

    @property
    def effective_strides(self: DSTFT):
        # Compute the strides between window (and not frames)
        expanded_stride = self.actual_strides.expand((self.T,))
        effective_strides = torch.zeros_like(expanded_stride)
        effective_strides[1:] = expanded_stride[1:]
        cat = (
            torch.cat(
                (
                    torch.tensor(
                        [self.N], dtype=self.dtype, device=self.device,
                    ),
                    self.actual_win_length.expand((self.N, self.T)).max(
                        dim=0, keepdim=False,
                    )[0],
                ),
                dim=0,
            ).diff()
            / 2
        )
        effective_strides = effective_strides - cat
        return effective_strides

    def forward(self: DSTFT, x: torch.tensor) -> tuple:
        # Perform the forward STFT and extract the magnitude, phase, real, and imaginary parts
        stft = self.stft(x, "forward")
        spec = stft.abs().pow(self.pow)[:, : self.F] + torch.finfo(x.dtype).eps
        return spec, stft  # , real, imag, phase

    def backward(
        self: DSTFT, x: torch.tensor, dl_ds: torch.tensor,
    ) -> torch.tensor:
        # Compute the gradient of the loss w.r.t. window length parameter with the chain rule
        dstft_dp = self.stft(x, "backward")
        dl_dp = torch.conj(dl_ds) * dstft_dp
        dl_dp = dl_dp.sum().real.expand(self.win_length.shape)
        return dl_dp

    def stft(self: DSTFT, x: torch.tensor, direction: str):
        # batch_size, length, device, dtype = x.shape[0], x.shape[-1], x.device, x.dtype

        # Generate strided signal and shift idx_frac
        folded_x, idx_frac = self.unfold(x)  # B, T, N; T

        # Generate the tapering window function for the STFT
        self.tap_win = self.window_function(
            direction=direction, idx_frac=idx_frac,
        ).permute(1, 0)  # T, N

        # Compute tapered x
        self.folded_x = folded_x[:, :, :]  # B, T, N
        self.tap_win = self.tap_win[None, :, :]  # 1, T, 1
        self.tapered_x = self.folded_x * self.tap_win  # B, T, N,

        spectr = torch.fft.rfft(self.tapered_x)

        shift = torch.arange(
            end=self.F,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        shift = idx_frac[:, None] * shift[None, :]  # T, N
        shift = torch.exp(2j * pi * shift / self.N)[None, ...]  # 1, T, N

        stft = spectr * shift
        return stft.permute(0, 2, 1)

    def inverse_dstft(self: DSTFT, stft: torch.Tensor) -> torch.tensor:
        """Compute inverse differentiable short-time Fourier transform (IDSTFT).

        Args:
        ----
            self (DSTFT): _description_
            stft (torch.Tensor): _description_

        Returns:
        -------
            torch.tensor: _description_

        """
        # shift
        # shift = torch.arange(
        #     end=self.F,
        #     device=self.device,
        #     dtype=self.dtype,
        #     requires_grad=False,
        # )
        # shift = idx_frac[:, None] * shift[None, :]  # T, N
        # stft = stft * torch.exp(2j * pi * shift / self.N)[None, ...]  # 1, T, N
        # print(stft.shape)

        # inverse
        # print(stft.shape, stft.dtype)
        ifft = torch.fft.irfft(stft, n=self.N, dim=-2)
        # print(ifft.shape, self.tap_win.sum(-1, keepdim=True).shape)
        
        # add shift
        self.itap_win = self.synt_win(None, None)
        ifft = ifft.permute(0, -1, -2) * self.itap_win
        
        # fold
        x_hat = self.fold(ifft)

        return x_hat

    def unfold(self: DSTFT, x: torch.tensor) -> torch.tensor:
        # frames index and strided x
        idx_floor = self.frames.floor()
        # print(self.frames.shape, self.frames)
        idx_frac = self.frames - idx_floor
        idx_floor = idx_floor.long()[:, None].expand((
            self.T,
            self.N,
        )) + torch.arange(0, self.N, device=self.device)
        idx_floor[idx_floor >= self.L] = -1
        # print(self.B, idx_floor.shape, x.shape)
        folded_x = x[:, idx_floor]
        folded_x[:, idx_floor < 0] = 0
        return folded_x, idx_frac

    def fold(self: DSTFT, folded_x: torch.tensor) -> torch.tensor:
        x_hat = torch.zeros(
            self.B, self.L, device=self.device, dtype=self.dtype,
        )
        # print(x_hat.shape, self.B, self.L)
        #print(folded_x.shape)
        for t in range(self.T):
            start_idx = max(0, int(self.frames[t]))
            end_idx = min(self.L - 1, int(self.frames[t]) + self.N)
            start_dec = start_idx - int(self.frames[t])
            end_dec = end_idx - int(self.frames[t])
            x_hat[:, start_idx:end_idx] += folded_x[:, t, start_dec:end_dec]
        return x_hat

    def window_function(self: DSTFT, direction: str, idx_frac) -> torch.tensor:
        if self.tapering_function not in {"hann", "hanning"}:
            raise ValueError(
                f"tapering_function must be one of '{('hann', 'hanning')}', but got padding_mode='{self.tapering_function}'",
            )
        else:
            # Create an array of indices to use as the base for the window function
            base = torch.arange(
                0, self.N, 1, dtype=self.dtype, device=self.device,
            )[:, None].expand([-1, self.T])
            base = base - idx_frac
            # Expand the win_length parameter to match the shape of the base array

        # calculate the tapering function and its derivate w.r.t. window length
        mask1 = base.ge(torch.ceil((self.N - 1 + self.actual_win_length) / 2))
        mask2 = base.le(torch.floor((self.N - 1 - self.actual_win_length) / 2))
        if (
            self.tapering_function == "hann"
            or self.tapering_function == "hanning"
        ):
            if direction == "forward":
                self.tap_win = 0.5 - 0.5 * torch.cos(
                    2
                    * pi
                    * (base + (self.actual_win_length - self.N + 1) / 2)
                    / self.actual_win_length,
                )
                self.tap_win[mask1] = 0
                self.tap_win[mask2] = 0
                #print(self.tap_win.max())
                self.tap_win = self.N* self.tap_win / self.tap_win.sum(dim=0, keepdim=True) # self.N* 
                #print(self.tap_win.max())
                #print(self.tap_win.shape)
                #print(self.N)
                #s = torch.sqrt(self.N /torch.sum(self.tap_win ** 2, dim=0))
                #print(s.shape)
                #self.tap_win = s* self.tap_win
                return self.tap_win.pow(self.win_pow)

            elif direction == "backward":
                f = torch.sin(
                    2
                    * pi
                    * (base + (self.actual_win_length - self.N + 1) / 2)
                    / self.actual_win_length,
                )
                d_tap_win = (
                    -pi
                    / self.actual_win_length.pow(2)
                    * ((self.N - 1) / 2 - base)
                    * f
                )
                d_tap_win[mask1] = 0
                d_tap_win[mask2] = 0
                d_tap_win = d_tap_win / self.N * 2
                return d_tap_win
        return None

    def synt_win(self: DSTFT, direction: str, idx_frac) -> torch.tensor:
        
        wins = torch.zeros(self.L)        
        for t in range(self.T):
            start_idx = max(0, int(self.frames[t]))
            end_idx = min(self.L, int(self.frames[t]) + self.N)
            start_dec = start_idx - int(self.frames[t])
            end_dec = end_idx - int(self.frames[t])
            wins[start_idx:end_idx] += (
                self.tap_win[:, t, start_dec:end_dec].squeeze().detach().cpu()
            )
        self.wins = wins
        self.iwins = torch.zeros(self.L)
        self.iwins[ self.wins > 0 ] = 1 / self.wins[self.wins > 0]
        
        plt.plot(self.wins, label="wins")
        plt.plot(self.iwins, label='iwins')
        plt.plot(self.iwins * self.wins)
        plt.legend()
        
        itap_win = torch.zeros_like(self.tap_win)
        for t in range(self.T):
            start_idx = max(0, int(self.frames[t]))
            end_idx = min(self.L, int(self.frames[t]) + self.N)
            start_dec = start_idx - int(self.frames[t])
            end_dec = end_idx - int(self.frames[t])
            itap_win[:, t, start_dec:end_dec] = (
            #self.tap_win[:, t, start_dec:end_dec]
            #/ wins[start_idx:end_idx] * 
            self.iwins[start_idx:end_idx]
        )
        return itap_win

    def coverage(self: DSTFT):  # in [0, 1]
        # compute coverage
        expanded_win, _ = self.actual_win_length.expand((self.N, self.T)).min(
            dim=0, keepdim=False,
        )
        cov = expanded_win[0]
        maxi = self.frames[0] + self.N / 2 + expanded_win[0] / 2
        for i in range(1, self.T):
            start = torch.min(
                self.L * torch.ones_like(expanded_win[i]),
                torch.max(
                    torch.zeros_like(expanded_win[i]),
                    self.frames[i] + self.N / 2 - expanded_win[i] / 2,
                ),
            )
            end = torch.min(
                self.L * torch.ones_like(expanded_win[i]),
                torch.max(
                    torch.zeros_like(expanded_win[i]),
                    self.frames[i] + self.N / 2 + expanded_win[i] / 2,
                ),
            )
            if end > maxi:
                cov += end - torch.max(start, maxi)
                maxi = end
        cov /= self.L
        return cov

    def plot(
        self: DSTFT,
        spec: torch.Tensor,
        x: torch.Tensor | None = None,
        marklist: Optional[List[Any]] = None,
        figsize=(6.4, 4.8),
        f_hat=None,
        fs=None,
        *,
        weights: bool = True,
        wins: bool = True,
        bar: bool = False,
        cmap: float = "jet",
        ylabel: float = "frequencies",
        xlabel: float = "frames",
    ):
        f_max = spec.shape[-2] if fs is None else fs / 2
        plt.figure(figsize=figsize)
        plt.title("Spectrogram")
        ax = plt.subplot()
        im = ax.imshow(
            spec[0].detach().cpu().log(),
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=[0, spec.shape[-1], 0, f_max],
        )
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if bar == True:
            plt.colorbar(im, ax=ax)
        if f_hat is not None:
            for f in f_hat:
                plt.plot(f, linewidth=0.5, c="k", alpha=0.7)
        plt.show()

        if weights == True:
            plt.figure(figsize=figsize)
            plt.title("Distribution of window lengths")
            ax = plt.subplot()
            im = ax.imshow(
                self.actual_win_length[: self.F].detach().cpu(),
                aspect="auto",
                origin="lower",
                cmap=cmap,
            )
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            if bar == True:
                plt.colorbar(im, ax=ax)
                im.set_clim(self.win_min, self.win_max)
            plt.show()

        if self.tap_win is not None and wins == True:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(self.T + 0.5 + x.squeeze().cpu().numpy(), linewidth=1)
            for i, start in enumerate(self.frames.detach().cpu()):
                ax.plot(
                    range(
                        int(start.floor().item()),
                        int(start.floor().item() + self.N),
                    ),
                    self.T
                    - i
                    - 1.3
                    + self.tap_win[:, i, :].squeeze().detach().cpu(),
                    c="#1f77b4",
                )

            if marklist is not None:
                for elem in marklist:
                    plt.axvline(elem, 0, self.T, c="gray")
            else:
                ax.axvline(x=0, ymin=0, ymax=self.T, c="gray")
                ax.axvline(x=x.shape[-1], ymin=0, ymax=self.T, c="gray")
            plt.show()

class DSTFTenergy(nn.Module):
    """Differentiable short-time Fourier transform (DSTFT) module.

    Args:
    ----
        nn (_type_): _description_

    """

    def __init__(
        self: DSTFT,
        x: torch.tensor,
        win_length: float,
        support: int,
        stride: int,
        pow: float = 1.0,
        win_pow: float = 1.0,
        win_p: str | None = None,
        stride_p: str | None = None,
        pow_p: str | None = None,
        win_requires_grad=True,
        stride_requires_grad: bool = True,
        pow_requires_grad: bool = False,
        # params: str = 'p_tf', # p, p_t, p_f, p_tf
        win_min: float | None = None,
        win_max: float | None = None,
        stride_min: float | None = None,
        stride_max: float | None = None,
        pow_min: float | None = None,
        pow_max: float | None = None,
        tapering_function: str = "hann",
        sr: int = 16_000,
        window_transform=None,
        stride_transform=None,
        dynamic_parameter: bool = False,
        first_frame: bool = False,
    ):
        super().__init__()

        # Constants and hyperparameters
        self.N = support  # support size
        self.F = int(1 + self.N / 2)  # nb of frequencies
        self.B = x.shape[0]  # batch size
        self.L = x.shape[-1]  # signal length
        self.device = x.device
        self.dtype = x.dtype

        self.win_requires_grad = win_requires_grad
        self.stride_requires_grad = stride_requires_grad
        self.pow_requires_grad = pow_requires_grad
        self.tapering_function = tapering_function
        self.dynamic_parameter = dynamic_parameter
        self.first_frame = first_frame
        self.sr = sr
        self.pow = pow
        self.tap_win = None

        # Register eps and min as a buffer tensor
        self.register_buffer(
            "eps",
            torch.tensor(
                torch.finfo(torch.float).eps,
                dtype=self.dtype,
                device=self.device,
            ),
        )
        self.register_buffer(
            "min",
            torch.tensor(
                torch.finfo(torch.float).min,
                dtype=self.dtype,
                device=self.device,
            ),
        )

        # Calculate the number of frames
        # self.T = int(
        #    1
        #    + torch.div(
        #        x.shape[-1] - (self.N - 1) - 1, stride, rounding_mode="floor",
        #    ),
        # )
        self.T = 1 + int(torch.div(x.shape[-1], stride, rounding_mode="floor"))

        if win_min is None:
            self.win_min = self.N / 20
        else:
            self.win_min = win_min
        if win_max is None:
            self.win_max = self.N
        else:
            self.win_max = win_max
        if stride_min is None:
            self.stride_min = 0
        else:
            self.stride_min = stride_min
        if stride_max is None:
            self.stride_max = max(self.N, abs(stride))
        else:
            self.stride_max = stride_max
        if pow_min is None:
            self.pow_min = 0.001
        else:
            self.pow_min = pow_min
        if pow_max is None:
            self.pow_max = 1000
        else:
            self.pow_max = pow_max

        # HOP LENGTH / FRAME INDEX
        # hop length constraints
        if stride_transform is None:
            self.stride_transform = self.__stride_transform
        else:
            self.stride_transform = stride_transform

        # Determine the shape of the stride/hop-length/ frame index parameters
        if stride_p is None:
            stride_size = (1,)
        elif stride_p == "t":
            stride_size = (self.T,)
        else:
            raise ValueError(f"stride_p error {stride_p}")
        # Create the window length parameter and assign it the appropriate shape
        self.init_stride = abs(stride)
        self.strides = nn.Parameter(
            torch.full(
                stride_size,
                self.init_stride,
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.stride_requires_grad,
        )

        # WIN LENGTH
        # win length constraints
        if window_transform is None:
            self.window_transform = self.__window_transform
        else:
            self.window_transform = window_transform

        # Determine the shape of the window length parameters
        if win_p is None:
            win_length_size = (1, 1)
        elif win_p == "t":
            win_length_size = (1, self.T)
        else:
            raise ValueError(
                f"win_p error {win_p}, maybe use ADSTFT instead (frequency varying windows)",
            )
        # Create the window length parameter and assign it the appropriate shape
        self.win_length = nn.Parameter(
            torch.full(
                win_length_size,
                abs(win_length),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.win_requires_grad,
        )

        # WIN POW
        if pow_p is None:
            win_pow_size = (1, 1)
        elif pow_p == "t":
            win_pow_size = (1, self.T)
        else:
            print("pow_p error", pow_p)
        self.win_pow = nn.Parameter(
            torch.full(
                win_pow_size,
                abs(win_pow),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.pow_requires_grad,
        )

    def __window_transform(self: DSTFT, w_in):
        w_out = torch.minimum(
            torch.maximum(
                w_in,
                torch.full_like(
                    w_in, self.win_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                w_in, self.win_max, dtype=self.dtype, device=self.device,
            ),
        )
        return w_out

    def __stride_transform(
        self: DSTFT, s_in: torch.Tensor,
    ):  # born stride entre 0 et 2N
        s_out = torch.minimum(
            torch.maximum(
                s_in,
                torch.full_like(
                    s_in, self.stride_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                s_in, self.stride_max, dtype=self.dtype, device=self.device,
            ),
        )
        return s_out

    def __pow_transform(
        self: DSTFT, p_in: torch.Tensor,
    ):  # born stride entre 0 et 2N
        p_out = torch.minimum(
            torch.maximum(
                p_in,
                torch.full_like(
                    p_in, self.pow_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                p_in, self.pow_max, dtype=self.dtype, device=self.device,
            ),
        )
        return p_out

    @property
    def actual_win_length(self: DSTFT):  # contraints
        return self.window_transform(self.win_length)

    @property
    def actual_strides(
        self,
    ):  # stride contraints, actual stride between frames
        return self.stride_transform(self.strides)

    @property
    def actual_pow(self: DSTFT):  # pow contraints
        return self.pow_transform(self.win_pow)

    @property
    def frames(self: DSTFT):
        # Compute the temporal position (indices) of frames (support)
        expanded_stride = self.actual_strides.expand((self.T,))
        frames = torch.zeros_like(expanded_stride)
        # frames[0] = - self.N / 2
        # if self.first_frame:
        #    frames[0] = (
        #        self.actual_win_length.expand((self.N, self.T))[:, 0].max(
        #            dim=0, keepdim=False,
        #        )[0]
        #        - self.N
        #    ) / 2
        frames -= self.N / 2 + self.init_stride
        frames += expanded_stride.cumsum(dim=0)

        return frames

    @property
    def effective_strides(self: DSTFT):
        # Compute the strides between window (and not frames)
        expanded_stride = self.actual_strides.expand((self.T,))
        effective_strides = torch.zeros_like(expanded_stride)
        effective_strides[1:] = expanded_stride[1:]
        cat = (
            torch.cat(
                (
                    torch.tensor(
                        [self.N], dtype=self.dtype, device=self.device,
                    ),
                    self.actual_win_length.expand((self.N, self.T)).max(
                        dim=0, keepdim=False,
                    )[0],
                ),
                dim=0,
            ).diff()
            / 2
        )
        effective_strides = effective_strides - cat
        return effective_strides

    def forward(self: DSTFT, x: torch.tensor) -> tuple:
        # Perform the forward STFT and extract the magnitude, phase, real, and imaginary parts
        stft = self.stft(x, "forward")
        #spec = stft.abs().pow(self.pow)[:, : self.F] + torch.finfo(x.dtype).eps

        magnitude = stft.abs()  # Get the magnitude of the STFT
        power_spectrum = magnitude.pow(2)  # Square the magnitude to get the power spectrum
        n = self.N  # Number of frequency bins

        # Handle both cases (even or odd number of frequency bins)
        if n % 2 == 0:
            power_spectrum_last = power_spectrum[..., -1].unsqueeze(-1)
            spec = torch.cat([2 * power_spectrum[..., :-1], power_spectrum_last], dim=-1)
        else:
            power_spectrum_first = power_spectrum[..., 0].unsqueeze(-1)
            spec = torch.cat([2 * power_spectrum[..., 1:], power_spectrum_first], dim=-1)
        spec +=  torch.finfo(x.dtype).eps
        #print(spec.max(), spec.min())
        return spec, stft  # , real, imag, phase

    def abs_forward(self: DSTFT, x: torch.tensor) -> tuple:
        # Perform the forward STFT and extract the magnitude, phase, real, and imaginary parts
        stft = self.stft(x, "forward")
        spec = stft.abs().pow(self.pow)[:, : self.F] + torch.finfo(x.dtype).eps
        return spec, stft
    
    def backward(  # why is this function here? autograd is used
        self: DSTFT, x: torch.tensor, dl_ds: torch.tensor,
    ) -> torch.tensor:
        # Compute the gradient of the loss w.r.t. window length parameter with the chain rule
        dstft_dp = self.stft(x, "backward")
        dl_dp = torch.conj(dl_ds) * dstft_dp
        dl_dp = dl_dp.sum().real.expand(self.win_length.shape)
        return dl_dp
    
    def stft(self: DSTFT, x: torch.tensor, direction: str):
        # batch_size, length, device, dtype = x.shape[0], x.shape[-1], x.device, x.dtype

        # Generate strided signal and shift idx_frac
        folded_x, idx_frac = self.unfold(x)  # B, T, N; T

        # Generate the tapering window function for the STFT
        self.tap_win = self.window_function(
            direction=direction, idx_frac=idx_frac,
        ).permute(1, 0)  # T, N

        # Compute tapered x
        self.folded_x = folded_x[:, :, :]  # B, T, N
        self.tap_win = self.tap_win[None, :, :]  # 1, T, 1
        self.tapered_x = self.folded_x * self.tap_win  # B, T, N,

        spectr = torch.fft.rfft(self.tapered_x)

        shift = torch.arange(
            end=self.F,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        shift = idx_frac[:, None] * shift[None, :]  # T, N
        shift = torch.exp(2j * pi * shift / self.N)[None, ...]  # 1, T, N

        stft = spectr * shift

        return stft.permute(0, 2, 1)

    def inverse_dstft(self: DSTFT, stft: torch.Tensor) -> torch.tensor:
        """Compute inverse differentiable short-time Fourier transform (IDSTFT).

        Args:
        ----
            self (DSTFT): _description_
            stft (torch.Tensor): _description_

        Returns:
        -------
            torch.tensor: _description_

        """
        # shift
        # shift = torch.arange(
        #     end=self.F,
        #     device=self.device,
        #     dtype=self.dtype,
        #     requires_grad=False,
        # )
        # shift = idx_frac[:, None] * shift[None, :]  # T, N
        # stft = stft * torch.exp(2j * pi * shift / self.N)[None, ...]  # 1, T, N
        # print(stft.shape)

        # inverse
        # print(stft.shape, stft.dtype)
        ifft = torch.fft.irfft(stft, n=self.N, dim=-2)
        # print(ifft.shape, self.tap_win.sum(-1, keepdim=True).shape)
        
        # add shift
        self.itap_win = self.synt_win(None, None)
        ifft = ifft.permute(0, -1, -2) * self.itap_win
        
        # fold
        x_hat = self.fold(ifft)

        return x_hat

    def unfold(self: DSTFT, x: torch.tensor) -> torch.tensor:
        # frames index and strided x
        idx_floor = self.frames.floor()
        # print(self.frames.shape, self.frames)
        idx_frac = self.frames - idx_floor
        idx_floor = idx_floor.long()[:, None].expand((
            self.T,
            self.N,
        )) + torch.arange(0, self.N, device=self.device)
        idx_floor[idx_floor >= self.L] = -1
        # print(self.B, idx_floor.shape, x.shape)
        folded_x = x[:, idx_floor]
        folded_x[:, idx_floor < 0] = 0
        return folded_x, idx_frac

    def fold(self: DSTFT, folded_x: torch.tensor) -> torch.tensor:
        x_hat = torch.zeros(
            self.B, self.L, device=self.device, dtype=self.dtype,
        )
        # print(x_hat.shape, self.B, self.L)
        #print(folded_x.shape)
        for t in range(self.T):
            start_idx = max(0, int(self.frames[t]))
            end_idx = min(self.L - 1, int(self.frames[t]) + self.N)
            start_dec = start_idx - int(self.frames[t])
            end_dec = end_idx - int(self.frames[t])
            x_hat[:, start_idx:end_idx] += folded_x[:, t, start_dec:end_dec]
        return x_hat

    def window_function(self: DSTFT, direction: str, idx_frac) -> torch.tensor:
        if self.tapering_function not in {"hann", "hanning"}:
            raise ValueError(
                f"tapering_function must be one of '{('hann', 'hanning')}', but got padding_mode='{self.tapering_function}'",
            )
        else:
            # Create an array of indices to use as the base for the window function
            base = torch.arange(
                0, self.N, 1, dtype=self.dtype, device=self.device,
            )[:, None].expand([-1, self.T])
            base = base - idx_frac
            # Expand the win_length parameter to match the shape of the base array

        # calculate the tapering function and its derivate w.r.t. window length
        mask1 = base.ge(torch.ceil((self.N - 1 + self.actual_win_length) / 2))
        mask2 = base.le(torch.floor((self.N - 1 - self.actual_win_length) / 2))
        if (
            self.tapering_function == "hann"
            or self.tapering_function == "hanning"
        ):
            if direction == "forward":
                self.tap_win = 0.5 - 0.5 * torch.cos(
                    2
                    * pi
                    * (base + (self.actual_win_length - self.N + 1) / 2)
                    / self.actual_win_length,
                )
                self.tap_win[mask1] = 0
                self.tap_win[mask2] = 0
                #print(self.tap_win.max())
                #self.tap_win = self.N* self.tap_win / self.tap_win.sum(dim=0, keepdim=True) # self.N* 
                #print(self.tap_win.max())
                #print(self.tap_win.shape)
                #print(self.N)
                s = torch.sqrt(self.N /torch.sum(self.tap_win ** 2, dim=0))
                #print(s.shape)
                self.tap_win = s* self.tap_win
                return self.tap_win.pow(self.win_pow)

            elif direction == "backward":
                f = torch.sin(
                    2
                    * pi
                    * (base + (self.actual_win_length - self.N + 1) / 2)
                    / self.actual_win_length,
                )
                d_tap_win = (
                    -pi
                    / self.actual_win_length.pow(2)
                    * ((self.N - 1) / 2 - base)
                    * f
                )
                d_tap_win[mask1] = 0
                d_tap_win[mask2] = 0
                d_tap_win = d_tap_win / self.N * 2
                return d_tap_win
        return None

    def synt_win(self: DSTFT, direction: str, idx_frac) -> torch.tensor:
        
        wins = torch.zeros(self.L)        
        for t in range(self.T):
            start_idx = max(0, int(self.frames[t]))
            end_idx = min(self.L, int(self.frames[t]) + self.N)
            start_dec = start_idx - int(self.frames[t])
            end_dec = end_idx - int(self.frames[t])
            wins[start_idx:end_idx] += (
                self.tap_win[:, t, start_dec:end_dec].squeeze().detach().cpu()
            )
        self.wins = wins
        self.iwins = torch.zeros(self.L)
        self.iwins[ self.wins > 0 ] = 1 / self.wins[self.wins > 0]
        
        plt.plot(self.wins, label="wins")
        plt.plot(self.iwins, label='iwins')
        plt.plot(self.iwins * self.wins)
        plt.legend()
        
        itap_win = torch.zeros_like(self.tap_win)
        for t in range(self.T):
            start_idx = max(0, int(self.frames[t]))
            end_idx = min(self.L, int(self.frames[t]) + self.N)
            start_dec = start_idx - int(self.frames[t])
            end_dec = end_idx - int(self.frames[t])
            itap_win[:, t, start_dec:end_dec] = (
            #self.tap_win[:, t, start_dec:end_dec]
            #/ wins[start_idx:end_idx] * 
            self.iwins[start_idx:end_idx]
        )
        return itap_win

    def coverage(self: DSTFT):  # in [0, 1]
        # compute coverage
        expanded_win, _ = self.actual_win_length.expand((self.N, self.T)).min(
            dim=0, keepdim=False,
        )
        cov = expanded_win[0]
        maxi = self.frames[0] + self.N / 2 + expanded_win[0] / 2
        for i in range(1, self.T):
            start = torch.min(
                self.L * torch.ones_like(expanded_win[i]),
                torch.max(
                    torch.zeros_like(expanded_win[i]),
                    self.frames[i] + self.N / 2 - expanded_win[i] / 2,
                ),
            )
            end = torch.min(
                self.L * torch.ones_like(expanded_win[i]),
                torch.max(
                    torch.zeros_like(expanded_win[i]),
                    self.frames[i] + self.N / 2 + expanded_win[i] / 2,
                ),
            )
            if end > maxi:
                cov += end - torch.max(start, maxi)
                maxi = end
        cov /= self.L
        return cov

    def plot(
        self: DSTFT,
        spec: torch.Tensor,
        x: torch.Tensor | None = None,
        marklist: Optional[List[Any]] = None,
        figsize=(6.4, 4.8),
        f_hat=None,
        fs=None,
        *,
        weights: bool = True,
        wins: bool = True,
        bar: bool = False,
        cmap: float = "jet",
        ylabel: float = "frequencies",
        xlabel: float = "frames",
    ):
        f_max = spec.shape[-2] if fs is None else fs / 2
        plt.figure(figsize=figsize)
        plt.title("Spectrogram")
        ax = plt.subplot()
        im = ax.imshow(
            spec[0].detach().cpu().log(),
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=[0, spec.shape[-1], 0, f_max],
        )
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if bar == True:
            plt.colorbar(im, ax=ax)
        if f_hat is not None:
            for f in f_hat:
                plt.plot(f, linewidth=0.5, c="k", alpha=0.7)
        plt.show()

        if weights == True:
            plt.figure(figsize=figsize)
            plt.title("Distribution of window lengths")
            ax = plt.subplot()
            im = ax.imshow(
                self.actual_win_length[: self.F].detach().cpu(),
                aspect="auto",
                origin="lower",
                cmap=cmap,
            )
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            if bar == True:
                plt.colorbar(im, ax=ax)
                im.set_clim(self.win_min, self.win_max)
            plt.show()

        if self.tap_win is not None and wins == True:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(self.T + 0.5 + x.squeeze().cpu().numpy(), linewidth=1)
            for i, start in enumerate(self.frames.detach().cpu()):
                ax.plot(
                    range(
                        int(start.floor().item()),
                        int(start.floor().item() + self.N),
                    ),
                    self.T
                    - i
                    - 1.3
                    + self.tap_win[:, i, :].squeeze().detach().cpu(),
                    c="#1f77b4",
                )

            if marklist is not None:
                for elem in marklist:
                    plt.axvline(elem, 0, self.T, c="gray")
            else:
                ax.axvline(x=0, ymin=0, ymax=self.T, c="gray")
                ax.axvline(x=x.shape[-1], ymin=0, ymax=self.T, c="gray")
            plt.show()

class ADSTFTfix(nn.Module):
    """Adaptive differentiable short-time Fourier transform (ADSTFT) module.

    Args:
    ----
        nn (_type_): _description_

    """

    def __init__(
        self: ADSTFT,
        x: torch.tensor,
        win_length: float,
        support: int,
        stride: int,
        pow: float = 1.0,
        win_pow: float = 1.0,
        win_p: str | None = None,
        stride_p: str | None = None,
        pow_p: str | None = None,
        win_requires_grad=True,
        stride_requires_grad: bool = True,
        pow_requires_grad: bool = False,
        # params: str = 'p_tf', # p, p_t, p_f, p_tf
        win_min: float | None = None,
        win_max: float | None = None,
        stride_min: float | None = None,
        stride_max: float | None = None,
        pow_min: float | None = None,
        pow_max: float | None = None,
        tapering_function: str = "hann",
        sr: int = 16_000,
        window_transform=None,
        stride_transform=None,
        dynamic_parameter: bool = False,
        first_frame: bool = False,
    ):
        super().__init__()

        self.kasper = None

        # Constants and hyperparameters
        self.N = support  # support size
        self.F = int(1 + self.N / 2)  # nb of frequencies
        self.B = x.shape[0]  # batch size
        self.L = x.shape[-1]  # signal length
        self.device = x.device
        self.dtype = x.dtype

        self.win_requires_grad = win_requires_grad
        self.stride_requires_grad = stride_requires_grad
        self.pow_requires_grad = pow_requires_grad
        self.tapering_function = tapering_function
        self.dynamic_parameter = dynamic_parameter
        self.first_frame = first_frame
        self.sr = sr
        self.pow = pow
        self.tap_win = None

        # Register eps and min as a buffer tensor
        self.register_buffer(
            "eps",
            torch.tensor(
                torch.finfo(torch.float).eps,
                dtype=self.dtype,
                device=self.device,
            ),
        )
        self.register_buffer(
            "min",
            torch.tensor(
                torch.finfo(torch.float).min,
                dtype=self.dtype,
                device=self.device,
            ),
        )

        # Calculate the number of frames
        # CHANGED
        """self.T = int(
            1
            + torch.div(
                x.shape[-1] - (self.N - 1) - 1, stride, rounding_mode="floor",
            ),
        )"""
        self.T = int(1 + torch.div(x.shape[-1] - (self.N - 1) - 1, stride, rounding_mode='floor'))
        print(self.T)

        if win_min is None:
            self.win_min = self.N / 20
        else:
            self.win_min = win_min
        if win_max is None:
            self.win_max = self.N
        else:
            self.win_max = win_max
        if stride_min is None:
            self.stride_min = 0
        else:
            self.stride_min = stride_min
        if stride_max is None:
            self.stride_max = max(self.N, abs(stride))
        else:
            self.stride_max = stride_max
        if pow_min is None:
            self.pow_min = 0.001
        else:
            self.pow_min = pow_min
        if pow_max is None:
            self.pow_max = 1000
        else:
            self.pow_max = pow_max

        # HOP LENGTH / FRAME INDEX
        if stride_transform is None:
            self.stride_transform = self.__stride_transform
        else:
            self.stride_transform = stride_transform
        # Determine the shape of the stride/hop-length/ frame index parameters
        if stride_p is None:
            stride_size = (1,)
        elif stride_p == "t":
            stride_size = (self.T,)
        else:
            raise ValueError(f"stride_p error {stride_p}")
        # Create the window length parameter and assign it the appropriate shape
        self.strides = nn.Parameter(
            torch.full(
                stride_size, abs(stride), dtype=self.dtype, device=self.device,
            ),
            requires_grad=self.stride_requires_grad,
        )

        # WIN LENGTH
        # win length constraints
        if window_transform is None:
            self.window_transform = self.__window_transform
        else:
            self.window_transform = window_transform
        # Determine the shape of the window length parameters
        if win_p is None:
            win_length_size = (1, 1)
        elif win_p == "t":
            win_length_size = (1, self.T)
        elif win_p == "f":
            win_length_size = (self.F, 1)
        elif win_p == "tf":
            win_length_size = (self.N, self.T)  # CHANGED F->N
        else:
            raise ValueError(f"win_p error {win_p}")
        # Create the window length parameter and assign it the appropriate shape
        self.win_length = nn.Parameter(
            torch.full(
                win_length_size,
                abs(win_length),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.win_requires_grad,
        )

        # WIN POW
        if pow_p is None:
            win_pow_size = (1, 1)
        elif pow_p == "t":
            win_pow_size = (1, self.T)
        elif pow_p == "f":
            win_pow_size = (self.F, 1)
        elif pow_p == "tf":
            win_pow_size = (self.F, self.T)
        else:
            print("pow_p error", pow_p)
        self.win_pow = nn.Parameter(
            torch.full(
                win_pow_size,
                abs(win_pow),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.pow_requires_grad,
        )

    def __window_transform(self: ADSTFT, w_in):
        """_summary_

        Args:
        ----
            w_in (_type_): _description_

        Returns:
        -------
            _type_: _description_

        """
        w_out = torch.minimum(
            torch.maximum(
                w_in,
                torch.full_like(
                    w_in, self.win_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                w_in, self.win_max, dtype=self.dtype, device=self.device,
            ),
        )
        return w_out

    def __stride_transform(self: ADSTFT, s_in):  # born stride entre 0 et 2N
        s_out = torch.minimum(
            torch.maximum(
                s_in,
                torch.full_like(
                    s_in, self.stride_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                s_in, self.stride_max, dtype=self.dtype, device=self.device,
            ),
        )
        return s_out

    def __pow_transform(self: ADSTFT, p_in):  # born stride entre 0 et 2N
        p_out = torch.minimum(
            torch.maximum(
                p_in,
                torch.full_like(
                    p_in, self.pow_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                p_in, self.pow_max, dtype=self.dtype, device=self.device,
            ),
        )
        return p_out

    @property
    def actual_win_length(self: ADSTFT):  # contraints
        return self.window_transform(self.win_length)

    @property
    def actual_strides(
        self,
    ):  # stride contraints, actual stride between frames
        return self.stride_transform(self.strides)

    @property
    def actual_pow(self: ADSTFT):  # pow contraints
        return self.pow_transform(self.win_pow)

    @property
    def frames(self: ADSTFT):
        # Compute the temporal position (indices) of frames (support)
        expanded_stride = self.actual_strides.expand((self.T,))
        frames = torch.zeros_like(expanded_stride)
        if self.first_frame:
            frames[0] = (
                self.actual_win_length.expand((self.N, self.T))[:, 0].max(
                    dim=0, keepdim=False,
                )[0]
                - self.N
            ) / 2
        frames[1:] = frames[0] + expanded_stride[1:].cumsum(dim=0)
        return frames

    @property
    def effective_strides(self: ADSTFT):
        # Compute the strides between window (and not frames)
        expanded_stride = self.actual_strides.expand((self.T,))
        effective_strides = torch.zeros_like(expanded_stride)
        effective_strides[1:] = expanded_stride[1:]
        cat = (
            torch.cat(
                (
                    torch.tensor(
                        [self.N], dtype=self.dtype, device=self.device,
                    ),
                    self.actual_win_length.expand((self.N, self.T)).max(
                        dim=0, keepdim=False,
                    )[0],
                ),
                dim=0,
            ).diff()
            / 2
        )
        effective_strides = effective_strides - cat
        return effective_strides

    def forward(self: ADSTFT, x: torch.tensor) -> tuple:
        # Perform the forward STFT and extract the magnitude, phase, real, and imaginary parts
        stft = self.stft(x, "forward")
        # detta är identiskt med dev
        #print((stft.abs().pow(self.pow)).shape)
        #print((stft.abs().pow(self.pow)[:, : self.F]).shape)
        spec = stft.abs().pow(self.pow)[:, : self.F] + torch.finfo(x.dtype).eps
        return spec, stft

    def backward(
        self: ADSTFT, x: torch.tensor, dl_ds: torch.tensor,
    ) -> torch.tensor:
        # Compute the gradient of the loss w.r.t. window length parameter with the chain rule
        dstft_dp = self.stft(x, "backward")
        dl_dp = torch.conj(dl_ds) * dstft_dp
        dl_dp = dl_dp.sum().real.expand(self.win_length.shape)
        # detta har .expand(self.win_length.shape) istället för .unsqueeze(0), viktigt? verkar inte så
        return dl_dp

    def stft(self: ADSTFT, x: torch.tensor, direction: str):
        # batch_size, length, device, dtype = x.shape[0], x.shape[-1], x.device, x.dtype

        # Generate strided signal and shift idx_frac
        folded_x, idx_frac = self.unfold(x)  # B, T, N; T

        # Generate the tapering window function for the STFT
        self.tap_win = self.window_function(
            direction=direction, idx_frac=idx_frac,
        ).permute(2, 1, 0)  # T, N, N

        # Generate tapering function shift
        shift = torch.arange(
            end=self.N,  # CHANGED F->N
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        shift = idx_frac[:, None] * shift[None, :]  # T, N

        # Compute tapered x
        self.folded_x = folded_x[:, :, None, :]  # B, T, 1, N
        self.tap_win = self.tap_win[None, :, :, :]  # 1, T, F, 1
        shift = torch.exp(2j * pi * shift / self.N)[
            None, :, :, None,
        ]  # 1, T, N, 1
        self.tapered_x = self.folded_x * self.tap_win * shift  # B, T, F, N

        # Generate Fourier coefficients
        coeff = torch.arange(
            end=self.N,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        coeff = coeff[:, None] @ coeff[None, :]  # CHANGED which one is commented
        #coeff = coeff[: self.F, None] @ coeff[None, :]
        coeff = torch.exp(-2j * pi * coeff / self.N)  # N, N
        
        # Perform the STFT
        coeff = coeff[None, None, :, :]  # 1, 1, N, N
        stft = (self.tapered_x * coeff).sum(dim=-1)

        torch.set_printoptions(precision=20)
        print((self.tapered_x[0, 0, 500, :]*coeff[0, 0, 500, :]).sum(dim=-1).abs())
        #print(self.tapered_x[0, 0, 500, 500:510])
        #print(coeff[0, 0, 500, 500:510])
        #print(stft[0, 0, 250].abs())
        print(stft[0, 0, 500].abs())
        torch.set_printoptions(precision=4)

        self.kasper = coeff[0, 0, 500, :]  #list(self.tapered_x[0, 0, 500, :])

        return stft.permute(0, 2, 1)

    def unfold(self: ADSTFT, x) -> torch.tensor:
        # frames index and strided x
        idx_floor = self.frames.floor()
        idx_frac = self.frames - idx_floor
        idx_floor = idx_floor.long()[:, None].expand((
            self.T,
            self.N,
        )) + torch.arange(0, self.N, device=self.device)
        idx_floor[idx_floor >= self.L] = -1
        folded_x = x[:, idx_floor]
        folded_x[:, idx_floor < 0] = 0
        return folded_x, idx_frac

    def window_function(
        self: ADSTFT, direction: str, idx_frac,
    ) -> torch.tensor:
        if self.tapering_function not in {"hann", "hanning"}:
            raise ValueError(
                f"tapering_function must be one of '{('hann', 'hanning')}', but got padding_mode='{self.tapering_function}'",
            )
        else:
            # Create an array of indices to use as the base for the window function
            base = torch.arange(
                0, self.N, 1, dtype=self.dtype, device=self.device,
            )[:, None, None].expand([-1, self.N, self.T])  # CHANGED F->N
            base = base - idx_frac
            # Expand the win_length parameter to match the shape of the base array
            # if self.actual_win_length.dim() == 3:
            #    self.expanded_win_length = self.actual_win_length.expand([self.N, self.N, self.T])
            # elif self.actual_win_length.dim() == 1:
            #    self.expanded_win_length = self.actual_win_length[:, None, None].expand([self.N, self.N, self.T])
            # elif self.actual_win_length.dim() == 2 and self.actual_win_length.shape[-1] == self.T:
            #    self.expanded_win_length = self.actual_win_length[:, None, :].expand([self.N, self.N, self.T])
            # elif self.actual_win_length.dim() == 2 and self.actual_win_length.shape[-1] == self.N:
            #    self.expanded_win_length = self.actual_win_length[:, :, None].expand([self.N, self.N, self.T])

        #mask1 = base.ge(torch.ceil((self.N - 1 + self.actual_win_length) / 2))
        #mask2 = base.le(torch.floor((self.N - 1 - self.actual_win_length) / 2))

        # calculate the tapering function and its derivate w.r.t. window length
        if self.tapering_function == 'hann' or self.tapering_function == 'hanning':
            if direction == 'forward':
                self.tap_win = 0.5 - 0.5 * torch.cos(2 * pi * (base + (self.actual_win_length-self.N+1)/2) / self.actual_win_length )                
                mask1 = base.ge(torch.ceil( (self.N-1+self.actual_win_length)/2))
                mask2 = base.le(torch.floor((self.N-1-self.actual_win_length)/2))            
                self.tap_win[mask1] = 0
                self.tap_win[mask2] = 0
                # new
                #self.tap_win = self.tap_win / self.N * 2
                # old CHANGED
                self.tap_win = self.tap_win / self.tap_win.sum(dim=0, keepdim=True)
                #print(self.tap_win.sum(dim=0, keepdim=True))
                #print(self.sr/self.N * 2 *1/self.actual_win_length) 
                return self.tap_win.pow(self.win_pow)
            
            elif direction == 'backward':
                f = torch.sin(2 * pi * (base - (self.N-1)/2) / self.actual_win_length) 
                # new
                #d_tap_win = -pi / self.actual_win_length.pow(2) * ((self.N - 1) / 2 - base) * f
                #d_tap_win[mask1] = 0
                #d_tap_win[mask2] = 0
                #d_tap_win = d_tap_win / self.N * 2           
                # old CHANGED
                d_tap_win = - pi / self.actual_win_length * ((self.N-1)/2 - base) * f
                d_tap_win[mask1] = 0
                d_tap_win[mask2] = 0
                return d_tap_win
        return None

    def coverage(self: ADSTFT):  # in [0, 1]
        # compute coverage
        expanded_win, _ = self.actual_win_length.expand((self.N, self.T)).min(
            dim=0, keepdim=False,
        )
        cov = expanded_win[0]
        maxi = self.frames[0] + self.N / 2 + expanded_win[0] / 2
        for i in range(1, self.T):
            start = torch.min(
                self.L * torch.ones_like(expanded_win[i]),
                torch.max(
                    torch.zeros_like(expanded_win[i]),
                    self.frames[i] + self.N / 2 - expanded_win[i] / 2,
                ),
            )
            end = torch.min(
                self.L * torch.ones_like(expanded_win[i]),
                torch.max(
                    torch.zeros_like(expanded_win[i]),
                    self.frames[i] + self.N / 2 + expanded_win[i] / 2,
                ),
            )
            if end > maxi:
                cov += end - torch.max(start, maxi)
                maxi = end
        cov /= self.L
        return cov

    def plot(
        self: ADSTFT,
        spec: torch.tensor,
        x: Optional[torch.tensor] = None,
        marklist: Optional[List[int]] = None,
        bar: bool = False,
        figsize: Tuple[float, float] = (6.4, 4.8),
        f_hat=None,
        fs=None,
        *,
        weights: bool = True,
        wins: bool = True,
        cmap: float = "jet",
        ylabel: float = "frequencies",
        xlabel: float = "frames",
    ):
        plt.figure(figsize=figsize)
        plt.title("Spectrogram")
        ax = plt.subplot()
        im = ax.imshow(
            spec[0].detach().cpu().log(),
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=[0, spec.shape[-1], 0, spec.shape[-2]],
        )
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if bar is True:
            plt.colorbar(im, ax=ax)
        plt.show()

        if weights is True:
            plt.figure(figsize=figsize)
            plt.title("Distribution of window lengths")
            ax = plt.subplot()
            im = ax.imshow(
                self.actual_win_length[: self.F].detach().cpu(),
                aspect="auto",
                origin="lower",
                cmap=cmap,
            )
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            if bar is True:
                plt.colorbar(im, ax=ax)
                im.set_clim(self.win_min, self.win_max)
            plt.show()

        if self.tap_win is not None and wins is True:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(self.T + 0.5 + x.squeeze().cpu().numpy(), linewidth=1)
            for i, start in enumerate(self.frames.detach().cpu()):
                ax.plot(
                    range(
                        int(start.floor().item()),
                        int(start.floor().item() + self.N),
                    ),
                    self.T
                    - i
                    - 1.3
                    + 150
                    * self.tap_win[:, i, :, :]
                    .mean(dim=1)
                    .squeeze()
                    .detach()
                    .cpu(),
                    c="#1f77b4",
                )

            if marklist is not None:
                for elem in marklist:
                    plt.axvline(elem, 0, self.T, c="gray")
            else:
                ax.axvline(x=0, ymin=0, ymax=self.T, c="gray")
                ax.axvline(x=x.shape[-1], ymin=0, ymax=self.T, c="gray")
            plt.show()

class ADSTFT(nn.Module):
    """Adaptive differentiable short-time Fourier transform (ADSTFT) module.

    Args:
    ----
        nn (_type_): _description_

    """

    def __init__(
        self: ADSTFT,
        x: torch.tensor,
        win_length: float,
        support: int,
        stride: int,
        pow: float = 1.0,
        win_pow: float = 1.0,
        win_p: str | None = None,
        stride_p: str | None = None,
        pow_p: str | None = None,
        win_requires_grad=True,
        stride_requires_grad: bool = True,
        pow_requires_grad: bool = False,
        # params: str = 'p_tf', # p, p_t, p_f, p_tf
        win_min: float | None = None,
        win_max: float | None = None,
        stride_min: float | None = None,
        stride_max: float | None = None,
        pow_min: float | None = None,
        pow_max: float | None = None,
        tapering_function: str = "hann",
        sr: int = 16_000,
        window_transform=None,
        stride_transform=None,
        dynamic_parameter: bool = False,
        first_frame: bool = False,
    ):
        super().__init__()

        self.kasper = None

        # Constants and hyperparameters
        self.N = support  # support size
        self.F = int(1 + self.N / 2)  # nb of frequencies
        self.B = x.shape[0]  # batch size
        self.L = x.shape[-1]  # signal length
        self.device = x.device
        self.dtype = x.dtype

        self.win_requires_grad = win_requires_grad
        self.stride_requires_grad = stride_requires_grad
        self.pow_requires_grad = pow_requires_grad
        self.tapering_function = tapering_function
        self.dynamic_parameter = dynamic_parameter
        self.first_frame = first_frame
        self.sr = sr
        self.pow = pow
        self.tap_win = None

        # Register eps and min as a buffer tensor
        self.register_buffer(
            "eps",
            torch.tensor(
                torch.finfo(torch.float).eps,
                dtype=self.dtype,
                device=self.device,
            ),
        )
        self.register_buffer(
            "min",
            torch.tensor(
                torch.finfo(torch.float).min,
                dtype=self.dtype,
                device=self.device,
            ),
        )

        # Calculate the number of frames
        self.T = int(
            1
            + torch.div(
                x.shape[-1] - (self.N - 1) - 1, stride, rounding_mode="floor",
            ),
        )

        if win_min is None:
            self.win_min = self.N / 20
        else:
            self.win_min = win_min
        if win_max is None:
            self.win_max = self.N
        else:
            self.win_max = win_max
        if stride_min is None:
            self.stride_min = 0
        else:
            self.stride_min = stride_min
        if stride_max is None:
            self.stride_max = max(self.N, abs(stride))
        else:
            self.stride_max = stride_max
        if pow_min is None:
            self.pow_min = 0.001
        else:
            self.pow_min = pow_min
        if pow_max is None:
            self.pow_max = 1000
        else:
            self.pow_max = pow_max

        # HOP LENGTH / FRAME INDEX
        if stride_transform is None:
            self.stride_transform = self.__stride_transform
        else:
            self.stride_transform = stride_transform
        # Determine the shape of the stride/hop-length/ frame index parameters
        if stride_p is None:
            stride_size = (1,)
        elif stride_p == "t":
            stride_size = (self.T,)
        else:
            raise ValueError(f"stride_p error {stride_p}")
        # Create the window length parameter and assign it the appropriate shape
        self.strides = nn.Parameter(
            torch.full(
                stride_size, abs(stride), dtype=self.dtype, device=self.device,
            ),
            requires_grad=self.stride_requires_grad,
        )

        # WIN LENGTH
        # win length constraints
        if window_transform is None:
            self.window_transform = self.__window_transform
        else:
            self.window_transform = window_transform
        # Determine the shape of the window length parameters
        if win_p is None:
            win_length_size = (1, 1)
        elif win_p == "t":
            win_length_size = (1, self.T)
        elif win_p == "f":
            win_length_size = (self.F, 1)
        elif win_p == "tf":
            win_length_size = (self.F, self.T)
        else:
            raise ValueError(f"win_p error {win_p}")
        # Create the window length parameter and assign it the appropriate shape
        self.win_length = nn.Parameter(
            torch.full(
                win_length_size,
                abs(win_length),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.win_requires_grad,
        )

        # WIN POW
        if pow_p is None:
            win_pow_size = (1, 1)
        elif pow_p == "t":
            win_pow_size = (1, self.T)
        elif pow_p == "f":
            win_pow_size = (self.F, 1)
        elif pow_p == "tf":
            win_pow_size = (self.F, self.T)
        else:
            print("pow_p error", pow_p)
        self.win_pow = nn.Parameter(
            torch.full(
                win_pow_size,
                abs(win_pow),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.pow_requires_grad,
        )

    def __window_transform(self: ADSTFT, w_in):
        """_summary_

        Args:
        ----
            w_in (_type_): _description_

        Returns:
        -------
            _type_: _description_

        """
        w_out = torch.minimum(
            torch.maximum(
                w_in,
                torch.full_like(
                    w_in, self.win_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                w_in, self.win_max, dtype=self.dtype, device=self.device,
            ),
        )
        return w_out

    def __stride_transform(self: ADSTFT, s_in):  # born stride entre 0 et 2N
        s_out = torch.minimum(
            torch.maximum(
                s_in,
                torch.full_like(
                    s_in, self.stride_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                s_in, self.stride_max, dtype=self.dtype, device=self.device,
            ),
        )
        return s_out

    def __pow_transform(self: ADSTFT, p_in):  # born stride entre 0 et 2N
        p_out = torch.minimum(
            torch.maximum(
                p_in,
                torch.full_like(
                    p_in, self.pow_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                p_in, self.pow_max, dtype=self.dtype, device=self.device,
            ),
        )
        return p_out

    @property
    def actual_win_length(self: ADSTFT):  # contraints
        return self.window_transform(self.win_length)

    @property
    def actual_strides(
        self,
    ):  # stride contraints, actual stride between frames
        return self.stride_transform(self.strides)

    @property
    def actual_pow(self: ADSTFT):  # pow contraints
        return self.pow_transform(self.win_pow)

    @property
    def frames(self: ADSTFT):
        # Compute the temporal position (indices) of frames (support)
        expanded_stride = self.actual_strides.expand((self.T,))
        frames = torch.zeros_like(expanded_stride)
        if self.first_frame:
            frames[0] = (
                self.actual_win_length.expand((self.N, self.T))[:, 0].max(
                    dim=0, keepdim=False,
                )[0]
                - self.N
            ) / 2
        frames[1:] = frames[0] + expanded_stride[1:].cumsum(dim=0)
        return frames

    @property
    def effective_strides(self: ADSTFT):
        # Compute the strides between window (and not frames)
        expanded_stride = self.actual_strides.expand((self.T,))
        effective_strides = torch.zeros_like(expanded_stride)
        effective_strides[1:] = expanded_stride[1:]
        cat = (
            torch.cat(
                (
                    torch.tensor(
                        [self.N], dtype=self.dtype, device=self.device,
                    ),
                    self.actual_win_length.expand((self.N, self.T)).max(
                        dim=0, keepdim=False,
                    )[0],
                ),
                dim=0,
            ).diff()
            / 2
        )
        effective_strides = effective_strides - cat
        return effective_strides

    def forward(self: ADSTFT, x: torch.tensor) -> tuple:
        # Perform the forward STFT and extract the magnitude, phase, real, and imaginary parts
        stft = self.stft(x, "forward")
        spec = stft.abs().pow(self.pow)[:, : self.F] + torch.finfo(x.dtype).eps
        return spec, stft

    def backward(
        self: ADSTFT, x: torch.tensor, dl_ds: torch.tensor,
    ) -> torch.tensor:
        # Compute the gradient of the loss w.r.t. window length parameter with the chain rule
        dstft_dp = self.stft(x, "backward")
        dl_dp = torch.conj(dl_ds) * dstft_dp
        dl_dp = dl_dp.sum().real.expand(self.win_length.shape)
        return dl_dp

    def stft(self: ADSTFT, x: torch.tensor, direction: str):
        # batch_size, length, device, dtype = x.shape[0], x.shape[-1], x.device, x.dtype

        # Generate strided signal and shift idx_frac
        folded_x, idx_frac = self.unfold(x)  # B, T, N; T

        # Generate the tapering window function for the STFT
        self.tap_win = self.window_function(
            direction=direction, idx_frac=idx_frac,
        ).permute(2, 1, 0)  # T, N, N

        # Generate tapering function shift
        shift = torch.arange(
            end=self.F,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        shift = idx_frac[:, None] * shift[None, :]  # T, N

        # Compute tapered x
        self.folded_x = folded_x[:, :, None, :]  # B, T, 1, N
        self.tap_win = self.tap_win[None, :, :, :]  # 1, T, F, 1
        shift = torch.exp(2j * pi * shift / self.N)[
            None, :, :, None,
        ]  # 1, T, N, 1
        self.tapered_x = self.folded_x * self.tap_win * shift  # B, T, F, N

        # Generate Fourier coefficients
        coeff = torch.arange(
            end=self.N,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        # coeff = coeff[:, None] @ coeff[None, :]
        coeff = coeff[: self.F, None] @ coeff[None, :]
        coeff = torch.exp(-2j * pi * coeff / self.N)  # N, N

        # Perform the STFT
        coeff = coeff[None, None, :, :]  # 1, 1, N, N
        stft = (self.tapered_x * coeff).sum(dim=-1)

        #torch.set_printoptions(precision=20)
        #print(self.tapered_x[0, 0, 500, 500:510])
        #print(coeff[0, 0, 500, 500:510])
        #print(stft[0, 0, 250].abs())
        #print(stft[0, 0, 500].abs())
        #torch.set_printoptions(precision=4)

        self.kasper = coeff[0, 0, 500, :]  #list(self.tapered_x[0, 0, 500, :])

        return stft.permute(0, 2, 1)

    def unfold(self: ADSTFT, x) -> torch.tensor:
        # frames index and strided x
        idx_floor = self.frames.floor()
        idx_frac = self.frames - idx_floor
        idx_floor = idx_floor.long()[:, None].expand((
            self.T,
            self.N,
        )) + torch.arange(0, self.N, device=self.device)
        idx_floor[idx_floor >= self.L] = -1
        folded_x = x[:, idx_floor]
        folded_x[:, idx_floor < 0] = 0
        return folded_x, idx_frac

    def window_function(
        self: ADSTFT, direction: str, idx_frac,
    ) -> torch.tensor:
        if self.tapering_function not in {"hann", "hanning"}:
            raise ValueError(
                f"tapering_function must be one of '{('hann', 'hanning')}', but got padding_mode='{self.tapering_function}'",
            )
        else:
            # Create an array of indices to use as the base for the window function
            base = torch.arange(
                0, self.N, 1, dtype=self.dtype, device=self.device,
            )[:, None, None].expand([-1, self.F, self.T])
            base = base - idx_frac
            # Expand the win_length parameter to match the shape of the base array
            # if self.actual_win_length.dim() == 3:
            #    self.expanded_win_length = self.actual_win_length.expand([self.N, self.N, self.T])
            # elif self.actual_win_length.dim() == 1:
            #    self.expanded_win_length = self.actual_win_length[:, None, None].expand([self.N, self.N, self.T])
            # elif self.actual_win_length.dim() == 2 and self.actual_win_length.shape[-1] == self.T:
            #    self.expanded_win_length = self.actual_win_length[:, None, :].expand([self.N, self.N, self.T])
            # elif self.actual_win_length.dim() == 2 and self.actual_win_length.shape[-1] == self.N:
            #    self.expanded_win_length = self.actual_win_length[:, :, None].expand([self.N, self.N, self.T])

        mask1 = base.ge(torch.ceil((self.N - 1 + self.actual_win_length) / 2))
        mask2 = base.le(torch.floor((self.N - 1 - self.actual_win_length) / 2))

        # calculate the tapering function and its derivate w.r.t. window length
        if (
            self.tapering_function == "hann"
            or self.tapering_function == "hanning"
        ):
            if direction == 'forward':
                self.tap_win = 0.5 - 0.5 * torch.cos(2 * pi * (base + (self.actual_win_length-self.N+1)/2) / self.actual_win_length )                
                mask1 = base.ge(torch.ceil( (self.N-1+self.actual_win_length)/2))
                mask2 = base.le(torch.floor((self.N-1-self.actual_win_length)/2))            
                self.tap_win[mask1] = 0
                self.tap_win[mask2] = 0
                # new
                #self.tap_win = self.tap_win / self.N * 2
                # old CHANGED
                self.tap_win = self.tap_win / self.tap_win.sum(dim=0, keepdim=True)
                return self.tap_win.pow(self.win_pow)
            
            elif direction == 'backward':
                f = torch.sin(2 * pi * (base - (self.N-1)/2) / self.actual_win_length) 
                # new
                #d_tap_win = -pi / self.actual_win_length.pow(2) * ((self.N - 1) / 2 - base) * f
                #d_tap_win[mask1] = 0
                #d_tap_win[mask2] = 0
                #d_tap_win = d_tap_win / self.N * 2           
                # old CHANGED
                d_tap_win = - pi / self.actual_win_length * ((self.N-1)/2 - base) * f
                d_tap_win[mask1] = 0
                d_tap_win[mask2] = 0
                return d_tap_win
        return None

    def coverage(self: ADSTFT):  # in [0, 1]
        # compute coverage
        expanded_win, _ = self.actual_win_length.expand((self.N, self.T)).min(
            dim=0, keepdim=False,
        )
        cov = expanded_win[0]
        maxi = self.frames[0] + self.N / 2 + expanded_win[0] / 2
        for i in range(1, self.T):
            start = torch.min(
                self.L * torch.ones_like(expanded_win[i]),
                torch.max(
                    torch.zeros_like(expanded_win[i]),
                    self.frames[i] + self.N / 2 - expanded_win[i] / 2,
                ),
            )
            end = torch.min(
                self.L * torch.ones_like(expanded_win[i]),
                torch.max(
                    torch.zeros_like(expanded_win[i]),
                    self.frames[i] + self.N / 2 + expanded_win[i] / 2,
                ),
            )
            if end > maxi:
                cov += end - torch.max(start, maxi)
                maxi = end
        cov /= self.L
        return cov

    def plot(
        self: ADSTFT,
        spec: torch.tensor,
        x: Optional[torch.tensor] = None,
        marklist: Optional[List[int]] = None,
        bar: bool = False,
        figsize: Tuple[float, float] = (6.4, 4.8),
        f_hat=None,
        fs=None,
        *,
        weights: bool = True,
        wins: bool = True,
        cmap: float = "jet",
        ylabel: float = "frequencies",
        xlabel: float = "frames",
    ):
        plt.figure(figsize=figsize)
        plt.title("Spectrogram")
        ax = plt.subplot()
        im = ax.imshow(
            spec[0].detach().cpu().log(),
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=[0, spec.shape[-1], 0, spec.shape[-2]],
        )
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if bar is True:
            plt.colorbar(im, ax=ax)
        plt.show()

        if weights is True:
            plt.figure(figsize=figsize)
            plt.title("Distribution of window lengths")
            ax = plt.subplot()
            im = ax.imshow(
                self.actual_win_length[: self.F].detach().cpu(),
                aspect="auto",
                origin="lower",
                cmap=cmap,
            )
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            if bar is True:
                plt.colorbar(im, ax=ax)
                im.set_clim(self.win_min, self.win_max)
            plt.show()

        if self.tap_win is not None and wins is True:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(self.T + 0.5 + x.squeeze().cpu().numpy(), linewidth=1)
            for i, start in enumerate(self.frames.detach().cpu()):
                ax.plot(
                    range(
                        int(start.floor().item()),
                        int(start.floor().item() + self.N),
                    ),
                    self.T
                    - i
                    - 1.3
                    + 150
                    * self.tap_win[:, i, :, :]
                    .mean(dim=1)
                    .squeeze()
                    .detach()
                    .cpu(),
                    c="#1f77b4",
                )

            if marklist is not None:
                for elem in marklist:
                    plt.axvline(elem, 0, self.T, c="gray")
            else:
                ax.axvline(x=0, ymin=0, ymax=self.T, c="gray")
                ax.axvline(x=x.shape[-1], ymin=0, ymax=self.T, c="gray")
            plt.show()

class ADSTFTenergy(nn.Module):
    """Adaptive differentiable short-time Fourier transform (ADSTFT) module.

    Args:
    ----
        nn (_type_): _description_

    """

    def __init__(
        self: ADSTFT,
        x: torch.tensor,
        win_length: float,
        support: int,
        stride: int,
        pow: float = 1.0,
        win_pow: float = 1.0,
        win_p: str | None = None,
        stride_p: str | None = None,
        pow_p: str | None = None,
        win_requires_grad=True,
        stride_requires_grad: bool = True,
        pow_requires_grad: bool = False,
        # params: str = 'p_tf', # p, p_t, p_f, p_tf
        win_min: float | None = None,
        win_max: float | None = None,
        stride_min: float | None = None,
        stride_max: float | None = None,
        pow_min: float | None = None,
        pow_max: float | None = None,
        tapering_function: str = "hann",
        sr: int = 16_000,
        window_transform=None,
        stride_transform=None,
        dynamic_parameter: bool = False,
        first_frame: bool = False,
    ):
        super().__init__()

        self.kasper = None

        # Constants and hyperparameters
        self.N = support  # support size
        self.F = int(1 + self.N / 2)  # nb of frequencies
        self.B = x.shape[0]  # batch size
        self.L = x.shape[-1]  # signal length
        self.device = x.device
        self.dtype = x.dtype

        self.win_requires_grad = win_requires_grad
        self.stride_requires_grad = stride_requires_grad
        self.pow_requires_grad = pow_requires_grad
        self.tapering_function = tapering_function
        self.dynamic_parameter = dynamic_parameter
        self.first_frame = first_frame
        self.sr = sr
        self.pow = pow
        self.tap_win = None

        # Register eps and min as a buffer tensor
        self.register_buffer(
            "eps",
            torch.tensor(
                torch.finfo(torch.float).eps,
                dtype=self.dtype,
                device=self.device,
            ),
        )
        self.register_buffer(
            "min",
            torch.tensor(
                torch.finfo(torch.float).min,
                dtype=self.dtype,
                device=self.device,
            ),
        )

        # Calculate the number of frames
        # CHANGED
        """self.T = int(
            1
            + torch.div(
                x.shape[-1] - (self.N - 1) - 1, stride, rounding_mode="floor",
            ),
        )"""
        self.T = int(1 + torch.div(x.shape[-1] - (self.N - 1) - 1, stride, rounding_mode='floor'))
        #print(self.T)

        if win_min is None:
            self.win_min = self.N / 20
        else:
            self.win_min = win_min
        if win_max is None:
            self.win_max = self.N
        else:
            self.win_max = win_max
        if stride_min is None:
            self.stride_min = 0
        else:
            self.stride_min = stride_min
        if stride_max is None:
            self.stride_max = max(self.N, abs(stride))
        else:
            self.stride_max = stride_max
        if pow_min is None:
            self.pow_min = 0.001
        else:
            self.pow_min = pow_min
        if pow_max is None:
            self.pow_max = 1000
        else:
            self.pow_max = pow_max

        # HOP LENGTH / FRAME INDEX
        if stride_transform is None:
            self.stride_transform = self.__stride_transform
        else:
            self.stride_transform = stride_transform
        # Determine the shape of the stride/hop-length/ frame index parameters
        if stride_p is None:
            stride_size = (1,)
        elif stride_p == "t":
            stride_size = (self.T,)
        else:
            raise ValueError(f"stride_p error {stride_p}")
        # Create the window length parameter and assign it the appropriate shape
        self.strides = nn.Parameter(
            torch.full(
                stride_size, abs(stride), dtype=self.dtype, device=self.device,
            ),
            requires_grad=self.stride_requires_grad,
        )

        # WIN LENGTH
        # win length constraints
        if window_transform is None:
            self.window_transform = self.__window_transform
        else:
            self.window_transform = window_transform
        # Determine the shape of the window length parameters
        if win_p is None:
            win_length_size = (1, 1)
        elif win_p == "t":
            win_length_size = (1, self.T)
        elif win_p == "f":
            win_length_size = (self.F, 1)
        elif win_p == "tf":
            win_length_size = (self.N, self.T)  # CHANGED F->N
        else:
            raise ValueError(f"win_p error {win_p}")
        # Create the window length parameter and assign it the appropriate shape
        self.win_length = nn.Parameter(
            torch.full(
                win_length_size,
                abs(win_length),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.win_requires_grad,
        )

        # WIN POW
        if pow_p is None:
            win_pow_size = (1, 1)
        elif pow_p == "t":
            win_pow_size = (1, self.T)
        elif pow_p == "f":
            win_pow_size = (self.F, 1)
        elif pow_p == "tf":
            win_pow_size = (self.F, self.T)
        else:
            print("pow_p error", pow_p)
        self.win_pow = nn.Parameter(
            torch.full(
                win_pow_size,
                abs(win_pow),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.pow_requires_grad,
        )

    def __window_transform(self: ADSTFT, w_in):
        """_summary_

        Args:
        ----
            w_in (_type_): _description_

        Returns:
        -------
            _type_: _description_

        """
        w_out = torch.minimum(
            torch.maximum(
                w_in,
                torch.full_like(
                    w_in, self.win_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                w_in, self.win_max, dtype=self.dtype, device=self.device,
            ),
        )
        return w_out

    def __stride_transform(self: ADSTFT, s_in):  # born stride entre 0 et 2N
        s_out = torch.minimum(
            torch.maximum(
                s_in,
                torch.full_like(
                    s_in, self.stride_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                s_in, self.stride_max, dtype=self.dtype, device=self.device,
            ),
        )
        return s_out

    def __pow_transform(self: ADSTFT, p_in):  # born stride entre 0 et 2N
        p_out = torch.minimum(
            torch.maximum(
                p_in,
                torch.full_like(
                    p_in, self.pow_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                p_in, self.pow_max, dtype=self.dtype, device=self.device,
            ),
        )
        return p_out

    @property
    def actual_win_length(self: ADSTFT):  # contraints
        return self.window_transform(self.win_length)

    @property
    def actual_strides(
        self,
    ):  # stride contraints, actual stride between frames
        return self.stride_transform(self.strides)

    @property
    def actual_pow(self: ADSTFT):  # pow contraints
        return self.pow_transform(self.win_pow)

    @property
    def frames(self: ADSTFT):
        # Compute the temporal position (indices) of frames (support)
        expanded_stride = self.actual_strides.expand((self.T,))
        frames = torch.zeros_like(expanded_stride)
        if self.first_frame:
            frames[0] = (
                self.actual_win_length.expand((self.N, self.T))[:, 0].max(
                    dim=0, keepdim=False,
                )[0]
                - self.N
            ) / 2
        frames[1:] = frames[0] + expanded_stride[1:].cumsum(dim=0)
        return frames

    @property
    def effective_strides(self: ADSTFT):
        # Compute the strides between window (and not frames)
        expanded_stride = self.actual_strides.expand((self.T,))
        effective_strides = torch.zeros_like(expanded_stride)
        effective_strides[1:] = expanded_stride[1:]
        cat = (
            torch.cat(
                (
                    torch.tensor(
                        [self.N], dtype=self.dtype, device=self.device,
                    ),
                    self.actual_win_length.expand((self.N, self.T)).max(
                        dim=0, keepdim=False,
                    )[0],
                ),
                dim=0,
            ).diff()
            / 2
        )
        effective_strides = effective_strides - cat
        return effective_strides

    def forward(self: DSTFT, x: torch.tensor) -> tuple:
        # Perform the forward STFT and extract the magnitude, phase, real, and imaginary parts
        stft = self.stft(x, "forward")
        #spec = stft.abs().pow(self.pow)[:, : self.F] + torch.finfo(x.dtype).eps

        magnitude = stft.abs()  # Get the magnitude of the STFT
        power_spectrum = magnitude.pow(2)  # Square the magnitude to get the power spectrum
        n = self.N  # Number of frequency bins

        # Handle both cases (even or odd number of frequency bins)
        if n % 2 == 0:
            power_spectrum_last = power_spectrum[..., -1].unsqueeze(-1)
            spec = torch.cat([2 * power_spectrum[..., :-1], power_spectrum_last], dim=-1)
        else:
            power_spectrum_first = power_spectrum[..., 0].unsqueeze(-1)
            spec = torch.cat([2 * power_spectrum[..., 1:], power_spectrum_first], dim=-1)
        spec +=  torch.finfo(x.dtype).eps
        #print(spec.max(), spec.min())
        return spec, stft  # , real, imag, phase

    def abs_forward(self: DSTFT, x: torch.tensor) -> tuple:
        # Perform the forward STFT and extract the magnitude, phase, real, and imaginary parts
        stft = self.stft(x, "forward")
        spec = stft.abs().pow(self.pow)[:, : self.F] + torch.finfo(x.dtype).eps
        return spec, stft
    
    def backward(
        self: ADSTFT, x: torch.tensor, dl_ds: torch.tensor,
    ) -> torch.tensor:
        # Compute the gradient of the loss w.r.t. window length parameter with the chain rule
        dstft_dp = self.stft(x, "backward")
        dl_dp = torch.conj(dl_ds) * dstft_dp
        dl_dp = dl_dp.sum().real.expand(self.win_length.shape)
        # detta har .expand(self.win_length.shape) istället för .unsqueeze(0), viktigt? verkar inte så
        return dl_dp

    def stft(self: ADSTFT, x: torch.tensor, direction: str):
        # batch_size, length, device, dtype = x.shape[0], x.shape[-1], x.device, x.dtype

        # Generate strided signal and shift idx_frac
        folded_x, idx_frac = self.unfold(x)  # B, T, N; T

        # Generate the tapering window function for the STFT
        self.tap_win = self.window_function(
            direction=direction, idx_frac=idx_frac,
        ).permute(2, 1, 0)  # T, N, N

        # Generate tapering function shift
        shift = torch.arange(
            end=self.N,  # CHANGED F->N
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        shift = idx_frac[:, None] * shift[None, :]  # T, N

        # Compute tapered x
        self.folded_x = folded_x[:, :, None, :]  # B, T, 1, N
        self.tap_win = self.tap_win[None, :, :, :]  # 1, T, F, 1
        shift = torch.exp(2j * pi * shift / self.N)[
            None, :, :, None,
        ]  # 1, T, N, 1
        self.tapered_x = self.folded_x * self.tap_win * shift  # B, T, F, N

        # Generate Fourier coefficients
        coeff = torch.arange(
            end=self.N,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        coeff = coeff[:, None] @ coeff[None, :]  # CHANGED which one is commented
        #coeff = coeff[: self.F, None] @ coeff[None, :]
        coeff = torch.exp(-2j * pi * coeff / self.N)  # N, N
        
        # Perform the STFT
        coeff = coeff[None, None, :, :]  # 1, 1, N, N
        stft = (self.tapered_x * coeff).sum(dim=-1)

        #self.kasper = coeff[0, 0, 500, :]  #list(self.tapered_x[0, 0, 500, :])

        return stft.permute(0, 2, 1)

    def unfold(self: ADSTFT, x) -> torch.tensor:
        # frames index and strided x
        idx_floor = self.frames.floor()
        idx_frac = self.frames - idx_floor
        idx_floor = idx_floor.long()[:, None].expand((
            self.T,
            self.N,
        )) + torch.arange(0, self.N, device=self.device)
        idx_floor[idx_floor >= self.L] = -1
        folded_x = x[:, idx_floor]
        folded_x[:, idx_floor < 0] = 0
        return folded_x, idx_frac

    def window_function(
        self: ADSTFT, direction: str, idx_frac,
    ) -> torch.tensor:
        if self.tapering_function not in {"hann", "hanning"}:
            raise ValueError(
                f"tapering_function must be one of '{('hann', 'hanning')}', but got padding_mode='{self.tapering_function}'",
            )
        else:
            # Create an array of indices to use as the base for the window function
            base = torch.arange(
                0, self.N, 1, dtype=self.dtype, device=self.device,
            )[:, None, None].expand([-1, self.N, self.T])  # CHANGED F->N
            base = base - idx_frac
            # Expand the win_length parameter to match the shape of the base array
            # if self.actual_win_length.dim() == 3:
            #    self.expanded_win_length = self.actual_win_length.expand([self.N, self.N, self.T])
            # elif self.actual_win_length.dim() == 1:
            #    self.expanded_win_length = self.actual_win_length[:, None, None].expand([self.N, self.N, self.T])
            # elif self.actual_win_length.dim() == 2 and self.actual_win_length.shape[-1] == self.T:
            #    self.expanded_win_length = self.actual_win_length[:, None, :].expand([self.N, self.N, self.T])
            # elif self.actual_win_length.dim() == 2 and self.actual_win_length.shape[-1] == self.N:
            #    self.expanded_win_length = self.actual_win_length[:, :, None].expand([self.N, self.N, self.T])

        #mask1 = base.ge(torch.ceil((self.N - 1 + self.actual_win_length) / 2))
        #mask2 = base.le(torch.floor((self.N - 1 - self.actual_win_length) / 2))

        # calculate the tapering function and its derivate w.r.t. window length
        if self.tapering_function == 'hann' or self.tapering_function == 'hanning':
            if direction == 'forward':
                self.tap_win = 0.5 - 0.5 * torch.cos(2 * pi * (base + (self.actual_win_length-self.N+1)/2) / self.actual_win_length )                
                mask1 = base.ge(torch.ceil( (self.N-1+self.actual_win_length)/2))
                mask2 = base.le(torch.floor((self.N-1-self.actual_win_length)/2))            
                self.tap_win[mask1] = 0
                self.tap_win[mask2] = 0
                # new
                #self.tap_win = self.tap_win / self.N * 2
                # old CHANGED
                #self.tap_win = self.tap_win / self.tap_win.sum(dim=0, keepdim=True)
                #print(self.tap_win.sum(dim=0, keepdim=True))
                #print(self.sr/self.N * 2 *1/self.actual_win_length)
                s = torch.sqrt(self.N /torch.sum(self.tap_win ** 2, dim=0))
                #print(s.shape)
                self.tap_win = s* self.tap_win 
                return self.tap_win.pow(self.win_pow)
            
            elif direction == 'backward':
                f = torch.sin(2 * pi * (base - (self.N-1)/2) / self.actual_win_length) 
                # new
                #d_tap_win = -pi / self.actual_win_length.pow(2) * ((self.N - 1) / 2 - base) * f
                #d_tap_win[mask1] = 0
                #d_tap_win[mask2] = 0
                #d_tap_win = d_tap_win / self.N * 2           
                # old CHANGED
                d_tap_win = - pi / self.actual_win_length * ((self.N-1)/2 - base) * f
                d_tap_win[mask1] = 0
                d_tap_win[mask2] = 0
                return d_tap_win
        return None

    def coverage(self: ADSTFT):  # in [0, 1]
        # compute coverage
        expanded_win, _ = self.actual_win_length.expand((self.N, self.T)).min(
            dim=0, keepdim=False,
        )
        cov = expanded_win[0]
        maxi = self.frames[0] + self.N / 2 + expanded_win[0] / 2
        for i in range(1, self.T):
            start = torch.min(
                self.L * torch.ones_like(expanded_win[i]),
                torch.max(
                    torch.zeros_like(expanded_win[i]),
                    self.frames[i] + self.N / 2 - expanded_win[i] / 2,
                ),
            )
            end = torch.min(
                self.L * torch.ones_like(expanded_win[i]),
                torch.max(
                    torch.zeros_like(expanded_win[i]),
                    self.frames[i] + self.N / 2 + expanded_win[i] / 2,
                ),
            )
            if end > maxi:
                cov += end - torch.max(start, maxi)
                maxi = end
        cov /= self.L
        return cov

    def plot(
        self: ADSTFT,
        spec: torch.tensor,
        x: Optional[torch.tensor] = None,
        marklist: Optional[List[int]] = None,
        bar: bool = False,
        figsize: Tuple[float, float] = (6.4, 4.8),
        f_hat=None,
        fs=None,
        *,
        weights: bool = True,
        wins: bool = True,
        cmap: float = "jet",
        ylabel: float = "frequencies",
        xlabel: float = "frames",
    ):
        plt.figure(figsize=figsize)
        plt.title("Spectrogram")
        ax = plt.subplot()
        im = ax.imshow(
            spec[0].detach().cpu().log(),
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=[0, spec.shape[-1], 0, spec.shape[-2]],
        )
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if bar is True:
            plt.colorbar(im, ax=ax)
        plt.show()

        if weights is True:
            plt.figure(figsize=figsize)
            plt.title("Distribution of window lengths")
            ax = plt.subplot()
            im = ax.imshow(
                self.actual_win_length[: self.F].detach().cpu(),
                aspect="auto",
                origin="lower",
                cmap=cmap,
            )
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            if bar is True:
                plt.colorbar(im, ax=ax)
                im.set_clim(self.win_min, self.win_max)
            plt.show()

        if self.tap_win is not None and wins is True:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(self.T + 0.5 + x.squeeze().cpu().numpy(), linewidth=1)
            for i, start in enumerate(self.frames.detach().cpu()):
                ax.plot(
                    range(
                        int(start.floor().item()),
                        int(start.floor().item() + self.N),
                    ),
                    self.T
                    - i
                    - 1.3
                    + 150
                    * self.tap_win[:, i, :, :]
                    .mean(dim=1)
                    .squeeze()
                    .detach()
                    .cpu(),
                    c="#1f77b4",
                )

            if marklist is not None:
                for elem in marklist:
                    plt.axvline(elem, 0, self.T, c="gray")
            else:
                ax.axvline(x=0, ymin=0, ymax=self.T, c="gray")
                ax.axvline(x=x.shape[-1], ymin=0, ymax=self.T, c="gray")
            plt.show()

class ADSTFTenergy2(nn.Module):
    """Adaptive differentiable short-time Fourier transform (ADSTFT) module.

    Args:
    ----
        nn (_type_): _description_

    """

    def __init__(
        self: ADSTFT,
        x: torch.tensor,
        win_length: float,
        support: int,
        stride: int,
        pow: float = 1.0,
        win_pow: float = 1.0,
        win_p: str | None = None,
        stride_p: str | None = None,
        pow_p: str | None = None,
        win_requires_grad=True,
        stride_requires_grad: bool = True,
        pow_requires_grad: bool = False,
        # params: str = 'p_tf', # p, p_t, p_f, p_tf
        win_min: float | None = None,
        win_max: float | None = None,
        stride_min: float | None = None,
        stride_max: float | None = None,
        pow_min: float | None = None,
        pow_max: float | None = None,
        tapering_function: str = "hann",
        sr: int = 16_000,
        window_transform=None,
        stride_transform=None,
        dynamic_parameter: bool = False,
        first_frame: bool = False,
    ):
        super().__init__()

        self.kasper = None

        # Constants and hyperparameters
        self.N = support  # support size
        self.F = int(1 + self.N / 2)  # nb of frequencies
        self.B = x.shape[0]  # batch size
        self.L = x.shape[-1]  # signal length
        self.device = x.device
        self.dtype = x.dtype

        self.win_requires_grad = win_requires_grad
        self.stride_requires_grad = stride_requires_grad
        self.pow_requires_grad = pow_requires_grad
        self.tapering_function = tapering_function
        self.dynamic_parameter = dynamic_parameter
        self.first_frame = first_frame
        self.sr = sr
        self.pow = pow
        self.tap_win = None

        # Register eps and min as a buffer tensor
        self.register_buffer(
            "eps",
            torch.tensor(
                torch.finfo(torch.float).eps,
                dtype=self.dtype,
                device=self.device,
            ),
        )
        self.register_buffer(
            "min",
            torch.tensor(
                torch.finfo(torch.float).min,
                dtype=self.dtype,
                device=self.device,
            ),
        )

        # Calculate the number of frames
        # CHANGED
        """self.T = int(
            1
            + torch.div(
                x.shape[-1] - (self.N - 1) - 1, stride, rounding_mode="floor",
            ),
        )"""
        self.T = int(1 + torch.div(x.shape[-1] - (self.N - 1) - 1, stride, rounding_mode='floor'))
        #print(self.T)

        if win_min is None:
            self.win_min = self.N / 20
        else:
            self.win_min = win_min
        if win_max is None:
            self.win_max = self.N
        else:
            self.win_max = win_max
        if stride_min is None:
            self.stride_min = 0
        else:
            self.stride_min = stride_min
        if stride_max is None:
            self.stride_max = max(self.N, abs(stride))
        else:
            self.stride_max = stride_max
        if pow_min is None:
            self.pow_min = 0.001
        else:
            self.pow_min = pow_min
        if pow_max is None:
            self.pow_max = 1000
        else:
            self.pow_max = pow_max

        # HOP LENGTH / FRAME INDEX
        if stride_transform is None:
            self.stride_transform = self.__stride_transform
        else:
            self.stride_transform = stride_transform
        # Determine the shape of the stride/hop-length/ frame index parameters
        if stride_p is None:
            stride_size = (1,)
        elif stride_p == "t":
            stride_size = (self.T,)
        else:
            raise ValueError(f"stride_p error {stride_p}")
        # Create the window length parameter and assign it the appropriate shape
        self.strides = nn.Parameter(
            torch.full(
                stride_size, abs(stride), dtype=self.dtype, device=self.device,
            ),
            requires_grad=self.stride_requires_grad,
        )

        # WIN LENGTH
        # win length constraints
        if window_transform is None:
            self.window_transform = self.__window_transform
        else:
            self.window_transform = window_transform
        # Determine the shape of the window length parameters
        if win_p is None:
            win_length_size = (1, 1)
        elif win_p == "t":
            win_length_size = (1, self.T)
        elif win_p == "f":
            win_length_size = (self.F, 1)
        elif win_p == "tf":
            win_length_size = (self.F, self.T)
        else:
            raise ValueError(f"win_p error {win_p}")
        # Create the window length parameter and assign it the appropriate shape
        self.win_length = nn.Parameter(
            torch.full(
                win_length_size,
                abs(win_length),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.win_requires_grad,
        )

        # WIN POW
        if pow_p is None:
            win_pow_size = (1, 1)
        elif pow_p == "t":
            win_pow_size = (1, self.T)
        elif pow_p == "f":
            win_pow_size = (self.F, 1)
        elif pow_p == "tf":
            win_pow_size = (self.F, self.T)
        else:
            print("pow_p error", pow_p)
        self.win_pow = nn.Parameter(
            torch.full(
                win_pow_size,
                abs(win_pow),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.pow_requires_grad,
        )

    def __window_transform(self: ADSTFT, w_in):
        """_summary_

        Args:
        ----
            w_in (_type_): _description_

        Returns:
        -------
            _type_: _description_

        """
        w_out = torch.minimum(
            torch.maximum(
                w_in,
                torch.full_like(
                    w_in, self.win_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                w_in, self.win_max, dtype=self.dtype, device=self.device,
            ),
        )
        return w_out

    def __stride_transform(self: ADSTFT, s_in):  # born stride entre 0 et 2N
        s_out = torch.minimum(
            torch.maximum(
                s_in,
                torch.full_like(
                    s_in, self.stride_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                s_in, self.stride_max, dtype=self.dtype, device=self.device,
            ),
        )
        return s_out

    def __pow_transform(self: ADSTFT, p_in):  # born stride entre 0 et 2N
        p_out = torch.minimum(
            torch.maximum(
                p_in,
                torch.full_like(
                    p_in, self.pow_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                p_in, self.pow_max, dtype=self.dtype, device=self.device,
            ),
        )
        return p_out

    @property
    def actual_win_length(self: ADSTFT):  # contraints
        return self.window_transform(self.win_length)

    @property
    def actual_strides(
        self,
    ):  # stride contraints, actual stride between frames
        return self.stride_transform(self.strides)

    @property
    def actual_pow(self: ADSTFT):  # pow contraints
        return self.pow_transform(self.win_pow)

    @property
    def frames(self: ADSTFT):
        # Compute the temporal position (indices) of frames (support)
        expanded_stride = self.actual_strides.expand((self.T,))
        frames = torch.zeros_like(expanded_stride)
        if self.first_frame:
            frames[0] = (
                self.actual_win_length.expand((self.N, self.T))[:, 0].max(
                    dim=0, keepdim=False,
                )[0]
                - self.N
            ) / 2
        frames[1:] = frames[0] + expanded_stride[1:].cumsum(dim=0)
        return frames

    @property
    def effective_strides(self: ADSTFT):
        # Compute the strides between window (and not frames)
        expanded_stride = self.actual_strides.expand((self.T,))
        effective_strides = torch.zeros_like(expanded_stride)
        effective_strides[1:] = expanded_stride[1:]
        cat = (
            torch.cat(
                (
                    torch.tensor(
                        [self.N], dtype=self.dtype, device=self.device,
                    ),
                    self.actual_win_length.expand((self.N, self.T)).max(
                        dim=0, keepdim=False,
                    )[0],
                ),
                dim=0,
            ).diff()
            / 2
        )
        effective_strides = effective_strides - cat
        return effective_strides

    def forward(self: DSTFT, x: torch.tensor) -> tuple:
        # Perform the forward STFT and extract the magnitude, phase, real, and imaginary parts
        stft = self.stft(x, "forward")
        #spec = stft.abs().pow(self.pow)[:, : self.F] + torch.finfo(x.dtype).eps

        magnitude = stft.abs()  # Get the magnitude of the STFT
        power_spectrum = magnitude.pow(2)  # Square the magnitude to get the power spectrum
        n = self.N  # Number of frequency bins

        # Handle both cases (even or odd number of frequency bins)
        if n % 2 == 0:
            power_spectrum_last = power_spectrum[..., -1].unsqueeze(-1)
            spec = torch.cat([2 * power_spectrum[..., :-1], power_spectrum_last], dim=-1)
        else:
            power_spectrum_first = power_spectrum[..., 0].unsqueeze(-1)
            spec = torch.cat([2 * power_spectrum[..., 1:], power_spectrum_first], dim=-1)
        spec +=  torch.finfo(x.dtype).eps
        #print(spec.max(), spec.min())
        return spec, stft  # , real, imag, phase

    def abs_forward(self: DSTFT, x: torch.tensor) -> tuple:
        # Perform the forward STFT and extract the magnitude, phase, real, and imaginary parts
        stft = self.stft(x, "forward")
        spec = stft.abs().pow(self.pow)[:, : self.F] + torch.finfo(x.dtype).eps
        return spec, stft
    
    def backward(
        self: ADSTFT, x: torch.tensor, dl_ds: torch.tensor,
    ) -> torch.tensor:
        # Compute the gradient of the loss w.r.t. window length parameter with the chain rule
        dstft_dp = self.stft(x, "backward")
        dl_dp = torch.conj(dl_ds) * dstft_dp
        dl_dp = dl_dp.sum().real.expand(self.win_length.shape)
        # detta har .expand(self.win_length.shape) istället för .unsqueeze(0), viktigt? verkar inte så
        return dl_dp

    def stft(self: ADSTFT, x: torch.tensor, direction: str):
        # batch_size, length, device, dtype = x.shape[0], x.shape[-1], x.device, x.dtype

        # Generate strided signal and shift idx_frac
        folded_x, idx_frac = self.unfold(x)  # B, T, N; T

        # Generate the tapering window function for the STFT
        self.tap_win = self.window_function(
            direction=direction, idx_frac=idx_frac,
        ).permute(2, 1, 0)  # T, N, N

        # Generate tapering function shift
        shift = torch.arange(
            end=self.F,  # CHANGED F->N
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        shift = idx_frac[:, None] * shift[None, :]  # T, N

        # Compute tapered x
        self.folded_x = folded_x[:, :, None, :]  # B, T, 1, N
        self.tap_win = self.tap_win[None, :, :, :]  # 1, T, F, 1
        shift = torch.exp(2j * pi * shift / self.N)[
            None, :, :, None,
        ]  # 1, T, N, 1
        self.tapered_x = self.folded_x * self.tap_win * shift  # B, T, F, N

        # Generate Fourier coefficients
        coeff = torch.arange(
            end=self.N,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        #coeff = coeff[:, None] @ coeff[None, :]  # CHANGED which one is commented
        coeff = coeff[: self.F, None] @ coeff[None, :]
        coeff = torch.exp(-2j * pi * coeff / self.N)  # N, N
        
        # Perform the STFT
        coeff = coeff[None, None, :, :]  # 1, 1, N, N
        stft = (self.tapered_x * coeff).sum(dim=-1)

        #self.kasper = coeff[0, 0, 500, :]  #list(self.tapered_x[0, 0, 500, :])

        return stft.permute(0, 2, 1)

    def unfold(self: ADSTFT, x) -> torch.tensor:
        # frames index and strided x
        idx_floor = self.frames.floor()
        idx_frac = self.frames - idx_floor
        idx_floor = idx_floor.long()[:, None].expand((
            self.T,
            self.N,
        )) + torch.arange(0, self.N, device=self.device)
        idx_floor[idx_floor >= self.L] = -1
        folded_x = x[:, idx_floor]
        folded_x[:, idx_floor < 0] = 0
        return folded_x, idx_frac

    def window_function(
        self: ADSTFT, direction: str, idx_frac,
    ) -> torch.tensor:
        if self.tapering_function not in {"hann", "hanning"}:
            raise ValueError(
                f"tapering_function must be one of '{('hann', 'hanning')}', but got padding_mode='{self.tapering_function}'",
            )
        else:
            # Create an array of indices to use as the base for the window function
            base = torch.arange(
                0, self.N, 1, dtype=self.dtype, device=self.device,
            )[:, None, None].expand([-1, self.F, self.T])  # CHANGED F->N
            base = base - idx_frac
            # Expand the win_length parameter to match the shape of the base array
            # if self.actual_win_length.dim() == 3:
            #    self.expanded_win_length = self.actual_win_length.expand([self.N, self.N, self.T])
            # elif self.actual_win_length.dim() == 1:
            #    self.expanded_win_length = self.actual_win_length[:, None, None].expand([self.N, self.N, self.T])
            # elif self.actual_win_length.dim() == 2 and self.actual_win_length.shape[-1] == self.T:
            #    self.expanded_win_length = self.actual_win_length[:, None, :].expand([self.N, self.N, self.T])
            # elif self.actual_win_length.dim() == 2 and self.actual_win_length.shape[-1] == self.N:
            #    self.expanded_win_length = self.actual_win_length[:, :, None].expand([self.N, self.N, self.T])

        #mask1 = base.ge(torch.ceil((self.N - 1 + self.actual_win_length) / 2))
        #mask2 = base.le(torch.floor((self.N - 1 - self.actual_win_length) / 2))

        # calculate the tapering function and its derivate w.r.t. window length
        if self.tapering_function == 'hann' or self.tapering_function == 'hanning':
            if direction == 'forward':
                self.tap_win = 0.5 - 0.5 * torch.cos(2 * pi * (base + (self.actual_win_length-self.N+1)/2) / self.actual_win_length )                
                mask1 = base.ge(torch.ceil( (self.N-1+self.actual_win_length)/2))
                mask2 = base.le(torch.floor((self.N-1-self.actual_win_length)/2))            
                self.tap_win[mask1] = 0
                self.tap_win[mask2] = 0
                # new
                #self.tap_win = self.tap_win / self.N * 2
                # old CHANGED
                self.tap_win = self.tap_win / self.tap_win.sum(dim=0, keepdim=True)
                #print(self.tap_win.sum(dim=0, keepdim=True))
                #print(self.sr/self.N * 2 *1/self.actual_win_length)
                #s = torch.sqrt(self.N /torch.sum(self.tap_win ** 2, dim=0))
                #print(s.shape)
                #self.tap_win = s* self.tap_win 
                return self.tap_win.pow(self.win_pow)
            
            elif direction == 'backward':
                f = torch.sin(2 * pi * (base - (self.N-1)/2) / self.actual_win_length) 
                # new
                #d_tap_win = -pi / self.actual_win_length.pow(2) * ((self.N - 1) / 2 - base) * f
                #d_tap_win[mask1] = 0
                #d_tap_win[mask2] = 0
                #d_tap_win = d_tap_win / self.N * 2           
                # old CHANGED
                d_tap_win = - pi / self.actual_win_length * ((self.N-1)/2 - base) * f
                d_tap_win[mask1] = 0
                d_tap_win[mask2] = 0
                return d_tap_win
        return None

    def coverage(self: ADSTFT):  # in [0, 1]
        # compute coverage
        expanded_win, _ = self.actual_win_length.expand((self.N, self.T)).min(
            dim=0, keepdim=False,
        )
        cov = expanded_win[0]
        maxi = self.frames[0] + self.N / 2 + expanded_win[0] / 2
        for i in range(1, self.T):
            start = torch.min(
                self.L * torch.ones_like(expanded_win[i]),
                torch.max(
                    torch.zeros_like(expanded_win[i]),
                    self.frames[i] + self.N / 2 - expanded_win[i] / 2,
                ),
            )
            end = torch.min(
                self.L * torch.ones_like(expanded_win[i]),
                torch.max(
                    torch.zeros_like(expanded_win[i]),
                    self.frames[i] + self.N / 2 + expanded_win[i] / 2,
                ),
            )
            if end > maxi:
                cov += end - torch.max(start, maxi)
                maxi = end
        cov /= self.L
        return cov

    def plot(
        self: ADSTFT,
        spec: torch.tensor,
        x: Optional[torch.tensor] = None,
        marklist: Optional[List[int]] = None,
        bar: bool = False,
        figsize: Tuple[float, float] = (6.4, 4.8),
        f_hat=None,
        fs=None,
        *,
        weights: bool = True,
        wins: bool = True,
        cmap: float = "jet",
        ylabel: float = "frequencies",
        xlabel: float = "frames",
    ):
        plt.figure(figsize=figsize)
        plt.title("Spectrogram")
        ax = plt.subplot()
        im = ax.imshow(
            spec[0].detach().cpu().log(),
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=[0, spec.shape[-1], 0, spec.shape[-2]],
        )
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if bar is True:
            plt.colorbar(im, ax=ax)
        plt.show()

        if weights is True:
            plt.figure(figsize=figsize)
            plt.title("Distribution of window lengths")
            ax = plt.subplot()
            im = ax.imshow(
                self.actual_win_length[: self.F].detach().cpu(),
                aspect="auto",
                origin="lower",
                cmap=cmap,
            )
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            if bar is True:
                plt.colorbar(im, ax=ax)
                im.set_clim(self.win_min, self.win_max)
            plt.show()

        if self.tap_win is not None and wins is True:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(self.T + 0.5 + x.squeeze().cpu().numpy(), linewidth=1)
            for i, start in enumerate(self.frames.detach().cpu()):
                ax.plot(
                    range(
                        int(start.floor().item()),
                        int(start.floor().item() + self.N),
                    ),
                    self.T
                    - i
                    - 1.3
                    + 150
                    * self.tap_win[:, i, :, :]
                    .mean(dim=1)
                    .squeeze()
                    .detach()
                    .cpu(),
                    c="#1f77b4",
                )

            if marklist is not None:
                for elem in marklist:
                    plt.axvline(elem, 0, self.T, c="gray")
            else:
                ax.axvline(x=0, ymin=0, ymax=self.T, c="gray")
                ax.axvline(x=x.shape[-1], ymin=0, ymax=self.T, c="gray")
            plt.show()

class FDSTFT(nn.Module):
    """Differentiable window length only short-time Fourier transform (DSTFT) module.
    only one window length fot stft optimizable by gradeint descent, no 

    Args:
    ----
        nn (_type_): _description_

    """

    def __init__(
        self: FDSTFT,
        x: torch.tensor,
        win_length: float,
        support: int,
        stride: int,
        win_requires_grad=True,
        win_min: float | None = None,
        win_max: float | None = None,
        tapering_function: str = "hann",
        sr: int = 16_000,
        window_transform=None,
        stride_transform=None,
    ):
        super().__init__()

        # Constants and hyperparameters
        self.N = support  # support size
        self.F = int(1 + self.N / 2)  # nb of frequencies
        self.B = x.shape[0]  # batch size
        self.L = x.shape[-1]  # signal length
        self.device = x.device
        self.dtype = x.dtype

        self.win_requires_grad = win_requires_grad
        self.tapering_function = tapering_function
        self.sr = sr
        self.tap_win = None

        # Register eps and min as a buffer tensor
        self.register_buffer(
            "eps",
            torch.tensor(
                torch.finfo(torch.float).eps,
                dtype=self.dtype,
                device=self.device,
            ),
        )
        self.register_buffer(
            "min",
            torch.tensor(
                torch.finfo(torch.float).min,
                dtype=self.dtype,
                device=self.device,
            ),
        )

        if win_min is None:
            self.win_min = self.N / 20
        else:
            self.win_min = win_min
        if win_max is None:
            self.win_max = self.N
        else:
            self.win_max = win_max

        # HOP LENGTH / FRAME INDEX
        # hop length constraints
        if stride_transform is None:
            self.stride_transform = self.__stride_transform
        else:
            self.stride_transform = stride_transform


        # WIN LENGTH
        # win length constraints
        if window_transform is None:
            self.window_transform = self.__window_transform
        else:
            self.window_transform = window_transform

        # Create the window length parameter and assign it the appropriate shape
        self.win_length = nn.Parameter(
            torch.full(
                1,
                abs(win_length),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.win_requires_grad,
        )


    def __window_transform(self: DSTFT, w_in):
        w_out = torch.minimum(
            torch.maximum(
                w_in,
                torch.full_like(
                    w_in, self.win_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                w_in, self.win_max, dtype=self.dtype, device=self.device,
            ),
        )
        return w_out

    def __stride_transform(
        self: DSTFT, s_in: torch.Tensor,
    ):  # born stride entre 0 et 2N
        s_out = torch.minimum(
            torch.maximum(
                s_in,
                torch.full_like(
                    s_in, self.stride_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                s_in, self.stride_max, dtype=self.dtype, device=self.device,
            ),
        )
        return s_out

    def __pow_transform(
        self: DSTFT, p_in: torch.Tensor,
    ):  # born stride entre 0 et 2N
        p_out = torch.minimum(
            torch.maximum(
                p_in,
                torch.full_like(
                    p_in, self.pow_min, dtype=self.dtype, device=self.device,
                ),
            ),
            torch.full_like(
                p_in, self.pow_max, dtype=self.dtype, device=self.device,
            ),
        )
        return p_out

    @property
    def actual_win_length(self: DSTFT):  # contraints
        return self.window_transform(self.win_length)

    @property
    def actual_strides(
        self,
    ):  # stride contraints, actual stride between frames
        return self.stride_transform(self.strides)

    @property
    def actual_pow(self: DSTFT):  # pow contraints
        return self.pow_transform(self.win_pow)

    @property
    def frames(self: DSTFT):
        # Compute the temporal position (indices) of frames (support)
        expanded_stride = self.actual_strides.expand((self.T,))
        frames = torch.zeros_like(expanded_stride)
        # frames[0] = - self.N / 2
        # if self.first_frame:
        #    frames[0] = (
        #        self.actual_win_length.expand((self.N, self.T))[:, 0].max(
        #            dim=0, keepdim=False,
        #        )[0]
        #        - self.N
        #    ) / 2
        frames -= self.N / 2 + self.init_stride
        frames += expanded_stride.cumsum(dim=0)

        return frames

    @property
    def effective_strides(self: DSTFT):
        # Compute the strides between window (and not frames)
        expanded_stride = self.actual_strides.expand((self.T,))
        effective_strides = torch.zeros_like(expanded_stride)
        effective_strides[1:] = expanded_stride[1:]
        cat = (
            torch.cat(
                (
                    torch.tensor(
                        [self.N], dtype=self.dtype, device=self.device,
                    ),
                    self.actual_win_length.expand((self.N, self.T)).max(
                        dim=0, keepdim=False,
                    )[0],
                ),
                dim=0,
            ).diff()
            / 2
        )
        effective_strides = effective_strides - cat
        return effective_strides

    def forward(self: DSTFT, x: torch.tensor) -> tuple:
        # Perform the forward STFT and extract the magnitude, phase, real, and imaginary parts
        stft = self.stft(x, "forward")
        spec = stft.abs().pow(self.pow)[:, : self.F] + torch.finfo(x.dtype).eps
        return spec, stft  # , real, imag, phase

    def backward(
        self: DSTFT, x: torch.tensor, dl_ds: torch.tensor,
    ) -> torch.tensor:
        # Compute the gradient of the loss w.r.t. window length parameter with the chain rule
        dstft_dp = self.stft(x, "backward")
        dl_dp = torch.conj(dl_ds) * dstft_dp
        dl_dp = dl_dp.sum().real.expand(self.win_length.shape)
        return dl_dp

    def stft(self: DSTFT, x: torch.tensor, direction: str):
        # batch_size, length, device, dtype = x.shape[0], x.shape[-1], x.device, x.dtype

        # Generate strided signal and shift idx_frac
        folded_x, idx_frac = self.unfold(x)  # B, T, N; T

        # Generate the tapering window function for the STFT
        self.tap_win = self.window_function(
            direction=direction, idx_frac=idx_frac,
        ).permute(1, 0)  # T, N

        # Compute tapered x
        self.folded_x = folded_x[:, :, :]  # B, T, N
        self.tap_win = self.tap_win[None, :, :]  # 1, T, 1
        self.tapered_x = self.folded_x * self.tap_win  # B, T, N,

        spectr = torch.fft.rfft(self.tapered_x)

        shift = torch.arange(
            end=self.F,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        shift = idx_frac[:, None] * shift[None, :]  # T, N
        shift = torch.exp(2j * pi * shift / self.N)[None, ...]  # 1, T, N

        stft = spectr * shift
        return stft.permute(0, 2, 1)

    def inverse_dstft(self: DSTFT, stft: torch.Tensor) -> torch.tensor:
        """Compute inverse differentiable short-time Fourier transform (IDSTFT).

        Args:
        ----
            self (DSTFT): _description_
            stft (torch.Tensor): _description_

        Returns:
        -------
            torch.tensor: _description_

        """
        # shift
        # shift = torch.arange(
        #     end=self.F,
        #     device=self.device,
        #     dtype=self.dtype,
        #     requires_grad=False,
        # )
        # shift = idx_frac[:, None] * shift[None, :]  # T, N
        # stft = stft * torch.exp(2j * pi * shift / self.N)[None, ...]  # 1, T, N
        # print(stft.shape)

        # inverse
        # print(stft.shape, stft.dtype)
        ifft = torch.fft.irfft(stft, n=self.N, dim=-2)
        # print(ifft.shape, self.tap_win.sum(-1, keepdim=True).shape)
        ifft = (
            ifft  # * self.tap_win.sum(dim=-1, keepdim=True).permute(0, 2, 1)
        )
        # print(ifft.shape, ifft.dtype)wins2 = torch.zeros(x.shape[-1])


        # tapered
        # tap_win = torch.conj(self.tap_win) #= self.tap_win[None, :, :]  # 1, T, 1
        ifft = ifft.permute(0, -1, -2)  # * tap_win
        # print(ifft.shape, ifft.dtype)

        # print(ifft.sum())

        # fold
        x_hat = self.fold(ifft)

        return x_hat

    def unfold(self: DSTFT, x: torch.tensor) -> torch.tensor:
        # frames index and strided x
        idx_floor = self.frames.floor()
        # print(self.frames.shape, self.frames)
        idx_frac = self.frames - idx_floor
        idx_floor = idx_floor.long()[:, None].expand((
            self.T,
            self.N,
        )) + torch.arange(0, self.N, device=self.device)
        idx_floor[idx_floor >= self.L] = -1
        # print(self.B, idx_floor.shape, x.shape)
        folded_x = x[:, idx_floor]
        folded_x[:, idx_floor < 0] = 0
        return folded_x, idx_frac

    def fold(self: DSTFT, folded_x: torch.tensor) -> torch.tensor:
        x_hat = torch.zeros(
            self.B, self.L, device=self.device, dtype=self.dtype,
        )
        # print(x_hat.shape, self.B, self.L)
        #print(folded_x.shape)
        for t in range(self.T):
            start_idx = max(0, int(self.frames[t]))
            end_idx = min(self.L - 1, int(self.frames[t]) + self.N)
            start_dec = start_idx - int(self.frames[t])
            end_dec = end_idx - int(self.frames[t])
            x_hat[:, start_idx:end_idx] += folded_x[:, t, start_dec:end_dec]
        return x_hat

    def window_function(self: DSTFT, direction: str, idx_frac) -> torch.tensor:
        if self.tapering_function not in {"hann", "hanning"}:
            raise ValueError(
                f"tapering_function must be one of '{('hann', 'hanning')}', but got padding_mode='{self.tapering_function}'",
            )
        else:
            # Create an array of indices to use as the base for the window function
            base = torch.arange(
                0, self.N, 1, dtype=self.dtype, device=self.device,
            )[:, None].expand([-1, self.T])
            base = base - idx_frac
            # Expand the win_length parameter to match the shape of the base array

        # calculate the tapering function and its derivate w.r.t. window length
        mask1 = base.ge(torch.ceil((self.N - 1 + self.actual_win_length) / 2))
        mask2 = base.le(torch.floor((self.N - 1 - self.actual_win_length) / 2))
        if (
            self.tapering_function == "hann"
            or self.tapering_function == "hanning"
        ):
            if direction == "forward":
                self.tap_win = 0.5 - 0.5 * torch.cos(
                    2
                    * pi
                    * (base + (self.actual_win_length - self.N + 1) / 2)
                    / self.actual_win_length,
                )
                self.tap_win[mask1] = 0
                self.tap_win[mask2] = 0
                # self.tap_win = self.tap_win / self.tap_win.sum(
                #    dim=0, keepdim=True,
                # )
                return self.tap_win.pow(self.win_pow)

            elif direction == "backward":
                f = torch.sin(
                    2
                    * pi
                    * (base + (self.actual_win_length - self.N + 1) / 2)
                    / self.actual_win_length,
                )
                d_tap_win = (
                    -pi
                    / self.actual_win_length.pow(2)
                    * ((self.N - 1) / 2 - base)
                    * f
                )
                d_tap_win[mask1] = 0
                d_tap_win[mask2] = 0
                d_tap_win = d_tap_win / self.N * 2
                return d_tap_win
        return None

    def synt_win(self: DSTFT, direction: str, idx_frac) -> torch.tensor:
        return

    def coverage(self: DSTFT):  # in [0, 1]
        # compute coverage
        expanded_win, _ = self.actual_win_length.expand((self.N, self.T)).min(
            dim=0, keepdim=False,
        )
        cov = expanded_win[0]
        maxi = self.frames[0] + self.N / 2 + expanded_win[0] / 2
        for i in range(1, self.T):
            start = torch.min(
                self.L * torch.ones_like(expanded_win[i]),
                torch.max(
                    torch.zeros_like(expanded_win[i]),
                    self.frames[i] + self.N / 2 - expanded_win[i] / 2,
                ),
            )
            end = torch.min(
                self.L * torch.ones_like(expanded_win[i]),
                torch.max(
                    torch.zeros_like(expanded_win[i]),
                    self.frames[i] + self.N / 2 + expanded_win[i] / 2,
                ),
            )
            if end > maxi:
                cov += end - torch.max(start, maxi)
                maxi = end
        cov /= self.L
        return cov

    def plot(
        self: DSTFT,
        spec: torch.Tensor,
        x: torch.Tensor | None = None,
        marklist: Optional[List[Any]] = None,
        figsize=(6.4, 4.8),
        f_hat=None,
        fs=None,
        *,
        weights: bool = True,
        wins: bool = True,
        bar: bool = False,
        cmap: float = "jet",
        ylabel: float = "frequencies",
        xlabel: float = "frames",
    ):
        f_max = spec.shape[-2] if fs is None else fs / 2
        plt.figure(figsize=figsize)
        plt.title("Spectrogram")
        ax = plt.subplot()
        im = ax.imshow(
            spec[0].detach().cpu().log(),
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=[0, spec.shape[-1], 0, f_max],
        )
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if bar == True:
            plt.colorbar(im, ax=ax)
        if f_hat is not None:
            for f in f_hat:
                plt.plot(f, linewidth=0.5, c="k", alpha=0.7)
        plt.show()

        if weights == True:
            plt.figure(figsize=figsize)
            plt.title("Distribution of window lengths")
            ax = plt.subplot()
            im = ax.imshow(
                self.actual_win_length[: self.F].detach().cpu(),
                aspect="auto",
                origin="lower",
                cmap=cmap,
            )
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            if bar == True:
                plt.colorbar(im, ax=ax)
                im.set_clim(self.win_min, self.win_max)
            plt.show()

        if self.tap_win is not None and wins == True:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(self.T + 0.5 + x.squeeze().cpu().numpy(), linewidth=1)
            for i, start in enumerate(self.frames.detach().cpu()):
                ax.plot(
                    range(
                        int(start.floor().item()),
                        int(start.floor().item() + self.N),
                    ),
                    self.T
                    - i
                    - 1.3
                    + self.tap_win[:, i, :].squeeze().detach().cpu(),
                    c="#1f77b4",
                )

            if marklist is not None:
                for elem in marklist:
                    plt.axvline(elem, 0, self.T, c="gray")
            else:
                ax.axvline(x=0, ymin=0, ymax=self.T, c="gray")
                ax.axvline(x=x.shape[-1], ymin=0, ymax=self.T, c="gray")
            plt.show()
