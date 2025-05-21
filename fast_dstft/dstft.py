from __future__ import annotations

from math import pi
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.utils.benchmark as benchmark

##############
### Timing ###
##############
import time
import os
import pandas as pd
csv_file = "timing.csv"
timing_log = {}
def time_function(f, device, csv_file=None, column_name=None):
    global timing_log

    # Timing
    if device.type in ['mps', 'cuda']:
        torch.cuda.synchronize() if device.type == 'cuda' else torch.mps.synchronize()
    start = time.time()
    result = f()
    if device.type in ['mps', 'cuda']:
        torch.cuda.synchronize() if device.type == 'cuda' else torch.mps.synchronize()
    end = time.time()
    elapsed = end - start

    # Print if this is what is wanted
    if csv_file is None or column_name is None:
        print(f"Elapsed time: {elapsed:.4f} seconds")
        return result
    
    # Update timing_log
    if column_name not in timing_log:
        timing_log[column_name] = []
    timing_log[column_name].append(elapsed)
    
    # Check if all columns have the same length (i.e., full row ready)
    lengths = [len(lst) for lst in timing_log.values()]
    if (not all(l == lengths[0] for l in lengths)) or (max(lengths) < 2):
        return result  # Not a full row yet, or just one, so we cannot know the number of columns
    
    # Build and append the new row
    row = {col: timing_log[col][-1] for col in sorted(timing_log)}
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(csv_file)
    df_row.to_csv(csv_file, mode='a', header=write_header, index=False)

    return result
def reset_timing():
    global timing_log
    timing_log = {}
    if os.path.exists(csv_file):
        os.remove(csv_file)
def analyze_csv_timings():
    # Load the CSV
    df = pd.read_csv(csv_file)

    # Drop completely empty columns (just in case)
    df = df.dropna(axis=1, how='all')

    # Keep only numeric timing columns
    timing_df = df.select_dtypes(include=[np.number])

    if timing_df.empty:
        print("No numeric timing data found.")
        return None, None, None

    # Drop the first row (initialization overhead)
    if len(timing_df) <= 1:
        print("Not enough data after dropping warm-up row.")
        return None, None, None
    timing_df = timing_df.iloc[1:]

    # Compute statistics
    averages = timing_df.mean()
    total_average_time = averages.sum()
    percentages = 100 * averages / total_average_time

    # Output
    print("MEANS")
    print("Mean time per step:")
    print(averages)
    print(f"\nTotal mean time per run: {total_average_time:.6f} seconds")
    print(f"\nPercentage of total mean time per step:")
    print(percentages)

    medians = timing_df.median()
    total_average_time = medians.sum()
    percentages = 100 * medians / total_average_time

    print(f"\nMEDIANS")
    print("Median time per step:")
    print(medians)
    print(f"\nTotal median time per run: {total_average_time:.6f} seconds")
    print(f"\nPercentage of total median time per step:")
    print(percentages)
import torch.utils.bottleneck as bottleneck
import numpy as np
from scipy.signal import stft
from scipy.stats import beta
from scipy.interpolate import BSpline
import torch.nn.functional as F
import cv2

#################
### fastDSTFT ###
#################
class fastDSTFT(nn.Module):
    ######################
    ### Initialization ###
    ######################
    def __init__(self,
                    n_fft: int,                
                    stride: int, 
                    win_max: float = None,  # this is an important parameter!
                    win_min: float = None,
                    initial_win_length: float = None,
                    window_function: str = 'beta',
                    spline_density: int = 2,  # number of splines per stride
                    spline_degree: int = 5,  # degree of the spline
                    sr: int = 16_000,
                    padding: str = "same",  # "valid", "same-rolloff", "same"
                ):
        super().__init__() 
        # Full initialization happens when the first batch of x is passed
        self.fully_initialized = False
        
        # Save the inputs for lazy initialization
        self.sr = sr
        self.stride = stride
        self.window_function = window_function
        self.N = n_fft
        self.F =  self.N//2 + 1  #int(1 + self.N/2)  # number of frequencies
        if initial_win_length is None: initial_win_length = support/2
        self.initial_win_length = initial_win_length
        self.spline_degree = spline_degree
        self.spline_density = spline_density
        if padding != "valid" and (self.spline_degree%2)==0:
            raise ValueError("Even spline degrees cannot give same-padding, as they cannot align with start and end")
        self.padding = padding
        if self.stride/self.spline_density % 1 != 0:
            raise ValueError("spline_density must divide the stride")  # tillfällig förenkling
        self.spline_stride = int(self.stride/self.spline_density)
        self.spline_support = int((self.spline_degree+1)*self.spline_stride)
        # the function below seems to fit well to the width of a spline (counted as the width containing 99.9% of the area under the spline)
        spline_width = -2+1.93*self.spline_stride*self.spline_degree**(1/2)
        if win_min is None: win_min = spline_width
        if win_min < spline_width:
            Warning(f"The minimum window length has been set to {win_min} which is less tham the width of a spline around {spline_width}")
        else:
            print(f"The minimum reachable window length with this configuration is around {-2+1.93*self.spline_stride*self.spline_degree**(1/2)}")
            # Note that, for example, a Hann window of length 100 has a width less than 100, so this might be a bit misleading
        self.min_length = win_min
        if win_max is None: win_max = self.N
        self.max_length = win_max
        if (1+(self.max_length-self.spline_support)/self.spline_stride) % 2 != 1:
            raise ValueError(f"An odd number of splines must fit perfectly into the maximum window length. \
                             \nspline_stride={self.spline_stride}, spline_support={self.spline_support}, max_length={self.max_length}")
        self.window_lengths = None
    def __lazy_initialization(self, x_sample):  # initializations that depend on x
        # New info about x
        self.x = x_sample
        self.L = x_sample.shape[-1]
        self.B = x_sample.shape[0]  
        self.dtype = x_sample.dtype                                                               # batch size  
        self.eps = torch.finfo(self.dtype).eps
        self.device = x_sample.device
        # Moving the class instance to the same device as x
        self.to(self.device)

        # For expanding the spline stft (step 3)
        self.s = int(1+(self.max_length-self.spline_support)/self.spline_stride)  # splines per frame

        # For computing spline stft (step 1)
        if self.padding=="valid":
            self.T = int(1 + torch.div(self.L - self.max_length, self.stride, rounding_mode='floor'))  # time steps
            L_useful = (self.max_length-self.stride)+self.T*self.stride  # we might miss some points in the end
        elif "same" in self.padding:
            self.T = 1 + int(torch.div(self.L, self.stride, rounding_mode="floor"))
            L_useful = self.spline_stride*int(torch.div(self.L, self.spline_stride, rounding_mode="floor"))  # still might miss some, but possibly less
            
            self.extraS = int((self.max_length/2)/self.spline_stride)  # the number of extra splines on each side
            if self.padding == "same": 
                # each sample is considered to "cover" an area of 0.5 to each side of it
                # the first frame is centered 0.5 before the first sample, 
                # the last frame is centered 0.5 after the last sample
                # this is the most symmetric way to do it, also with respect to the spline windows
                # this note also hold for valid padding
                frame_centers = self.stride*torch.arange(self.T, device=self.device) -0.5
                spline_centers = torch.arange(-self.max_length/2+self.spline_support/2-0.5, frame_centers[-1]+self.max_length/2-self.spline_support/2+self.spline_stride-0.5, self.spline_stride)
                unfolded_spline_centers = spline_centers.unfold(0, self.s, self.spline_density).T.contiguous()
                self.fake_stft_mask = (((unfolded_spline_centers-self.spline_support/2+0.5)<0) | ((unfolded_spline_centers+self.spline_support/2-0.5)>(L_useful-1))).to(self.device)
        else:
            raise ValueError("the padding mode is not known")
        self.S = int(1+(L_useful-self.spline_support)/self.spline_stride)
        self.bspline_window = self._generate_bspline().to(self.device)  # generating the B-spline window
        self.x_windowed_pad_buffer = torch.zeros((self.B, self.N, self.S), device=self.device)  # speeds up padding

        # For modulating the spline stft (step 2)
        spline_window_starts = self.spline_stride*torch.arange(0, self.S, device=self.device)  # spline offsets from the signal start
        self.modulation_factors = torch.exp(-1j * 2 * torch.pi * torch.arange(self.F, device=self.device).unsqueeze(1) / self.N * spline_window_starts.unsqueeze(0))
        # the phase reference is the start of the signal, index 0

        # If fake stft values are needed (step 2.5)
        if "same" in self.padding:
            self.spline_stft_pad_buffer = torch.zeros((self.B, self.F, self.extraS+self.S+self.extraS), dtype=torch.complex64, device=self.device)
       
        # For the coefficients (step 4)
        offsets = torch.arange(-self.max_length/2+self.spline_support/2,  # positions of splines in a window relative to the middle
                            self.max_length/2-self.spline_support/2+self.spline_stride, 
                            self.spline_stride, device=self.device) 
        self.normalized_offsets = offsets / self.max_length  # offsets normalized to [-0.5, 0.5], used for Hann windows
        # (S)
        x = 0.5+self.normalized_offsets
        x_1mx = (x*(1-x)).unsqueeze(0).unsqueeze(2)
        # (1, s, 1)
        # going through logs to avoid numerical issues
        self.log_x_1mx = torch.log(x_1mx)  # used for Beta windows
        # (1, s, 1)

        # The PARAMETER
        self.window_lengths = nn.Parameter(torch.full((self.F, self.T), self.initial_win_length, dtype=self.dtype, device=self.device), requires_grad=True)
        # ? self.register_parameter("window_lengths", self.window_lengths)

        self.fully_initialized = True
    def _generate_bspline(self):
        num_knots = self.spline_degree +2
        # Create the knot vector for a single basis element
        #knots = np.linspace(0, self.spline_support, num_knots)
        #spline = BSpline.basis_element(knots)
        #t = np.linspace(0, self.spline_support, self.spline_support)
        # above does not give a partititon of unity, the sum of the splines only goes to 0.99
        # this is correct: 
        knots = np.linspace(-0.5, self.spline_support-0.5, num_knots)  # the knots are between the samples
        spline_object = BSpline.basis_element(knots)
        t = np.linspace(0, self.spline_support-1, self.spline_support)
        spline = torch.from_numpy(spline_object(t)).float()
        return spline/spline.sum()
        
    ####################
    ### Forward pass ###
    ####################
    # CPU: handles batches size 1024+, fastest at 64 with 0.0016 per sample
    # GPU: handles batches of up to 512, fastest at 16 with 0.0016 per sample
    def forward(self, x):  # B=64: gpu->cpu takes 0.0037, cpu->gpu takes 0.0050
        if not self.fully_initialized:  # do the x-contextual initialization
            self.__lazy_initialization(x)
        # compute the stft by stfting with spline window and then combining the values
        stft = self._stft(x)
        return stft.abs() + self.eps  # compute the spectrogram 
    def _stft(self, x):  # timing at B=64, using "same" padding
        # Compute preliminary STFT with spline windows (63%, 33% of computation using CPU, GPU)
        spline_stft = self._spline_stft(x)  # step 1
        #spline_stft = time_function(lambda: self._spline_stft(x), self.device, csv_file, "spline_stft")
        # (B, F, S)

        # Modulate the FFTs to compensate for their temporal position (6%=0.002, 14%=0.003 of computation using CPU, GPU)
        spline_stft.mul_(self.modulation_factors)  # step 2
        #time_function(lambda: spline_stft.mul_(self.modulation_factors), self.device, csv_file, "modulation")
        # (B, F, S)

        if "same" in self.padding:  # step 2.5, adding some "fake" zeros on both ends
            self.spline_stft_pad_buffer[:, :, self.extraS:self.extraS+self.S] = spline_stft
            spline_stft = self.spline_stft_pad_buffer  # F.pad(spline_stft, pad=(self.extraS, self.extraS), mode='constant', value=0)
            # (B, F, extraS+S+extraS)

        # Build tensor of all splines with repetition (<1% of computation)
        stride_B, stride_F, stride_S = spline_stft.stride()
        expanded_spline_stfts = spline_stft.as_strided(
            size=(self.B, self.F, self.s, self.T),
            stride=(stride_B, stride_F, stride_S, self.spline_density*stride_S),
        )  # step 3
        # (B, F, s, T) it thinks it is, but is actually still (B, F, S)->memory saving!

        # Get the spline coefficients from the window lengths (4%, 4% of computation using CPU, GPU)
        coeffs = self._coefficients(self.window_lengths).unsqueeze(0)  # step 4
        #coeffs = time_function(lambda: self._coefficients(self.window_lengths).unsqueeze(0), self.device, csv_file, "coefficients")
        # (1, F, s, T)
        
        # Calculate the STFT (33%=0.018, 63%=0.037 of computation using CPU, GPU)
        return self.__fast_contraction(expanded_spline_stfts, coeffs)
        #return time_function(lambda: self.__fast_contraction(expanded_spline_stfts, coeffs), self.device, csv_file, "final computation")  # step 5
        # (B, F, T)
    def _spline_stft(self, x):  # faster and more transparent than torch.stft
        # Extract slices of x
        x_unfolded = x.unfold(1, self.spline_support, self.spline_stride).transpose(1, 2)
        # (B, spline_support, S)

        # Apply B-spline windowing
        x_windowed = x_unfolded*self.bspline_window.view(1, -1, 1)
        # (B, spline_support, S)

        # Zero-pad each frame to length N. symmetrical matches torch.stft, but gives a more complicated reference
        self.x_windowed_pad_buffer[:, :self.spline_support, :] = x_windowed  # self.pad_buffer[:, int(self.N/2-self.spline_support/2):int(self.N/2+self.spline_support/2), :] = x_unfolded
        # (B, N, S)

        # FFT the windows (96% of this function on CPU)
        return torch.fft.rfft(self.x_windowed_pad_buffer, dim=1)
        # (B, F, S)
    @torch.compile  # speeds up the "coefficients" function by a lot
    def _coefficients(self, window_lengths):  # get the spline coefficients
        # The Greville abscissae are the centers of the splines!
        if self.window_function == 'beta':
            # Calculate the a=b parameter of the Beta distribution (4% of this function on CPU)
            a = 1/2*((3*self.max_length/window_lengths)**2 -1).unsqueeze(1)
            # (F, 1, T)

            # Raise the base to the exponent a (17% of this function on CPU)
            log_pdf_ish = (a - 1) * self.log_x_1mx
            # (F, s, T)

            # Coefficients for fake spline stfts should be zero to preserve the spectrogram energy (this takes a while actually, about 25% of the function with it)
            if self.padding == "same":
                log_pdf_ish.masked_fill_(self.fake_stft_mask.unsqueeze(0), -torch.inf)
            # (F, s, T)

            # Normalize the coefficients to 1 (55% of this function on CPU) and exponentiate back (24% of this function on CPU)
            return torch.exp(log_pdf_ish -log_pdf_ish.logsumexp(dim=1, keepdim=True))  # F.softmax(log_pdf_ish, dim=1)
            # (F, s, T)
        elif self.window_function == 'hann':
            half_width = (window_lengths/(2 * self.N)).unsqueeze(1)
            # (F, 1, T)

            normalized_offsets = self.normalized_offsets.unsqueeze(0)
            if len(half_width.shape)==3:  # if a has all time steps, we need to add a dimension
                normalized_offsets = normalized_offsets.unsqueeze(2)
            # (1, s, 1)

            # normalized offsets is already centered
            out = 0.5 * (1 + torch.cos(torch.pi * normalized_offsets / half_width))
            out[torch.abs(normalized_offsets) > half_width] = 0
            return out / out.sum(dim=1, keepdim=True)
            # (F, s, T)
        else:
            raise NotImplementedError(f"Window function '{self.window_function}' not implemented.")
    def __fast_contraction(self, expanded_spline_stfts, coeffs):  # wrapper because compile and complex don't like each other
        real, imag = self.__combine(expanded_spline_stfts.real, expanded_spline_stfts.imag, coeffs)
        return torch.complex(real, imag)
    @torch.compile  # compiled "combine" function avoids materializing expanded_spline_stfts
    def __combine(self, expanded_spline_stfts_real, expanded_spline_stfts_imag, coeffs):  # ~half time by not making complex multiply with coeffs
        real = (expanded_spline_stfts_real * coeffs).sum(dim=2)
        imag = (expanded_spline_stfts_imag * coeffs).sum(dim=2)
        return real, imag
     
    ##################################
    ### set/get the window lengths ###
    ##################################
    def get_window_lengths(self):  # get detached window lengths
        return self.window_lengths.detach()
    def set_window_lengths(self, new_window_lengths):
        with torch.no_grad:
            self.window_lengths.data.copy_(torch.tensor(new_window_lengths))

    ########################################
    ### Warm start of the window lengths ### not super reliable or well motivated right now, only tried with "valid" padding
    ########################################
    def warm_start(self):
        with torch.no_grad():
            self.window_lengths.data.fill_(self.min_length)
        image = np.array(self().squeeze(0).detach())
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
        ordered_window_length = self.__enforce_ordering(smoothed_window_length)
        with torch.no_grad():
            self.window_lengths.data.copy_(torch.tensor(ordered_window_length))
    def __enforce_ordering(self, value):  # helper function for warm start
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
    
    ############################################################################
    ### Put window lengths back within the allowed range during optimization ###
    ############################################################################
    def put_windows_back(self):
        minimum = self.min_length
        maximum = self.max_length
        with torch.no_grad():
            self.window_lengths.clamp_(min=minimum, max=maximum)

    ###############################################
    ### Plot the spectrogram and window lengths ###
    ###############################################
    def plot(self, spec, log: bool = True, weights: bool = True, title=""):
        plt.figure(figsize=(6.4, 4.8))
        plt.title("Spectrogram "+title)
        ax = plt.subplot()
        if log:
            im = ax.imshow(spec[0].detach().cpu().log(), 
                aspect="auto", 
                origin="lower", 
                cmap="jet",
                interpolation='nearest',
                )
        else:
            im = ax.imshow(spec[0].detach().cpu(), 
                aspect="auto", 
                origin="lower", 
                cmap="jet",
                interpolation='nearest',
                )
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
            im.set_clim(self.min_length, self.max_length)
            plt.show()

#######################
### Transform ideas ###
#######################
# transforms get gradient issues, exploding or vanishing:( maybe fixable
#self.window_lengths = torch.full((self.F, self.T), initial_win_length, dtype=self.dtype, device=self.device)
#self.transformed_window_lengths = nn.Parameter(self.window_transform(self.window_lengths), requires_grad=True)
def set_window_lengths(self, new_window_lengths):
    with torch.no_grad:
        self.transformed_window_lengths.data.copy_(self.window_transform(torch.tensor(new_window_lengths)))
# normalization
def window_transform(self, window_lengths):  # takes real window lengths to transformed window lengths
    return (window_lengths -self.min_length)/(self.max_length-self.min_length)
def inverse_window_transform(self, transformed_window_lengths):
    return self.min_length +(self.max_length-self.min_length)*transformed_window_lengths
def put_windows_back(self):
    with torch.no_grad():
        self.transformed_window_lengths.clamp_(min=self.window_transform(self.min_length), max=self.window_transform(self.max_length))
# little transform: faster but slightly unstable
def window_transform(self, window_lengths):  # takes real window lengths to transformed window lengths
    if self.window_function == "beta":  # the transformed parameter is a normalized "a"
        self.a_max = 1/2*((3*self.N/self.min_length)**2 -1)
        self.a_min = 1/2*((3*self.N/self.max_length)**2 -1)
        transformed_window_lengths = 1/2*((3*self.N/window_lengths)**2 -1)
        return transformed_window_lengths
    else: raise NotImplementedError("window_transform is not available for Hann yet")
def inverse_window_transform(self, transformed_window_lengths):
    if self.window_function == "beta":  # the transformed parameter is "a"
        window_lengths = 3*self.N /torch.sqrt(1 +2*transformed_window_lengths)
        return window_lengths
    else: raise NotImplementedError("window_transform is not available for Hann yet")
def put_windows_back(self):
    with torch.no_grad():
        self.transformed_window_lengths.clamp_(min=self.a_min, max=self.a_max)
def coefficients(self):  # get the spline coefficients
    # The Greville abscissae are the centers of the splines! Kind of, boundary effect?
    if self.window_function == 'beta':
        # Calculate the a=b parameter of the Beta distribution (46% of GPU computation, 0.00018 s)
        a = self.transformed_window_lengths.unsqueeze(1)
        #?
        # (F, T)

        # Raise the base to the exponent a (29% of GPU computation, 0.00011)
        log_pdf_ish = (a - 1) * self.log_x_1mx
        #log_pdf = log_pdf_ish -log_pdf_ish.logsumexp(dim=1, keepdim=True)
        #pdf = torch.exp(log_pdf)
        # (F, s, T)
        #return pdf

        # Normalize the coefficients to 1 and exponentiate back (24% of GPU computation, 0.00009)
        return torch.exp(log_pdf_ish -log_pdf_ish.logsumexp(dim=1, keepdim=True))  # F.softmax(log_pdf_ish, dim=1)
    elif self.window_function == 'hann':
        half_width = (lambda_/(2 * self.N)).unsqueeze(1)
        # (F, 1, T) for high_memory forward

        normalized_offsets = normalized_offsets.unsqueeze(0)
        if len(half_width.shape)==3:  # if a has all time steps, we need to add a dimension
            normalized_offsets = normalized_offsets.unsqueeze(2)
        # (1, s, 1) for high_memory forward

        # normalized offsets is already centered
        out = 0.5 * (1 + torch.cos(torch.pi * normalized_offsets / half_width))
        out[torch.abs(normalized_offsets) > half_width] = 0
        out = out / out.sum(dim=1, keepdim=True)
        #mask = torch.abs(normalized_offsets) <= half_width
        #out = torch.zeros((self.F, self.s, self.T), dtype=torch.float32)
        #double_normalized_offsets = normalized_offsets / half_width
        #out[mask] = 0.5 * (1 + torch.cos(torch.pi * double_normalized_offsets[mask]))
        #out = out / out.sum(dim=1, keepdim=True)
        # (F, s, T) for high_memory forward

        return out
    else:
        raise NotImplementedError(f"Window function '{self.window_function}' not implemented.")
# big transform: ?
def window_transform(self, window_lengths):  # takes real window lengths to transformed window lengths
    if self.window_function == "beta":  # the transformed parameter is a normalized "a"
        self.a_max = 1/2*((3*self.N/self.min_length)**2 -1)
        self.a_min = 1/2*((3*self.N/self.max_length)**2 -1)
        transformed_window_lengths = (
                                    (1/2*((3*self.N/window_lengths)**2 -1)
                                        -self.a_min)
                                        /(self.a_max-self.a_min)
                                    )
        return transformed_window_lengths
    else: raise NotImplementedError("window_transform is not available for Hann yet")
def inverse_window_transform(self, transformed_window_lengths):
    if self.window_function == "beta":  # the transformed parameter is a normalized "a"
        window_lengths = (
                        3*self.N /torch.sqrt(
                            1 +2*(self.a_min+(self.a_max-self.a_min)*transformed_window_lengths)
                            )
                        )
        return window_lengths
    else: raise NotImplementedError("window_transform is not available for Hann yet")
def put_windows_back(self):
    with torch.no_grad():
        self.transformed_window_lengths.clamp_(min=0, max=1)
def coefficients(self):  # get the spline coefficients
    # The Greville abscissae are the centers of the splines! Kind of, boundary effect?
    if self.window_function == 'beta':
        # Calculate the a=b parameter of the Beta distribution (46% of GPU computation, 0.00018 s)
        a = self.a_min + (self.a_max-self.a_min)*self.transformed_window_lengths.unsqueeze(1)
        #?
        # (F, T)

        # Raise the base to the exponent a (29% of GPU computation, 0.00011)
        log_pdf_ish = (a - 1) * self.log_x_1mx
        #log_pdf = log_pdf_ish -log_pdf_ish.logsumexp(dim=1, keepdim=True)
        #pdf = torch.exp(log_pdf)
        # (F, s, T)
        #return pdf

        # Normalize the coefficients to 1 and exponentiate back (24% of GPU computation, 0.00009)
        return torch.exp(log_pdf_ish -log_pdf_ish.logsumexp(dim=1, keepdim=True))  # F.softmax(log_pdf_ish, dim=1)
    elif self.window_function == 'hann':
        half_width = (lambda_/(2 * self.N)).unsqueeze(1)
        # (F, 1, T) for high_memory forward

        normalized_offsets = normalized_offsets.unsqueeze(0)
        if len(half_width.shape)==3:  # if a has all time steps, we need to add a dimension
            normalized_offsets = normalized_offsets.unsqueeze(2)
        # (1, s, 1) for high_memory forward

        # normalized offsets is already centered
        out = 0.5 * (1 + torch.cos(torch.pi * normalized_offsets / half_width))
        out[torch.abs(normalized_offsets) > half_width] = 0
        out = out / out.sum(dim=1, keepdim=True)
        #mask = torch.abs(normalized_offsets) <= half_width
        #out = torch.zeros((self.F, self.s, self.T), dtype=torch.float32)
        #double_normalized_offsets = normalized_offsets / half_width
        #out[mask] = 0.5 * (1 + torch.cos(torch.pi * double_normalized_offsets[mask]))
        #out = out / out.sum(dim=1, keepdim=True)
        # (F, s, T) for high_memory forward

        return out
    else:
        raise NotImplementedError(f"Window function '{self.window_function}' not implemented.")

#############################################################
### Modified ADSTFT with normalization, used as reference ###
#############################################################
class ADSTFT2(nn.Module):
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
        return self.win_length  # self.window_transform(self.win_length)

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
                # hann
                self.tap_win = 0.5 - 0.5 * torch.cos(2 * pi * (base + (self.actual_win_length-self.N+1)/2) / self.actual_win_length )                
                mask1 = base.ge(torch.ceil( (self.N-1+self.actual_win_length)/2))
                mask2 = base.le(torch.floor((self.N-1-self.actual_win_length)/2))            
                self.tap_win[mask1] = 0
                self.tap_win[mask2] = 0
                self.tap_win = self.tap_win / self.tap_win.sum(dim=0, keepdim=True)
                # beta
                #a = 1/2*((3*self.N/self.actual_win_length)**2 -1).unsqueeze(0)
                #log_pdf_ish = (a - 1) * (((0.5+base)/self.N)*(1-(0.5+base)/self.N)).log()
                #self.tap_win = torch.exp(log_pdf_ish -log_pdf_ish.logsumexp(dim=0, keepdim=True))
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

    def put_windows_back(self):
        minimum = self.win_min
        maximum = self.win_max
        with torch.no_grad():
            self.win_length.clamp_(min=minimum, max=maximum)

######################################
### ADSTFT from the original paper ###
######################################
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

##################################
### Code from the current repo ###  lacks window normalization, breaking the optimization
##################################
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
            if direction == "forward":
                self.tap_win = 0.5 - 0.5 * torch.cos(
                    2
                    * pi
                    * (base + (self.actual_win_length - self.N + 1) / 2)
                    / self.actual_win_length,
                )
                # mask1 = base.ge(torch.ceil( (self.N-1+self.actual_win_length)/2))
                # mask2 = base.le(torch.floor((self.N-1-self.actual_win_length)/2))
                self.tap_win[mask1] = 0
                self.tap_win[mask2] = 0
                # self.tap_win = self.tap_win.pow(self.actual_pow)
                self.tap_win = self.tap_win / self.N * 2
                return self.tap_win

            elif direction == "backward":
                # f = torch.sin(2 * pi * (base - (self.N-1)/2) /
                #             self.actual_win_length)
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

######################################## the discrepancy between ADSTFTfix and the original method is
### Fixed code from the current repo ### just that we calculate over F instead of N. then, when summing,
######################################## machine precision causes a slightly different result
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
            win_length_size = (self.F, self.T)  # CHANGE F->N to reproduce original paper
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
            end=self.F,  # CHANGE F->N to reproduce original paper
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
        #coeff = coeff[:, None] @ coeff[None, :] 
        coeff = coeff[: self.F, None] @ coeff[None, :]  # CHANGE to above to reproduce original paper
        coeff = torch.exp(-2j * pi * coeff / self.N)  # N, N
        
        # Perform the STFT
        coeff = coeff[None, None, :, :]  # 1, 1, N, N
        stft = (self.tapered_x * coeff).sum(dim=-1)

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
            )[:, None, None].expand([-1, self.F, self.T])  # CHANGE F->N to reproduce original paper
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

##################################################################################
### Experiments with energy normalization and output instead of absolute value ### 
################################################################################## 
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
