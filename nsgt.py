import torch
import numpy as np
import matplotlib.pyplot as plt
from nnAudio.Spectrogram import NSGT

# Create a toy signal
sr = 16000
t = np.linspace(0, 1, sr)
x = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

# Define linear frequency bins
n_bins = 32
f_min = 100
f_max = 4000
frequencies = np.linspace(f_min, f_max, n_bins)

# Define custom window lengths (example: linearly increase)
window_lengths = np.linspace(128, 1024, n_bins).astype(int)

# Use NSGT (you can supply custom scale and window lengths here)
nsgt = NSGT(scale_type='linear',
            fs=sr,
            fmin=f_min,
            fmax=f_max,
            bins_per_octave=n_bins,  # just needed for config
            length=x_tensor.shape[-1],
            device='cpu',
            multiresolution=True,
            gamma=0,
            slice_length=None,      # whole signal
            n_fft=4096,
            win_type='hann',
            freq_bins=frequencies.tolist(),        # linear frequencies
            window_lengths=window_lengths.tolist() # custom window lengths
)

# Compute the transform
X = nsgt(x_tensor)  # shape: [batch, freq_bins, time]

# Plot magnitude
plt.imshow(20 * np.log10(np.abs(X[0].numpy()) + 1e-6),
           aspect='auto',
           origin='lower',
           extent=[0, t[-1], f_min, f_max])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Custom NSGT Spectrogram')
plt.colorbar(label='Magnitude (dB)')
plt.show()
