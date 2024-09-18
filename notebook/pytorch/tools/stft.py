import torch
import torchaudio
import matplotlib.pyplot as plt
import h5py
# Load data from HDF5 files
with h5py.File('./SCNN-Jamendo-train.h5', 'r') as train_file:
    train_data = torch.tensor(train_file['X'][:])
    train_labels = torch.tensor(train_file['Y'][:])


# Check the shape of the data
print(train_data.shape)  # Should Output (13357, 32000, 1)

# Remove last dimension (13357, 32000)
data = train_data.squeeze(axis=-1)

# Set STFT Parameters
n_fft = 2048
hop_length = 507
win_length = n_fft

# Create a Hann window
window = torch.hann_window(win_length)

# Save spectrograms
spectrograms = []

for waveform in data:
    # Transform numpy array into torch tensor
    waveform = torch.tensor(waveform)
    
    # STFT Transform
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )
    # Calculate Spectrogram
    spectrogram = torch.abs(stft)
    
    # Store Spectrogram
    spectrograms.append(spectrogram)

# Stack all spectrograms
spectrograms = torch.stack(spectrograms)
# spectrograms = spectrograms[:,:1024, :]

# Choose the first spectrogram to visualize
spectrogram = spectrograms[1].numpy()

# Transpose the Spectrogram
spectrogram = spectrogram.T

# Plot Spectrogram
plt.figure(figsize=(10, 6))
plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar()
plt.title('Spectrogram of the first audio sample')
plt.xlabel('Time Frames')
plt.ylabel('Frequency Bins')
plt.show()
