import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from bm3d import bm3d
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

# Load the original image
try:
    original_image = Image.open("491.jpg").convert('RGB')
except FileNotFoundError:
    print("Error: '491.jpg' not found in the current directory.")
    exit()

# Convert the original image to a NumPy array (float for calculations)
original_np = np.array(original_image, dtype=np.float64) / 255.0

# Add Gaussian noise
sigma = 25
noisy_np = original_np + np.random.normal(0, sigma / 255.0, original_np.shape)
noisy_np = np.clip(noisy_np, 0, 1)

# Apply BM3D denoising
sigma_psd = sigma / 255.0  # BM3D expects sigma in the range [0, 1]
denoised_np = bm3d(noisy_np, sigma_psd=sigma_psd)
denoised_np = np.clip(denoised_np, 0, 1)

# --- Calculate PSNR Values for Display ---
# Convert NumPy arrays to PyTorch tensors for PSNR calculation
original_tensor = torch.from_numpy(original_np).permute(2, 0, 1).float().unsqueeze(0)
noisy_tensor = torch.from_numpy(noisy_np).permute(2, 0, 1).float().unsqueeze(0)
denoised_tensor = torch.from_numpy(denoised_np).permute(2, 0, 1).float().unsqueeze(0)

# For this example, we'll assume the 'device' is CPU for simplicity
device = torch.device("cpu")
original_tensor_device = original_tensor.to(device)
noisy_tensor_device = noisy_tensor.to(device)
denoised_tensor_device = denoised_tensor.to(device)

# Calculate PSNR for Noisy vs Clean
mse_noisy = F.mse_loss(noisy_tensor_device, original_tensor_device)
if mse_noisy.item() > 1e-10: # Avoid log(0)
    psnr_noisy = 10 * torch.log10(1.0 / mse_noisy).item()
else:
    psnr_noisy = float('inf')

# Calculate PSNR for Denoised vs Clean
mse_denoised = F.mse_loss(denoised_tensor_device, original_tensor_device)
if mse_denoised.item() > 1e-10: # Avoid log(0)
    psnr_denoised = 10 * torch.log10(1.0 / mse_denoised).item()
else:
    psnr_denoised = float('inf')

print(f"Displaying Images - PSNR (Noisy vs Clean): {psnr_noisy:.2f} dB")
print(f"Displaying Images - PSNR (Denoised vs Clean): {psnr_denoised:.2f} dB")

# --- Display Results ---

# Move tensors to CPU and permute C,H,W to H,W,C for matplotlib
original_display_np = original_np
noisy_display_np = noisy_np
denoised_display_np = denoised_np

fig, ax = plt.subplots(1, 3, figsize=(18, 7)) # Slightly increased height for titles

# Display Original Image
ax[0].imshow(np.clip(original_display_np, 0, 1))
ax[0].set_title('Original Image')
ax[0].axis('off')

# Display Noisy Image with PSNR
ax[1].imshow(np.clip(noisy_display_np, 0, 1))
# Format title to include PSNR value
title_noisy = f'Noisy Input Image\nPSNR: {psnr_noisy:.2f} dB'
ax[1].set_title(title_noisy)
ax[1].axis('off')

# Display Denoised Image with PSNR
ax[2].imshow(np.clip(denoised_display_np, 0, 1))
# Format title to include PSNR value
title_denoised = f'Denoised Output Image\nPSNR: {psnr_denoised:.2f} dB'
ax[2].set_title(title_denoised)
ax[2].axis('off')

plt.tight_layout(pad=1.5) # Add padding to prevent title overlap if needed
plt.savefig("comparison_bm3d_sigma_25.png")
plt.show()

print("Saved comparison image as 'comparison_bm3d_sigma_25.png'")