# -*- coding: utf-8 -*-
"""
ZS-N2N_Single_Image_Output.ipynb

Modified version to process a single image file (JPG, PNG, etc.),
and output a comparison image (Original, Noisy, Denoised) with PSNR values.
"""

# ========================================
# Imports
# ========================================
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os
import time # To potentially measure time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Use torchvision for loading standard image formats
try:
    from torchvision.io import read_image, ImageReadMode
    from torchvision.transforms.functional import convert_image_dtype
except ImportError:
    print("Torchvision not found. Please install it: pip install torchvision")
    exit()

# ========================================
# Configuration
# ========================================

# --- USER INPUT: Specify the path to your single test image ---
# Use a relative path (e.g., 'my_images/test.png') or an absolute path
input_image_path = '../../../test_images/491.jpg' # <<< CHANGE THIS TO YOUR IMAGE PATH

# --- Output file name ---
output_filename = 'zs_n2n_sample_output.png'

# --- Device Configuration ---
# Automatically select GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- Noise Parameters ---
noise_type = 'poiss' # Either 'gauss' or 'poiss'
noise_level = 25    # Sigma for Gaussian (0-255 scale), Lambda for Poisson (0-1 scale)

# --- Training Hyperparameters ---
max_epoch = 3000     # training epochs (you might reduce this for quicker testing)
lr = 0.001           # learning rate
step_size = 1000     # number of epochs at which learning rate decays
gamma = 0.5          # factor by which learning rate decays
chan_embed = 48      # Network channel embedding size

# ========================================
# Helper Functions (mostly unchanged, minor adjustments)
# ========================================
def get_poisson_lambda_for_target_psnr(img, target_psnr=20.5, device='cpu'):
    """Finds the Poisson lambda value that gives a noisy image with similar PSNR to Gaussian σ=25."""
    # candidate_lambdas = torch.logspace(-1, 1, steps=30).to(device)  # From 0.1 to 10
    candidate_lambdas = torch.logspace(-1, 2, steps=50).to(device)  # 0.1 to 100

    best_lambda = candidate_lambdas[0].item()
    best_psnr_diff = float('inf')

    for lam in candidate_lambdas:
        noisy = torch.poisson(lam * img) / lam
        noisy = torch.clamp(noisy, 0, 1)
        psnr = calculate_psnr(noisy, img)
        diff = abs(psnr - target_psnr)
        if diff < best_psnr_diff:
            best_psnr_diff = diff
            best_lambda = lam.item()
    return best_lambda


def add_noise(x, noise_level, noise_type='gauss'):
    """Adds Gaussian or Poisson noise to an image tensor (expects input in 0-1 range)."""
    if noise_type == 'gauss':
        noise = torch.normal(0, noise_level / 255.0, x.shape, device=x.device)
        noisy = x + noise
    elif noise_type == 'poiss':
        x_non_neg = torch.clamp(x, min=0.0)
        # Poisson noise level lambda is often defined relative to the max intensity (1.0 here)
        # The original code used noise_level directly, assuming it's scaled appropriately.
        # Let's keep it consistent with the original code snippet.
        noisy = torch.poisson(noise_level * x_non_neg) / noise_level
    else:
        raise ValueError("noise_type must be 'gauss' or 'poiss'")
    # Clamp the noisy image to the valid range [0, 1]
    noisy = torch.clamp(noisy, 0, 1)
    return noisy

class network(nn.Module):
    """Simple 2-layer CNN for denoising."""
    def __init__(self, n_chan, chan_embed=48):
        super(network, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x

def pair_downsampler(img):
    """Creates two downsampled images using fixed kernels."""
    c = img.shape[1]
    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)
    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)
    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)
    return output1, output2

def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Calculates Mean Squared Error."""
    gt = gt.to(pred.device, dtype=pred.dtype)
    loss = torch.nn.MSELoss()
    return loss(gt, pred)

# Model needs to be accessible in the loss function scope
current_model = None

def loss_func(noisy_img):
    """Calculates the ZS-N2N loss (Residual + Consistency)."""
    global current_model
    if current_model is None:
        raise ValueError("Model not set for loss calculation")

    noisy1, noisy2 = pair_downsampler(noisy_img)
    pred1_noise = current_model(noisy1)
    pred2_noise = current_model(noisy2)
    denoised1 = noisy1 - pred1_noise
    denoised2 = noisy2 - pred2_noise

    # Residual Loss Component (mapping one downsampled noisy to the other)
    loss_res = 0.5 * (mse(noisy1, denoised2) + mse(noisy2, denoised1))

    # Consistency Loss Component
    noisy_denoised = noisy_img - current_model(noisy_img) # Denoise the full noisy image
    denoised1_from_full, denoised2_from_full = pair_downsampler(noisy_denoised)
    loss_cons = 0.5 * (mse(denoised1, denoised1_from_full) + mse(denoised2, denoised2_from_full))

    loss = loss_res + loss_cons
    return loss

def train(model, optimizer, noisy_img_train):
    """Performs one training step."""
    global current_model
    current_model = model # Set the global model for loss_func

    model.train()
    noisy_img_train = noisy_img_train.to(device) # Ensure input is on the correct device
    loss = loss_func(noisy_img_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    current_model = None # Unset the global model
    return loss.item()

def denoise(model, noisy_img_denoise):
    """Applies the trained model to denoise an image."""
    model.eval()
    with torch.no_grad():
        noisy_img_denoise = noisy_img_denoise.to(device) # Ensure input is on the correct device
        predicted_noise = model(noisy_img_denoise)
        denoised_img = torch.clamp(noisy_img_denoise - predicted_noise, 0, 1)
    return denoised_img

def calculate_psnr(img1, img2, max_val=1.0):
    """Calculates PSNR between two images (tensors, range 0-1)."""
    img1 = img1.to(device)
    img2 = img2.to(device)
    mse_val = mse(img1, img2).item()
    if mse_val <= 1e-10: # Prevent division by zero / log(0)
        return float('inf')
    else:
        psnr = 10 * np.log10(max_val**2 / mse_val)
        return psnr

# ========================================
# Main Processing Logic for Single Image
# ========================================

if not os.path.exists(input_image_path):
    print(f"Error: Input image not found at {input_image_path}")
else:
    print(f"Processing image: {input_image_path}")
    start_time = time.time()

    try:
        # --- Load and Preprocess Image ---
        # Read image, forcing RGB (handles grayscale by converting)
        # Using ORIGINAL mode might be better if you want to handle grayscale specifically
        # img_tensor_uint8 = read_image(input_image_path, mode=ImageReadMode.RGB)
        img_tensor_uint8 = read_image(input_image_path) # Load with original channels
        if img_tensor_uint8.shape[0] == 1:
             print("Loaded image is grayscale.")
             # Option 1: Convert grayscale to RGB if network expects 3 channels
             # img_tensor_uint8 = img_tensor_uint8.repeat(3, 1, 1)
             # print("Converted grayscale to RGB.")
             # Option 2: Keep as grayscale (ensure network n_chan=1)
             pass # Keep as grayscale
        elif img_tensor_uint8.shape[0] == 4: # Handle RGBA
             print("Loaded image has alpha channel (RGBA), removing alpha.")
             img_tensor_uint8 = img_tensor_uint8[:3, :, :] # Keep only RGB


        # Normalize to 0-1 float32
        clean_img_chw_float = convert_image_dtype(img_tensor_uint8, dtype=torch.float32)
        # Add batch dimension (BCHW)
        clean_img = clean_img_chw_float.unsqueeze(0)
        # Move to the designated device
        clean_img = clean_img.to(device)
        print(f"Loaded clean image shape: {clean_img.shape}") # BCHW
        n_chan = clean_img.shape[1] # Get number of channels

        # --- Add Noise ---
        # if noise_type == 'poiss':
        #     noise_level = get_poisson_lambda_for_target_psnr(clean_img, target_psnr=20.5, device=device)
        #     print(f"Adjusted Poisson noise level (lambda) for target PSNR: {noise_level:.3f}")
        noisy_img = add_noise(clean_img, noise_level, noise_type)

        noisy_img = noisy_img.to(device) # Ensure it's on device

        # --- Initialize Model, Optimizer, Scheduler ---
        model = network(n_chan, chan_embed=chan_embed).to(device)
        print(f"Network parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        # --- Training Loop ---
        print("Starting ZS-N2N training...")
        for epoch in tqdm(range(max_epoch), desc="Training"):
            train(model, optimizer, noisy_img)
            scheduler.step()
            # Optional: Print loss or PSNR periodically during training
            # if (epoch + 1) % 500 == 0:
            #     loss_val = train(model, optimizer, noisy_img) # Rerun train just to get loss? Or store loss
            #     print(f"Epoch {epoch+1}/{max_epoch}, Loss: {loss_val:.4f}")

        training_time = time.time() - start_time
        print(f"Training finished in {training_time:.2f} seconds.")

        # --- Denoise the Image ---
        print("Denoising image...")
        denoised_img = denoise(model, noisy_img)
        denoising_time = time.time() - start_time - training_time
        print(f"Denoising finished in {denoising_time:.2f} seconds.")

        # --- Calculate PSNR Values ---
        psnr_noisy = calculate_psnr(noisy_img, clean_img)
        psnr_denoised = calculate_psnr(denoised_img, clean_img)
        print(f"PSNR (Noisy vs Clean): {psnr_noisy:.2f} dB")
        print(f"PSNR (Denoised vs Clean): {psnr_denoised:.2f} dB")

        # --- Prepare Images for Plotting ---
        # Move tensors to CPU, remove batch dim, convert CHW to HWC, convert to NumPy
        clean_np = clean_img.cpu().squeeze(0).permute(1, 2, 0).numpy()
        noisy_np = noisy_img.cpu().squeeze(0).permute(1, 2, 0).numpy()
        denoised_np = denoised_img.cpu().squeeze(0).permute(1, 2, 0).numpy()

        # Clip values to [0, 1] just in case of minor floating point issues
        clean_np = np.clip(clean_np, 0, 1)
        noisy_np = np.clip(noisy_np, 0, 1)
        denoised_np = np.clip(denoised_np, 0, 1)

        # --- Create and Save Comparison Plot ---
        print(f"Saving comparison image to {output_filename}...")
        fig, ax = plt.subplots(1, 3, figsize=(18, 6)) # Adjust figsize as needed

        # Determine cmap for grayscale if necessary
        cmap = 'gray' if clean_np.shape[2] == 1 else None
        if cmap == 'gray': # Remove channel dim for imshow if grayscale
             clean_np = clean_np.squeeze(axis=2)
             noisy_np = noisy_np.squeeze(axis=2)
             denoised_np = denoised_np.squeeze(axis=2)


        ax[0].imshow(clean_np, cmap=cmap)
        ax[0].set_title('Original (Clean)')
        ax[0].axis('off') # Hide axes ticks

        ax[1].imshow(noisy_np, cmap=cmap)
        ax[1].set_title(f'Noisy\nPSNR: {psnr_noisy:.2f} dB')
        ax[1].axis('off')

        ax[2].imshow(denoised_np, cmap=cmap)
        ax[2].set_title(f'Denoised (ZS-N2N)\nPSNR: {psnr_denoised:.2f} dB')
        ax[2].axis('off')

        plt.tight_layout() # Adjust spacing between subplots
        plt.savefig(output_filename, bbox_inches='tight', dpi=150) # Save the figure
        plt.close(fig) # Close the figure to free memory
        print("Comparison image saved successfully.")

    except FileNotFoundError:
         print(f"Error: Could not find the image file at {input_image_path}")
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging

# End of script
print("\nScript finished.")