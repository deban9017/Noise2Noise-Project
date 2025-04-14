# -*- coding: utf-8 -*-
"""
ZS-N2N_Batch_ImageFiles.ipynb

Modified version to process multiple standard image files (JPG, PNG) from a directory.
"""
# ========================================
# Imports
# ========================================
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os
import pandas as pd # For saving results to CSV

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Use torchvision for loading standard image formats
try:
    from torchvision.io import read_image
    from torchvision.transforms.functional import convert_image_dtype
except ImportError:
    print("Torchvision not found. Please install it: pip install torchvision")
    exit()
# ========================================

# ### **Zero-Shot Noise2Noise: Efficient Image Denoising without any Data**
# ... (rest of the original description remains the same) ...

# ========================================
# Configuration
# ========================================

#Enter device here, 'cuda' for GPU, and 'cpu' for CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Directory containing the 50 test images (JPG, PNG, etc.)
# !!! IMPORTANT: Make sure this directory exists and contains your test images !!!
test_image_dir = 'test_images_50' # <--- CHANGE THIS if your directory name is different

# Noise parameters (same as original script)
noise_type = 'gauss' # Either 'gauss' or 'poiss'
noise_level = 25     # Pixel range is 0-255 for Gaussian, and 0-1 for Poission

# Training hyperparameters (same as original script)
max_epoch = 3000     # training epochs
lr = 0.001           # learning rate
step_size = 1000     # number of epochs at which learning rate decays
gamma = 0.5          # factor by which learning rate decays
chan_embed = 48      # Network channel embedding size (adjust if needed based on image size)



# ========================================
# Helper Functions (mostly unchanged)
# ========================================

def add_noise(x, noise_level, noise_type='gauss'):
    """Adds Gaussian or Poisson noise to an image tensor (expects input in 0-1 range)."""
    if noise_type == 'gauss':
        # Ensure noise tensor is created on the same device as x
        noise = torch.normal(0, noise_level/255.0, x.shape, device=x.device)
        noisy = x + noise
        noisy = torch.clamp(noisy, 0, 1)
    elif noise_type == 'poiss':
        # Ensure input is non-negative for Poisson
        x_non_neg = torch.clamp(x, min=0.0)
        # Ensure noise tensor is created on the same device as x
        noisy = torch.poisson(noise_level * x_non_neg) / noise_level
        noisy = torch.clamp(noisy, 0, 1) # Clamp result as well
    else:
        raise ValueError("noise_type must be 'gauss' or 'poiss'")
    return noisy

class network(nn.Module):
    """Simple 2-layer CNN for denoising."""
    def __init__(self, n_chan, chan_embed=48):
        super(network, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # Ensure layers are created on the correct device implicitly by model.to(device)
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
    #img has shape B C H W
    c = img.shape[1]
    # Ensure filters are created on the correct device
    filter1 = torch.FloatTensor([[[[0 ,0.5],[0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)
    filter2 = torch.FloatTensor([[[[0.5 ,0],[0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)
    return output1, output2

def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Calculates Mean Squared Error."""
    # Ensure tensors are on the same device and type
    gt = gt.to(pred.device, dtype=pred.dtype)
    loss = torch.nn.MSELoss()
    return loss(gt, pred)

# Global variable to hold the current model being trained for loss calculation
current_model = None

def loss_func(noisy_img):
    """Calculates the ZS-N2N loss (Residual + Consistency)."""
    global current_model # Access the model being trained for the current image
    if current_model is None:
        raise ValueError("Model not set for loss calculation")

    noisy1, noisy2 = pair_downsampler(noisy_img)

    # Ensure model inputs are on the correct device
    noisy1, noisy2 = noisy1.to(device), noisy2.to(device)

    pred1 = noisy1 - current_model(noisy1)
    pred2 = noisy2 - current_model(noisy2)

    loss_res = 0.5 * (mse(noisy1, pred2) + mse(noisy2, pred1))

    # Ensure model input is on the correct device
    noisy_img_dev = noisy_img.to(device)
    noisy_denoised = noisy_img_dev - current_model(noisy_img_dev)
    denoised1, denoised2 = pair_downsampler(noisy_denoised)

    loss_cons = 0.5 * (mse(pred1, denoised1) + mse(pred2, denoised2))

    loss = loss_res + loss_cons
    return loss

def train(model, optimizer, noisy_img):
    """Performs one training step."""
    global current_model
    current_model = model # Set the global model for loss_func

    model.train() # Set model to training mode
    # Ensure data is on the correct device before passing to loss_func
    noisy_img = noisy_img.to(device)
    loss = loss_func(noisy_img)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    current_model = None # Unset the global model
    return loss.item()

def test(model, noisy_img, clean_img):
    """Calculates PSNR of the denoised image."""
    global current_model
    current_model = model # Set the global model for denoise function

    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        # Ensure data is on the correct device before passing to denoise
        noisy_img_dev = noisy_img.to(device)
        pred = denoise(model, noisy_img_dev) # Use the denoise function

        # Ensure clean_img is on the same device and type as pred for mse
        clean_img_dev = clean_img.to(pred.device, dtype=pred.dtype)
        MSE = mse(clean_img_dev, pred).item()

        # Handle potential division by zero or negative MSE
        if MSE <= 0:
             print(f"Warning: Non-positive MSE ({MSE}) encountered. PSNR calculation may be invalid.")
             PSNR = -np.inf # Or some other indicator of invalidity
        else:
             # PSNR assumes max signal value is 1.0 since we normalized images to 0-1
             PSNR = 10 * np.log10(1.0 / MSE)

    current_model = None # Unset the global model
    return PSNR

def denoise(model, noisy_img):
    """Applies the trained model to denoise an image."""
    global current_model
    current_model = model # Set the global model

    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        # Ensure data is on the correct device before passing to the model
        noisy_img_dev = noisy_img.to(device)
        # The denoised image is noisy_img - predicted_noise
        predicted_noise = model(noisy_img_dev)
        pred = torch.clamp(noisy_img_dev - predicted_noise, 0, 1)

    current_model = None # Unset the global model
    return pred

# ========================================
# Main Processing Loop
# ========================================

# Check if the test image directory exists
if not os.path.isdir(test_image_dir):
    print(f"Error: Directory not found: {test_image_dir}")
    print("Please ensure the directory exists and contains your JPG, PNG, etc. image files.")
else:
    # Get list of image files with common image extensions (case-insensitive)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    try:
        all_files = os.listdir(test_image_dir)
        image_files = sorted([f for f in all_files if f.lower().endswith(image_extensions)])
    except FileNotFoundError:
        print(f"Error: Cannot access directory: {test_image_dir}")
        image_files = [] # Prevent further errors

    if not image_files:
        print(f"No image files ({', '.join(image_extensions)}) found in {test_image_dir}.")
    else:
        print(f"Found {len(image_files)} images to process.")

        psnr_results = []
        image_filenames = []

        # Loop through each image file
        for filename in tqdm(image_files, desc="Processing Images"):
            image_path = os.path.join(test_image_dir, filename)
            print(f"\n--- Processing: {filename} ---")

            try:
                # --- Load and Preprocess Image ---
                # Read image directly into a CHW tensor (uint8, 0-255)
                img_tensor_uint8 = read_image(image_path)

                # Normalize to 0-1 float32
                clean_img_chw_float = convert_image_dtype(img_tensor_uint8, dtype=torch.float32)

                # Add batch dimension (BCHW)
                clean_img = clean_img_chw_float.unsqueeze(0)

                # Move to the designated device
                clean_img = clean_img.to(device)
                print(f"Loaded clean image shape: {clean_img.shape}") # BCHW
                # --- End Load and Preprocess ---

                # Add noise (already ensures output is on the same device as clean_img)
                noisy_img = add_noise(clean_img, noise_level, noise_type)

                # --- Re-initialize model and optimizer for each image ---
                # Determine number of channels from the loaded image
                n_chan = clean_img.shape[1]
                if n_chan not in [1, 3]: # Handle grayscale (1) or RGB (3)
                    print(f"Warning: Unexpected number of channels ({n_chan}) for {filename}. Assuming 3 channels.")
                    # If grayscale, replicate channel if model expects 3. Or adjust model.
                    # For simplicity here, we'll proceed but ZS-N2N might need adjustments for grayscale.
                    # If the image is grayscale (C=1) and loaded as such, ensure the network `n_chan` matches.
                    pass # Keep n_chan as loaded

                model = network(n_chan, chan_embed=chan_embed)
                model = model.to(device) # Move model to device
                # print(f"Network parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

                optimizer = optim.Adam(model.parameters(), lr=lr)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
                # --- End Re-initialization ---

                # Start denoising (training loop for the current image)
                # Ensure noisy_img is passed to train (which will move it to device if needed)
                for epoch in tqdm(range(max_epoch), desc=f"Training {filename[:15]}...", leave=False):
                    train(model, optimizer, noisy_img) # noisy_img is already on device from add_noise
                    scheduler.step()

                # Test (calculate PSNR) for the current image
                # Pass the original clean_img (on device) and noisy_img (on device)
                current_psnr = test(model, noisy_img, clean_img)
                print(f"Finished processing {filename}. PSNR: {current_psnr:.2f} dB")

                # Store results
                psnr_results.append(current_psnr)
                image_filenames.append(filename)

                # Optional: Display or save comparison images (uncomment if needed)
                # Make sure to handle potential channel differences if displaying grayscale
                # with torch.no_grad():
                #      denoised_img = denoise(model, noisy_img) # Get denoised image
                # clean_cpu = clean_img.cpu().squeeze(0).permute(1, 2, 0).numpy()
                # noisy_cpu = noisy_img.cpu().squeeze(0).permute(1, 2, 0).numpy()
                # denoised_cpu = denoised_img.cpu().squeeze(0).permute(1, 2, 0).numpy()
                #
                # # Adjust for grayscale display if necessary
                # if clean_cpu.shape[2] == 1:
                #     clean_cpu = clean_cpu.squeeze(axis=2)
                #     noisy_cpu = noisy_cpu.squeeze(axis=2)
                #     denoised_cpu = denoised_cpu.squeeze(axis=2)
                #     cmap = 'gray'
                # else:
                #     cmap = None # Default for RGB
                #
                # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                # ax[0].imshow(np.clip(clean_cpu, 0, 1), cmap=cmap)
                # ax[0].set_title('Ground Truth')
                # ax[0].axis('off')
                # ax[1].imshow(np.clip(noisy_cpu, 0, 1), cmap=cmap)
                # ax[1].set_title('Noisy')
                # ax[1].axis('off')
                # ax[2].imshow(np.clip(denoised_cpu, 0, 1), cmap=cmap)
                # ax[2].set_title(f'Denoised (PSNR: {current_psnr:.2f} dB)')
                # ax[2].axis('off')
                # plt.suptitle(filename)
                # # plt.savefig(f"output_{filename}.png") # Example of saving
                # plt.show()
                # plt.close(fig) # Close the figure to save memory

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                # Optionally store NaN or skip the file
                # psnr_results.append(np.nan)
                # image_filenames.append(filename)
                continue # Move to the next image

        # ========================================
        # Final Results
        # ========================================

        if psnr_results: # Check if any images were processed successfully
            # Calculate average PSNR
            # Handle potential non-finite values (like -inf from zero MSE) before averaging
            valid_psnr_results = [p for p in psnr_results if np.isfinite(p)]
            if valid_psnr_results:
                 avg_psnr = np.mean(valid_psnr_results)
                 print(f"\n========================================")
                 print(f"Average PSNR over {len(valid_psnr_results)} successfully processed images: {avg_psnr:.2f} dB")
                 print(f"========================================")
            else:
                 print("\nNo valid PSNR values obtained.")
                 avg_psnr = np.nan

            # Save results to CSV
            try:
                df = pd.DataFrame({
                    'filename': image_filenames,
                    'psnr_db': psnr_results
                })
                csv_filename = 'psnr_results.csv'
                df.to_csv(csv_filename, index=False)
                print(f"PSNR results saved to {csv_filename}")
            except Exception as e:
                print(f"Error saving results to CSV: {e}")
        else:
            print("\nNo images were processed successfully.")

# End of script check
print("\nScript finished.")