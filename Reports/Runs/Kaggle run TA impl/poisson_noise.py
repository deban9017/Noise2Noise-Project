import os
import cv2
import numpy as np
import bm3d
import csv
from tqdm import tqdm
import warnings

# --- Configuration ---
INPUT_FOLDER = '../../../test_images_50'
OUTPUT_CSV = 'poisson_bm3d_vst_color.csv'
GAUSS_SIGMA_EQUIV = 4.8
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

# --- Helper Functions ---

def anscombe_forward(x):
    x_flt = x.astype(np.float64)
    return 2.0 * np.sqrt(np.maximum(0, x_flt + 3.0/8.0))

def anscombe_inverse(y):
    y_flt = y.astype(np.float64)
    base = np.maximum(0, y_flt / 2.0)
    return base**2 - 3.0/8.0

def psnr(img1, img2):
    return cv2.PSNR(img1, img2)

def add_poisson_noise_with_target_psnr(img_clean, target_psnr=20.5, tolerance=0.2, max_iters=20):
    low, high = 0.01, 1.0
    best_scaling = None
    best_psnr = -np.inf

    for _ in range(max_iters):
        mid = (low + high) / 2
        scaled_img = img_clean.astype(np.float64) * mid
        noisy = np.random.poisson(np.maximum(scaled_img, 0))
        noisy_img = (noisy / mid).clip(0, 255).astype(np.uint8)
        current_psnr = psnr(img_clean, noisy_img)

        if abs(current_psnr - target_psnr) < tolerance:
            return noisy_img.astype(np.float64), mid, current_psnr

        if current_psnr > target_psnr:
            high = mid
        else:
            low = mid

        if abs(current_psnr - target_psnr) < abs(best_psnr - target_psnr):
            best_psnr = current_psnr
            best_scaling = mid
            best_noisy = noisy_img

    return best_noisy.astype(np.float64), best_scaling, best_psnr

def denoise_color_poisson_bm3d(image_noisy_float):
    img_noisy_ansc = anscombe_forward(image_noisy_float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        denoised_img_ansc = bm3d.bm3d(img_noisy_ansc, sigma_psd=1.0)
    img_denoised_float = anscombe_inverse(denoised_img_ansc)
    return np.clip(img_denoised_float, 0, 255)

# --- Main ---
results = []
denoised_psnr_list = []

if not os.path.isdir(INPUT_FOLDER):
    print(f"Error: Input folder '{INPUT_FOLDER}' not found.")
    exit()

print(f"Processing images from: {INPUT_FOLDER}")
print(f"Saving results to: {OUTPUT_CSV}")

try:
    image_files = [f for f in os.listdir(INPUT_FOLDER)
                   if os.path.isfile(os.path.join(INPUT_FOLDER, f)) and f.lower().endswith(VALID_EXTENSIONS)]
    if not image_files:
        print(f"Error: No valid image files found in '{INPUT_FOLDER}'.")
        exit()
except OSError as e:
    print(f"Error accessing input folder: {e}")
    exit()

for filename in tqdm(image_files, desc="Processing Images"):
    try:
        img_path = os.path.join(INPUT_FOLDER, filename)
        img_clean_bgr_uint8 = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_clean_bgr_uint8 is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            results.append([filename, 'Read Error', 'Read Error', 'Read Error'])
            continue

        # --- FIXED HERE ---
        img_noisy_float, scaling_factor, psnr_noisy = add_poisson_noise_with_target_psnr(img_clean_bgr_uint8)
        img_noisy_uint8 = img_noisy_float.astype(np.uint8)

        img_denoised_float = denoise_color_poisson_bm3d(img_noisy_float)
        img_denoised_uint8 = img_denoised_float.astype(np.uint8)
        psnr_denoised = cv2.PSNR(img_clean_bgr_uint8, img_denoised_uint8)

        denoised_psnr_list.append(psnr_denoised)
        results.append([filename, f"{scaling_factor:.4f}", f"{psnr_noisy:.4f}", f"{psnr_denoised:.4f}"])

    except Exception as e:
        print(f"\nError processing {filename}: {e}. Skipping.")
        results.append([filename, 'Processing Error', 'Processing Error', 'Processing Error'])

# --- Save CSV ---
try:
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image', 'Poisson_Scaling_Factor', 'PSNR_Noisy_Poisson', 'PSNR_BM3D_Denoised'])
        writer.writerows(results)
    print(f"\nResults successfully saved to {OUTPUT_CSV}")
except IOError as e:
    print(f"\nError writing CSV file: {e}")

# --- Average PSNR ---
if denoised_psnr_list:
    avg_psnr = sum(denoised_psnr_list) / len(denoised_psnr_list)
    print(f"\nAverage PSNR of BM3D-Denoised images: {avg_psnr:.4f} dB")
else:
    print("\nNo images were successfully processed to calculate average PSNR.")

print("Batch processing complete.")







# #===========VISUALIZATION===================

# import os
# import cv2
# import numpy as np
# import bm3d
# import matplotlib.pyplot as plt
# import warnings

# # --- Configuration ---
# IMAGE_PATH = '../../../test_images/491.jpg'  # Update this path if needed
# TARGET_PSNR = 20.5
# TOLERANCE = 0.2
# MAX_ITERS = 20

# # --- Helper Functions ---
# def anscombe_forward(x):
#     x_flt = x.astype(np.float64)
#     return 2.0 * np.sqrt(np.maximum(0, x_flt + 3.0 / 8.0))

# def anscombe_inverse(y):
#     y_flt = y.astype(np.float64)
#     base = np.maximum(0, y_flt / 2.0)
#     return base ** 2 - 3.0 / 8.0

# def psnr(img1, img2):
#     return cv2.PSNR(img1, img2)

# def add_poisson_noise(image_uint8, scaling_factor):
#     image_float = image_uint8.astype(np.float64)
#     lambda_ = np.maximum(0, image_float * scaling_factor)
#     poisson_counts = np.random.poisson(lambda_)
#     if scaling_factor > 1e-9:
#         noisy_img_float = poisson_counts / scaling_factor
#     else:
#         noisy_img_float = poisson_counts
#     return np.clip(noisy_img_float, 0, 255)

# def add_poisson_noise_with_target_psnr(img_clean, target_psnr=20.5, tolerance=0.2, max_iters=20):
#     low, high = 0.01, 1.0
#     best_scaling = None
#     best_psnr = -np.inf
#     h, w, c = img_clean.shape

#     for _ in range(max_iters):
#         mid = (low + high) / 2
#         noisy_img = add_poisson_noise(img_clean, mid).astype(np.uint8)
#         current_psnr = psnr(img_clean, noisy_img)

#         if abs(current_psnr - target_psnr) < tolerance:
#             return noisy_img, mid, current_psnr

#         if current_psnr > target_psnr:
#             high = mid
#         else:
#             low = mid

#         if abs(current_psnr - target_psnr) < abs(best_psnr - target_psnr):
#             best_psnr = current_psnr
#             best_scaling = mid
#             best_noisy = noisy_img

#     return best_noisy, best_scaling, best_psnr

# def denoise_color_poisson_bm3d(image_noisy_float):
#     img_noisy_ansc = anscombe_forward(image_noisy_float)
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", category=RuntimeWarning)
#         denoised_img_ansc = bm3d.bm3d(img_noisy_ansc, sigma_psd=1.0)
#     img_denoised_float = anscombe_inverse(denoised_img_ansc)
#     return np.clip(img_denoised_float, 0, 255)

# # --- Load Image ---
# img_clean = cv2.imread(IMAGE_PATH)
# if img_clean is None:
#     raise FileNotFoundError(f"Could not read image from {IMAGE_PATH}")
# img_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)

# # --- Add Poisson Noise (adaptive scaling factor to match PSNR) ---
# img_noisy, scaling_factor, psnr_noisy = add_poisson_noise_with_target_psnr(
#     img_clean, target_psnr=TARGET_PSNR, tolerance=TOLERANCE, max_iters=MAX_ITERS)

# # --- Denoise ---
# img_denoised = denoise_color_poisson_bm3d(img_noisy.astype(np.float64)).astype(np.uint8)

# # --- Compute PSNR ---
# psnr_denoised = psnr(img_clean, img_denoised)

# # --- Plot ---
# fig, axs = plt.subplots(1, 3, figsize=(18, 6))
# axs[0].imshow(img_clean)
# axs[0].set_title("Clean Image\nPSNR: ∞")
# axs[0].axis('off')

# axs[1].imshow(img_noisy)
# axs[1].set_title(f"Noisy Image\nPSNR: {psnr_noisy:.2f} dB")
# axs[1].axis('off')

# axs[2].imshow(img_denoised)
# axs[2].set_title(f"Denoised Image\nPSNR: {psnr_denoised:.2f} dB")
# axs[2].axis('off')

# plt.tight_layout()
# plt.savefig('poisson_ansc_sample_output.png', dpi=300, bbox_inches='tight')
# plt.show()

# print(f"Final PSNR (Noisy): {psnr_noisy:.4f} dB | PSNR (Denoised): {psnr_denoised:.4f} dB")
# print(f"Used Poisson Scaling Factor: {scaling_factor:.6f}")
