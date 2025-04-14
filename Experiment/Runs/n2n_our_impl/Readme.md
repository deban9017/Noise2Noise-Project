This is our implementation of the n2n model using unet.

- Trained using 491 butterfly images. (1st 491 images)
    - Dataset: https://www.kaggle.com/datasets/dimensi0n/imagenet-256?select=admiral
- tested on the general test set of: https://kaggle.com/datasets/e8af2f37400b0413cf04ee1cedf4e45a951ec7b007a2ca1a34c3b6c340eb0ee1

- The model is trained using 256x256 images.

- **Average PSNR over 50 successfully processed images: 27.19 dB** (Results saved to unet_iterative_psnr.csv)