This is the same implementation of zero-shot noise2noise as in the paper.
Updated to test on 50 test images.

### Kaggle dataset:
https://kaggle.com/datasets/8af2f37400b0413cf04ee1cedf4e45a951ec7b007a2ca1a34c3b6c340eb0ee1 
(Take only test images)

### Details:
- Model was run using the python file. Test images are expected to be in ./test_images_50
- psnr values of output images are saved in psnr_results.csv
- notebook is the original implementation.
- zs_output_image.py takes test image, saves the output image as zs_n2n_sample_output.png


- **Average PSNR over 50 successfully processed images: 28.41 dB**