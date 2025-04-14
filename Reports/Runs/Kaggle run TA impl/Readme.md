### Kaggle dataset:
https://kaggle.com/datasets/8af2f37400b0413cf04ee1cedf4e45a951ec7b007a2ca1a34c3b6c340eb0ee1 

### Kaggle notebook:
https://www.kaggle.com/code/deban9017/aiml-proj-ta-code

### This is the kaggle run of the TA's implementation.
[Took about 45 min with P100 GPU]

### Dataset description:
- training set: 500 image (256x256) 
- test set: 50 image (256x256)
- validation set: 100 image (256x256)


### Outputs generated:
- Average PSNR over 50 images: 24.26 dB [using model, checkpoint0.pth]
    - To use the model run the notebook (train part commented out) in kaggle and load given dataset. Also keep the model in the same directory as the dataset.
- Average BM3D PSNR over 50 images: 28.16 dB

