# Intel Image Classification using DenseNet169

## Project Overview

This project aims to classify images from the Intel Image Classification dataset using a DenseNet169 model. Two approaches are compared: 
1. Training the DenseNet169 model from scratch.
2. Fine-tuning the DenseNet169 model pre-trained on the ImageNet dataset.

By comparing these approaches, the goal is to analyze the impact of pre-trained weights on model performance.

## Dataset

The Intel Image Classification dataset consists of 25,000 images classified into six categories: 
- Buildings
- Forests
- Glaciers
- Mountains
- Seas
- Streets

Each image is 150x150 pixels.

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

### Dataset Structure
- **Training set**: 14,034 images
- **Validation set**: 3,000 images
- **Test set**: 7,301 images

## Models

### DenseNet169
DenseNet (Dense Convolutional Network) is used as the backbone of the classifier. DenseNet169 is particularly effective in image classification due to its dense connectivity patterns. In this project, DenseNet169 was implemented in two ways:
1. **Model 1**: Training from scratch, initialized with random weights.
2. **Model 2**: Fine-tuning using pre-trained weights from the ImageNet dataset.

### Image Augmentation
Image augmentation techniques such as random rotation, mirroring, cropping, and adding noise were applied to improve the model's robustness and generalization capability.

## Implementation

The model is implemented using TensorFlow and Keras. Here's a brief breakdown of the key elements:
- **Callback functions**: For early stopping and reducing learning rate on plateaus.
- **Image Augmentation**: Implemented using Keras' `ImageDataGenerator`.
- **Training**: The model is trained for a few epochs with both random initialization and pre-trained weights.

### Dependencies

Ensure you have the following libraries installed:
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Seaborn

### Training

Run the training for both models using the Jupyter notebook provided. Model 1 is trained from scratch, while Model 2 uses ImageNet pre-trained weights. The training configuration includes:
- **Loss function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

Training results, including accuracy and loss, are plotted and compared between the two models.

### Evaluation

Evaluate the models on the test set to compare the accuracy of the two approaches:
1. **Model 1**: Trained from scratch.
2. **Model 2**: Fine-tuned from pre-trained weights.

You will observe that Model 2, using pre-trained ImageNet weights, achieves faster convergence and better generalization than the model trained from scratch.

## Results

The project compares the accuracies and losses for both training and validation sets. The models are expected to achieve the following:
- **Model 1**: Lower initial accuracy with slower convergence.
- **Model 2**: Higher accuracy due to transfer learning from ImageNet.

## References

- Intel Image Classification dataset: [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- Wu, X., Yang, H., Liu, R., & Chen, Z. (2020). *An Xception Based Convolutional Neural Network for Scene Image Classification with Transfer Learning*. 2nd International Conference on Information Technology and Computer Application (ITCA).
- Mohammad Rahimzadeh, Soroush Parvin, Amirali Askari, Elnaz Safi (2024).*WISE-SRNET: A NOVEL ARCHITECTURE FOR ENHANCING IMAGE CLASSIFICATION BY LEARNING SPATIAL RESOLUTION OF FEATURE MAPS*
- Arshleen Kaur, Vinay Kukreja, Nitin Thapliyal, Manisha Aeri, Rishabh Sharma, Shanmugasundaram Hariharan(2024).*Fine-tuned EfficientNet and MobileNetV2 Models for Intel Images Classification*. 3rd International Conference for Innovation in Technology (INOCON) Karnataka, India.

## Future Work

- Experiment with more epochs to improve accuracy.
- Try different pre-trained models such as Xception, InceptionV3, and ResNet.
- Use advanced data augmentation techniques and regularization to prevent overfitting.
