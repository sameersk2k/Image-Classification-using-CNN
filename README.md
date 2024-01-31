# Image-Classification-using-CNN
Classifying Images -Convolutional Neural Network (CNN) Implementation with Variations and Optimizations of Alexnet

# Convolutional Neural Network (CNN) Implementation

This repository showcases the implementation of Convolutional Neural Networks (CNNs) using TensorFlow and Keras for image classification. The project explores various CNN architectures, activation functions, and optimization techniques to enhance the performance of image classification models.

## Data Preprocessing

The dataset is preprocessed using the `ImageDataGenerator` from Keras, which allows real-time data augmentation. The images are rescaled to a range of [0, 1], and the dataset is split into training and validation sets with a 70-30 split. This preprocessing ensures a diverse and robust training process.

## Visualization

Visualizations are included to provide insights into the dataset. A batch of training images is plotted using Matplotlib, offering a glimpse into the variety and complexity of the images. Additionally, RGB channel distribution scatter plots are generated to understand the distribution of pixel values across color channels.

## Base CNN Architecture (BAlexnet)

The primary CNN architecture, referred to as BAlexnet, is implemented. It consists of convolutional layers with varying filter sizes, max-pooling layers, fully connected layers, dropout layers, and softmax activation for classification. This architecture serves as the baseline for further experimentation.

## Training Loop

The training loop is implemented using a custom function, `new_train_alex`. This function iterates over a specified number of epochs, updating the model weights through backpropagation and utilizing the Adam optimizer. Training metrics such as loss and accuracy are monitored throughout the process.

## Leaky ReLU Activation

A variation of the BAlexnet model, named Alexnet1, is introduced with Leaky ReLU activation instead of traditional ReLU. Leaky ReLU can mitigate the vanishing gradient problem and enhance the learning capacity of the model.

## Stochastic Gradient Descent (SGD) Optimizer

Another variation, Alexnet2, employs the Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.0001 and momentum of 0.9. This optimizer is known for its simplicity and effectiveness in optimizing non-convex loss functions.

## Batch Normalization

An enhanced version, BAlexnet1, incorporates batch normalization after each convolutional layer. Batch normalization helps in stabilizing and accelerating the training process by normalizing the input of each layer.

## Early Stopping

The `es_train_alex` function introduces early stopping with a patience of 2. Training stops if the validation loss does not improve for a certain number of epochs, preventing overfitting and improving model generalization.

## Results and Visualization

Training and validation metrics such as loss and accuracy are monitored and visualized over epochs. The best model based on validation accuracy is saved for future use.

## Conclusion

This project provides a comprehensive exploration of CNN architectures, activation functions, and optimization techniques for image classification tasks. Users can experiment with different configurations to understand their impact on training and validation performance.

## References

- [TensorFlow Documentation: ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
- [AlexNet Paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- [Batch Normalization Paper](https://arxiv.org/abs/1502.03167)
- [Adam Optimizer Paper](https://arxiv.org/abs/1412.6980)
- [SGD Optimizer Blog](https://ruder.io/optimizing-gradient-descent/)
