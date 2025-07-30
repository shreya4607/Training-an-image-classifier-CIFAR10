# Training-an-image-classifier-CIFAR10
This repository contains a PyTorch implementation of an image classification pipeline for the CIFAR-10 dataset. The pipeline includes data preprocessing, model training, performance evaluation using key metrics, and ROC curve analysis for each class.


#Dataset
The CIFAR-10 dataset consists of 60,000 color images in 10 classes, with 6,000 images per class. The dataset is split into:
50,000 training images
10,000 test images
Each image is 32x32 pixels and belongs to one of the following classes: plane, car, bird, cat, deer, dog, frog, horse, ship, and truck.


#Workflow
1. Data Preprocessing
Images are transformed to tensors and normalized using torchvision.transforms.
Data is loaded using PyTorch's DataLoader with configurable batch size and shuffling.
2. Model Training
A convolutional neural network (CNN) is trained on the training dataset.
Training uses CrossEntropyLoss as the loss function.
Optimization is done using either SGD or Adam.
3. Evaluation Metrics
After training, the model is evaluated on the test dataset.
Metrics are calculated using scikit-learn:
Accuracy
Precision
Recall
F1-score
4. ROC and AUC
Class-wise ROC curves are plotted in a one-vs-rest manner.
AUC (Area Under Curve) is calculated for each class.
Curves are plotted using matplotlib.

