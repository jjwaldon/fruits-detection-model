# Fruit Object Detection Model

This repository contains code to create an object detection model for fruit images using a dataset from Kaggle. The dataset can be found [here](https://www.kaggle.com/datasets/mbkinaci/fruit-images-for-object-detection).

## Setup

Before running the code, make sure you have the necessary dependencies installed. You can install them using pip and the provided requirements.txt file:

```bash
pip install -r requirements.txt
```

## Data Preparation

The dataset is organized into four directories: 'APPLE', 'BANANA', 'MIXED', and 'ORANGE'. Images are preprocessed, then resized, normalized, and categorized.

## Model Architecture

Model uses Convolutional Neural Network (CNN) with the following architecture:

1. Conv2D layer with 64 filters, (3, 3) kernel size, and ReLU activation function.
2. MaxPooling2D layer with (3, 3) pool size.
3. Conv2D layer with 64 filters, (3, 3) kernel size, and ReLU activation function.
4. MaxPooling2D layer with (3, 3) pool size.
5. Conv2D layer with 16 filters, (3, 3) kernel size, and ReLU activation function.
6. Flatten layer to prepare for dense layers.
7. Dense layer with 64 units and ReLU activation function.
8. Dense layer with 16 units and softmax activation function.


## Training and Evaluation

Data is split into training and validation sets. After training, model accuracy and loss for both training and validation is present on graphs.
![1](https://github.com/jjwaldon/fruits-detection-model/assets/147337604/61a19feb-d5a3-4f97-afef-37e1ac2910b9)
![2](https://github.com/jjwaldon/fruits-detection-model/assets/147337604/c15af0b3-05e8-4c3a-966f-e0fb6590fae8)



## Running the Code

To run the code, ensure that you have downloaded the dataset and stored it in appropriate directories. Adjust the path accordingly in the code (`PATH`). Then, execute the Python script.
