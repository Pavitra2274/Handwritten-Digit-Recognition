## Overview

Developed a Convolutional Neural Network (CNN) model trained on the MNIST dataset for handwritten digit recognition, achieving <strong>98.97%</strong> classification accuracy. The model effectively extracts spatial features through convolution and pooling layers, demonstrating strong generalization on unseen test data.

## Dataset

<b>Name:</b>MNIST Handwritten Digits<br>
<b>Size:</b> 70,000 grayscale images (60,000 training + 10,000 testing)<br>
<b>Image Size:</b> 28×28 pixels<br>
<b>Classes:</b> Digits from 0 to 9<br>

## Tech Stack

<b>Language:</b> Python<br>
<b>Framework:</b> PyTorch (Deep Learning), Flask (Web Backend)<br>
<b>Frontend:</b> HTML, CSS, JavaScript (Canvas API, Chart.js)<br>
<b>Libraries:</b> torch, torchvision – model training & inference<br>
&nbsp;&nbsp;&nbsp;&nbsp;PIL – image processing<br>
&nbsp;&nbsp;&nbsp;&nbsp;flask – serving predictions<br>
&nbsp;&nbsp;&nbsp;&nbsp;chart.js – probability visualization in browser<br>
Environment: Runs locally in browser for digit drawing and prediction

## CNN Model Architecture

<div align="center">
  <img src="images/cnn_architecture_diagram.png"
       alt="CNN Architecture Diagram (Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → Flatten → FC → FC)"
       width="700" height="300">
  <p><em>CNN: 1×28×28 → Conv(32) → ReLU → MaxPool → Conv(64) → ReLU → MaxPool → Flatten(3136) → FC(128) → FC(10)</em></p>
</div>

## How to Run Locally

### 1️⃣ Clone the repository

    git clone https://github.com/Pavitra2274/Handwritten-Digit-Recognition.git
    cd Handwritten-Digit-Recognition

### 2️⃣ Install dependencies

    pip install torch torchvision flask pillow

### 3️⃣ Make sure trained model file exists

    (Should be named mnist_cnn.pth in the project root)

### 4️⃣ Run Flask app

    python app.py

### 5️⃣ Open browser and visit

    http://127.0.0.1:5000/

### Usage:

Draw a digit (0–9) on the black canvas.

Click Predict to see the predicted digit and class probabilities as a bar chart.

Click Clear to reset the canvas.
