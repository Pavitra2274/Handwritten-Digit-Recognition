import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps
import io
import base64
import re

app = Flask(__name__)

# Define CNN architecture
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model = CNNModel()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=torch.device("cpu")))
model.eval()

# Image transform pipeline
transform = transforms.Compose([
    transforms.Grayscale(),             # Ensure grayscale
    transforms.Resize((28, 28)),        # Resize to MNIST format
    transforms.ToTensor(),              # Convert to tensor (0-1)
    # Uncomment the following line if normalization is needed
    # transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
])

# Helper: convert base64 to PIL image and invert
def base64_to_pil(img_base64):
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    byte_data = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(byte_data)).convert('L')  # Convert to grayscale
   # image = ImageOps.invert(image)  # Invert to match MNIST (white digit on black)
    return image

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image = base64_to_pil(data['image'])
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1).numpy().flatten()
        prediction = int(torch.argmax(outputs, dim=1).item())

    return jsonify({
        'prediction': prediction,
        'probabilities': probabilities.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
