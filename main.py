# from flask import Flask, request, jsonify
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import torch.nn.functional as F
# import os

# app = Flask(__name__)

# # Load pre-trained ResNet model
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# model.eval()

# # Define image transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Function to analyze tattoo complexity
# def analyze_tattoo(image_path):
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0)
    
#     with torch.no_grad():
#         output = model(image)
#         probabilities = F.softmax(output[0], dim=0)
    
#     complexity_score = torch.max(probabilities).item()
#     complexity_level = 'Simple' if complexity_score < 0.5 else 'Complex'
#     return complexity_level, complexity_score

# # Function to estimate price and duration
# def estimate_price_and_duration(complexity_level):
#     if complexity_level == 'Simple':
#         return {'price': 100, 'duration': '1 hour'}
#     else:
#         return {'price': 300, 'duration': '3 hours'}

# @app.route('/analyze', methods=['POST'])
# def analyze_image():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'}), 400
    
#     image = request.files['image']
#     image_path = "images.jpg"
#     image.save(image_path)
    
#     complexity_level, complexity_score = analyze_tattoo(image_path)
#     quote = estimate_price_and_duration(complexity_level)
#     os.remove(image_path)  # Cleanup uploaded image
    
#     return jsonify({
#         'complexity': complexity_level,
#         'score': round(complexity_score, 2),
#         'estimated_price': quote['price'],
#         'estimated_duration': quote['duration']
#     })

# if __name__ == '__main__':
#     app.run(debug=True)
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load the pre-trained model (ResNet18)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='IMAGENET1K_V1')
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to analyze tattoo complexity
def analyze_tattoo(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output[0], dim=0)
    
    complexity_score = torch.max(probabilities).item()
    complexity_level = 'Simple' if complexity_score < 0.5 else 'Complex'
    
    return complexity_level, complexity_score

# Pricing function
def estimate_price_and_duration(complexity_level):
    if complexity_level == 'Simple':
        return {'price': 100, 'duration': '1 hour'}
    else:
        return {'price': 300, 'duration': '3 hours'}

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    image_path = "uploaded_image.jpg"
    file.save(image_path)

    complexity_level, complexity_score = analyze_tattoo(image_path)
    quote = estimate_price_and_duration(complexity_level)

    response = {
        "complexity": complexity_level,
        "score": round(complexity_score, 2),
        "estimated_price": quote['price'],
        "estimated_duration": quote['duration']
    }
    
    os.remove(image_path)  # Delete uploaded image after processing
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
