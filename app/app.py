from flask import Flask, render_template, request, jsonify
from PIL import Image
import io, base64, os, time
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F

app = Flask(__name__)

# Base folder to save images
BASE_SAVE_FOLDER = 'static/rawdata'
os.makedirs(BASE_SAVE_FOLDER, exist_ok=True)

# Labels
labels = ['a', 'ba', 'ca', 'da', 'ga', 'ha', 'ja', 'ka', 'la', 'ma', 'mba',
          'mpa', 'na', 'nca', 'nda', 'nga', 'ngka', 'nja', 'pa', 'ra', 'rha-gha',
          'sa', 'ta', 'wa', 'ya']

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, len(labels))
model.load_state_dict(torch.load('../models/komering_character_resnet18.pth', map_location=device))
model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([transforms.Resize((50, 50)), transforms.ToTensor()])

def predict_image(img):
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_t)
        probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        pred_idx = probs.argmax()
    return labels[pred_idx], float(probs[pred_idx])

@app.route('/')
def index():
    return render_template('index.html', labels=labels)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image'].split(',')[1]
    chosen_label = request.json['label']  # label chosen by user
    img = Image.open(io.BytesIO(base64.b64decode(data))).convert('RGB')

    # Ensure white background
    white_bg = Image.new("RGB", img.size, (255, 255, 255))
    white_bg.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
    img = white_bg

    # Predict label and confidence
    pred_label, score = predict_image(img)

    # Save in chosen label folder
    save_folder = os.path.join(BASE_SAVE_FOLDER, chosen_label)
    os.makedirs(save_folder, exist_ok=True)
    timestamp = int(time.time() * 1000)
    filename = f"{chosen_label}_{timestamp}.png"
    img.save(os.path.join(save_folder, filename))

    return jsonify({
        'prediction': pred_label,
        'score': score,
        'saved_as': os.path.join(chosen_label, filename)
    })

if __name__ == '__main__':
    app.run(debug=True)
