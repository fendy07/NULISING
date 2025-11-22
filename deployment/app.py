from flask import Flask, request, jsonify
from PIL import Image
import io, base64, os, time
import torch
import torch.nn as nn
from torchvision import models, transforms

app = Flask(__name__)

# Base folder to save images
BASE_SAVE_FOLDER = 'static/rawdata'
os.makedirs(BASE_SAVE_FOLDER, exist_ok=True)

# Labels (model output)
labels = ['a', 'ba', 'ca', 'da', 'ga', 'ha', 'ja', 'ka', 'la', 'ma', 'mba',
          'mpa', 'na', 'nca', 'nda', 'nga', 'ngka', 'nja', 'pa', 'ra', 'rha-gha',
          'sa', 'ta', 'wa', 'ya']

# Fixed input size
IMG_SIZE = (50, 50)  # resize all images to 50x50

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, len(labels))
model.load_state_dict(torch.load('../models/komering_character_resnet18.pth', map_location=device))
model.to(device)
model.eval()

# Transform for model input
transform = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])

# Predict function
def predict_image(img):
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_t)
        probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        pred_idx = probs.argmax()
    return labels[pred_idx], float(probs[pred_idx])

# REST API endpoint
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        if 'image' not in data or 'label' not in data:
            return jsonify({'message': 'Missing image or label'}), 400

        chosen_label = data['label']
        img_data = data['image'].split(',')[1]
        img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert('RGB')

        # Ensure white background and resize
        white_bg = Image.new("RGB", IMG_SIZE, (255, 255, 255))
        img_resized = img.resize(IMG_SIZE)
        white_bg.paste(img_resized, mask=img_resized.split()[3] if img_resized.mode == 'RGBA' else None)
        img = white_bg

        # Predict
        pred_label, score = predict_image(img)

        # Save in chosen label folder
        save_folder = os.path.join(BASE_SAVE_FOLDER, chosen_label)
        os.makedirs(save_folder, exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = f"{chosen_label}_{timestamp}.png"
        img.save(os.path.join(save_folder, filename))

        # Message based on score
        if score >= 0.85:
            msg = "High confidence prediction"
        elif score >= 0.60:
            msg = "Moderate confidence prediction"
        else:
            msg = "Low confidence prediction"

        return jsonify({
            'label': chosen_label,
            'score': round(score, 3),
            'message': msg
        })

    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
