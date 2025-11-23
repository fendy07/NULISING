from flask import Flask, request, jsonify
from PIL import Image
import io, base64, os, time
import numpy as np
import onnxruntime as ort

app = Flask(__name__)

# Base folder to save images
BASE_SAVE_FOLDER = 'static/rawdata'
os.makedirs(BASE_SAVE_FOLDER, exist_ok=True)

# Labels
labels = ['a', 'ba', 'ca', 'da', 'ga', 'ha', 'ja', 'ka', 'la', 'ma', 'mba',
          'mpa', 'na', 'nca', 'nda', 'nga', 'ngka', 'nja', 'pa', 'ra', 'rha-gha',
          'sa', 'ta', 'wa', 'ya']

# Load ONNX model
onnx_model_path = "../komering_character_resnet18.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# Image transform
def transform_image(img):
    img = img.resize((50, 50))
    img = np.array(img).astype(np.float32) / 255.0  # normalize 0-1
    if img.shape[-1] == 4:  # RGBA -> RGB
        img = img[..., :3]
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # add batch dim
    return img

def predict_image(img):
    img_t = transform_image(img)
    ort_inputs = {ort_session.get_inputs()[0].name: img_t}
    ort_outs = ort_session.run(None, ort_inputs)
    # softmax
    logits = ort_outs[0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    pred_idx = np.argmax(probs)
    return labels[pred_idx], float(probs[0][pred_idx])

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
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

        # Return JSON with score in message
        return jsonify({
            "label": pred_label,
            "message": f"Predicted '{pred_label}' with confidence {score:.2f}",
            "score": score
        })

    except Exception as e:
        return jsonify({
            "label": "",
            "message": f"Error: {str(e)}",
            "score": 0.0
        }), 400

@app.route('/')
def index():
    return "Komering Character Recognition API is running"

if __name__ == '__main__':
    app.run(debug=True)
