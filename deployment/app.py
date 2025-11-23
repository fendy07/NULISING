from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import io
import base64
import os
import time
import numpy as np
import onnxruntime as ort

app = Flask(__name__)

# ====================== CONFIG ======================
BASE_SAVE_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'rawdata')
os.makedirs(BASE_SAVE_FOLDER, exist_ok=True)

labels = ['a', 'ba', 'ca', 'da', 'ga', 'ha', 'ja', 'ka', 'la', 'ma', 'mba',
          'mpa', 'na', 'nca', 'nda', 'nga', 'ngka', 'nja', 'pa', 'ra', 'rha-gha',
          'sa', 'ta', 'wa', 'ya']


ONNX_MODEL_PATH = "../komering_resnet18_FINAL.onnx"

# ====================== LOAD MODEL ======================
print("[INFO] Loading Komering ONNX model...")
ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name
print(f"[INFO] Model loaded! Input name: {input_name}")

# ====================== UTILS ======================
def softmax(x, axis=1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def preprocess_image(pil_img):
    img = pil_img.convert('RGB').resize((50, 50), Image.LANCZOS)
    x = np.array(img, dtype=np.float32) / 255.0
    x = x.transpose(2, 0, 1)
    x = np.expand_dims(x, axis=0).astype(np.float32)
    return x

def predict_image(pil_img):
    input_data = preprocess_image(pil_img)
    logits = ort_session.run(None, {input_name: input_data})[0]
    probs = softmax(logits).flatten()
    pred_idx = int(np.argmax(probs))
    predicted = labels[pred_idx]
    confidence = float(probs[pred_idx])
    top5_idx = np.argsort(probs)[-5:][::-1]
    top5 = {labels[i]: float(probs[i]) for i in top5_idx}
    return predicted, confidence, top5

# ====================== API DOCUMENTATION ======================
HTML_DOCS = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Komering Character Recognition API</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f7f9; color: #333; margin: 40px; line-height: 1.6; }
        .container { max-width: 900px; margin: auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; }
        h2 { color: #3498db; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        pre { background: #2c3e50; color: #f1c40f; padding: 15px; border-radius: 8px; overflow-x: auto; }
        code { background: #ecf0f1; padding: 2px 6px; border-radius: 4px; font-size: 90%; }
        .endpoint { background: #e8f4fc; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #3498db; }
        .status { color: green; font-weight: bold; }
        ul { padding-left: 20px; }
        footer { text-align: center; margin-top: 50px; color: #7f8c8d; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Komering Script Character Recognition API</h1>
        <p style="text-align:center;"><span class="status">● LIVE & READY</span></p>

        <h2>POST /api/predict</h2>
        <div class="endpoint">
            <strong>Purpose:</strong> Recognize a single Komering character from an image<br>
            <strong>Method:</strong> <code>POST</code><br>
            <strong>Content-Type:</strong> <code>application/json</code>
        </div>

        <h2>Request Body (JSON)</h2>
        <pre>{
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "label": "ba"      // optional, for logging & accuracy
}</pre>

        <h2>Response Example</h2>
        <pre>{
    "predicted": "ba",
    "confidence": 0.9987,
    "confidence_percent": 99.87,
    "user_label": "ba",
    "correct": true,
    "top5": {
        "ba": 0.9987,
        "pa": 0.0008,
        "ma": 0.0003,
        "a": 0.0001,
        "nga": 0.00005
    },
    "saved_image": "/static/rawdata/ba/ba_1739123456789.png",
    "message": "Success! Predicted 'ba'"
}</pre>

        <h2>Supported Characters (25 classes)</h2>
        <p><code>a, ba, ca, da, ga, ha, ja, ka, la, ma, mba, mpa, na, nca, nda, nga, ngka, nja, pa, ra, rha-gha, sa, ta, wa, ya</code></p>

        <h2>Test with cURL</h2>
        <pre>curl -X POST https://yourusername.pythonanywhere.com/api/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "image": "data:image/png;base64,$(base64 -w 0 your_image.png)",
    "label": "ka"
  }'</pre>

        <footer>
            <p>Komering Character Recognition • ONNX + Flask • 2025</p>
        </footer>
    </div>
</body>
</html>
'''

# ====================== ROUTES ======================
@app.route('/')
def home():
    return render_template_string(HTML_DOCS)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "Missing 'image' field in JSON"}), 400

        b64 = data['image']
        if ',' in b64:
            b64 = b64.split(',', 1)[1]

        user_label = data.get('label', 'unknown')

        # Decode and validate image
        try:
            img_data = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_data))
        except Exception as e:
            return jsonify({"error": "Invalid base64 image data", "details": str(e)}), 400

        # Predict
        pred, conf, top5 = predict_image(img)

        # Save image
        save_dir = os.path.join(BASE_SAVE_FOLDER, user_label)
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{user_label}_{int(time.time()*1000)}.png"
        save_path = os.path.join(save_dir, filename)
        img.save(save_path)

        return jsonify({
            "predicted": pred,
            "confidence": round(conf, 4),
            "confidence_percent": round(conf * 100, 2),
            "user_label": user_label,
            "correct": pred == user_label,
            "top5": top5,
            "saved_image": f"/static/rawdata/{user_label}/{filename}",
            "message": f"Success! Predicted '{pred}'"
        })

    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# ====================== WSGI ======================
application = app

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)