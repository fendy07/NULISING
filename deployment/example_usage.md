# Example Usage of Komering Character Recognition REST API

## REST API Endpoint

**POST** `/api/predict`

---

## Headers

Content-Type: application/json
---

## Request Body (JSON)

```json
{
  "label": "a",
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```
* label → The folder name where the image will be saved.
* image → A base64-encoded image string (from a canvas or image file).

Example Response
```json
{
  "label": "a",
  "score": 0.912,
  "message": "High confidence prediction"
}
```
* label → The label provided by the user (folder name).
* score → Model confidence for its predicted class (0 to 1).
* message → Prediction confidence message:
    * >= 0.85 → High confidence prediction
    * >= 0.60 → Moderate confidence prediction
    * < 0.60 → Low confidence prediction

Notes
* All images are resized to 50x50 pixels before prediction.
* Images are saved in the folder:
static/rawdata/<label>/<label>_<timestamp>.png
	•	The API handles images with a white background automatically.
