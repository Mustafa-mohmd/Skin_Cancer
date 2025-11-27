import os
import time
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Use the directory containing this file as base (works on PythonAnywhere too)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ENHANCED_FOLDER = os.path.join(BASE_DIR, "static", "enhanced")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ENHANCED_FOLDER, exist_ok=True)

app = Flask(__name__)
# Prefer environment secret in production (set this on PythonAnywhere web app settings)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-this-secret")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

model_handle = None
model_error = None

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_uploads():
    try:
        for f in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"Failed to clean uploads folder: {e}")

def cleanup_enhanced():
    try:
        for f in os.listdir(ENHANCED_FOLDER):
            file_path = os.path.join(ENHANCED_FOLDER, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"Failed to clean enhanced folder: {e}")

def get_model():
    global model_handle, model_error
    if model_handle is not None or model_error is not None:
        return
    try:
        from model import load_model
        model_handle = load_model()
    except Exception as e:
        model_error = str(e)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    cleanup_uploads()
    cleanup_enhanced()
    if request.method == "GET":
        return render_template("predict.html", prediction=None, enhanced_url=None, error=None)

    # POST
    if "image" not in request.files:
        flash("No image file part in the request.")
        return redirect(request.url)

    file = request.files["image"]
    if file.filename == "":
        flash("No selected file.")
        return redirect(request.url)

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload PNG, JPG, JPEG, or BMP.")
        return redirect(request.url)

    filename = secure_filename(file.filename)
    unique_name = f"{int(time.time()*1000)}_{filename}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(save_path)

    # Ensure model is loaded
    get_model()
    if model_error is not None:
        return render_template("predict.html", prediction=None, enhanced_url=None, error=f"Model not loaded: {model_error}")

    # Enhance image
    enhanced_url = None
    try:
        from utils import enhance_image
        enhanced_name = f"enh_{unique_name}.jpg"
        enhanced_full_path = os.path.join(ENHANCED_FOLDER, enhanced_name)
        enhance_image(save_path, enhanced_full_path)
        enhanced_url = url_for("static", filename=f"enhanced/{enhanced_name}")
    except Exception as e:
        flash(f"Image enhancement failed: {e}")

    # Predict
    prediction_text = None
    try:
        from model import predict
        label, score = predict(model_handle, save_path)
        prediction_text = f"{label} ({score:.2f})"
    except Exception as e:
        return render_template("predict.html", prediction=None, enhanced_url=enhanced_url, error=f"Prediction failed: {e}")

    return render_template("predict.html", prediction=prediction_text, enhanced_url=enhanced_url, error=None)

# ---------- Local run only ----------
# PythonAnywhere runs your app via WSGI and imports the `app` object.
# We only start the builtin Flask server when running this file directly (local dev).
if __name__ == "__main__" and "PYTHONANYWHERE_DOMAIN" not in os.environ:
    app.run(host="127.0.0.1", port=5000, debug=True)
