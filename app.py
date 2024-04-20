from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import io
import os

# Initialize Flask application
app = Flask(__name__)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the pre-trained machine learning models
svm_model_path = os.path.join(script_dir, 'svm_model.pkl')
naive_bayes_model_path = os.path.join(script_dir, 'NB_model.pkl')
pca_model_path = os.path.join(script_dir, 'pca_model.pkl')
lda_model_path = os.path.join(script_dir, 'lda_model.pkl')
target_path = os.path.join(script_dir, 'target_data.csv')

svm_model = joblib.load(svm_model_path)
naive_bayes_model = joblib.load(naive_bayes_model_path)
pca = joblib.load(pca_model_path)
lda = joblib.load(lda_model_path)

# Function to load a single image from a file-path
def load_img(file_path, slice_, color, resize):
    default_slice = (slice(0, 250), slice(0, 250))  # Default slice size

    # Use the provided slice or the default
    if slice_ is None: 
        slice_ = default_slice
    else: 
        slice_ = tuple(s or ds for s, ds in zip(slice_, default_slice))

    # Calculate the height and width from the slice
    h_slice, w_slice = slice_
    h = (h_slice.stop - h_slice.start) // (h_slice.step or 1)
    w = (w_slice.stop - w_slice.start) // (w_slice.step or 1)

    # Apply resizing if needed
    if resize is not None:
        resize = float(resize)
        h = int(resize * h)
        w = int(resize * w)

    # Initialize the image array
    if not color: 
        face = np.zeros((h, w), dtype=np.float32)
    else: 
        face = np.zeros((h, w, 3), dtype=np.float32)

    # Load and process the image
    pil_img = Image.open(file_path)
    pil_img = pil_img.crop((w_slice.start, h_slice.start, w_slice.stop, h_slice.stop))

    if resize is not None: 
        pil_img = pil_img.resize((w, h))
    face = np.asarray(pil_img, dtype=np.float32)

    # Normalize pixel values and convert to grayscale if not color
    face /= 255.0
    if not color: 
        face = face.mean(axis=2)

    return face


# Define a route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if the request contains a file
        if 'image' not in request.files:
            return render_template('index.html', error='No image provided')
        # Read the image file
        image_file = request.files['image']

        face = load_img(image_file,slice_=None , color = False , resize = 0.4)
        X = face.reshape(1, -1)
        X_t = pca.transform(X)
        X_t = lda.transform(X_t)
        naive_bayes_prediction = naive_bayes_model.predict(X_t)
        svm_prediction = svm_model.predict(X_t)

        # Check if the file is empty
        if image_file.filename == '':
            return render_template('index.html', error='No image provided')
        target = pd.read_csv(target_path)
        predicted_name_svm = str(target['person_name'][svm_prediction[0]])
        predicted_name_bayes = str(target['person_name'][naive_bayes_prediction[0]])
        # Return the predictions as a JSON response
        # Return the predictions as variables to the template
        return render_template('index.html', svm_prediction=predicted_name_svm, naive_bayes_prediction=predicted_name_bayes)


    return render_template('index.html', error=None)

# Run the Flask application,
if __name__ == '__main__':
    app.run(debug=True)
