from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
from keras.models import load_model
import numpy as np
import os
from pathlib import Path

UPLOAD_FOLDER = 'static/file/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model configuration matching the DenseNet201 notebook
IMAGE_SIZE = (224, 224)
CLASSES = ['NORMAL', 'TUBERCULOSIS', 'PNEUMONIA', 'COVID19']

def load_trained_model():
    """Load the final trained model from either .keras or .h5 format"""
    model_keras_path = 'final_model.keras'
    model_h5_path = 'final_model.h5'
    
    if os.path.exists(model_keras_path):
        return load_model(model_keras_path)
    elif os.path.exists(model_h5_path):
        return load_model(model_h5_path)
    else:
        raise FileNotFoundError(f"Model not found. Please ensure {model_keras_path} or {model_h5_path} exists in the project root.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        try:
            file1 = request.files['filename']
            if file1.filename == '':
                return render_template('index.html', msg='No file selected', view='style=display:block', view1='style=display:none')
            
            imgfile = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
            file1.save(imgfile)
            
            # Load model
            model = load_trained_model()
            
            # Load and preprocess image
            img_ = image.load_img(imgfile, target_size=IMAGE_SIZE + (3,))
            img_array = image.img_to_array(img_)
            img_processed = np.expand_dims(img_array, axis=0)
            
            # Apply DenseNet preprocessing
            img_processed = preprocess_input(img_processed)
            
            # Make prediction
            prediction = model.predict(img_processed)
            pred_index = np.argmax(prediction)
            confidence = float(prediction[0][pred_index]) * 100
            result = str(CLASSES[pred_index]).title()
            
            # Format result with confidence
            result_msg = f'{result} (Confidence: {confidence:.2f}%)'
            
            return render_template('index.html', msg=result_msg, src=imgfile, view='style=display:block', view1='style=display:none')
        
        except Exception as e:
            error_msg = f'Error: {str(e)}'
            return render_template('index.html', msg=error_msg, view='style=display:block', view1='style=display:none')

if __name__ == '__main__':
    app.run(debug=True)