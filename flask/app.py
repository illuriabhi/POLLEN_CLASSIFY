from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
model = load_model('model1.h5', compile=False)

labels = ['arecaceae', 'anadenanthera', 'arrabidaea', 'cecropia', 'chromolaena', 'combretum', 'croton', 'dipteryx', 'eucalyptus', 'ficus', 'guarea', 'hymenaea', 'inga', 'jacaranda', 'machaerium', 'myrcia', 'ocotea', 'piper', 'protium', 'psidium', 'schinus', 'solanum', 'urochloa']
@app.route('/predict-page')
def predict_page():
    return render_template('prediction.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        f = request.files['file']
        filename = secure_filename(f.filename)

        upload_folder = os.path.join(app.root_path, 'static/uploads')
        os.makedirs(upload_folder, exist_ok=True)

        filepath = os.path.join(upload_folder, filename)
        f.save(filepath)

        img = load_img(filepath, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        result = labels[np.argmax(prediction)]

        # Adjust relative path for browser access
        rel_img_path = os.path.join('static/uploads', filename)

        return render_template('prediction.html', prediction=result, img_path=rel_img_path)
    except Exception as e:
        print("Error:", e)
        return "An error occurred: " + str(e)

if __name__ == '__main__':
    app.run(debug=True)

application = app
