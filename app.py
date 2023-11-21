from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

app = Flask(__name__, static_url_path='/static')

# Make sure to provide the correct path to your H5 model file
model = load_model(r"C:\\Users\\Admin\\Downloads\\project\\project\\wcv.h5")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)
        
        img = image.load_img(filepath, target_size=(180, 180, 3))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)
        
        prediction = np.argmax(model.predict(img_data), axis=1)
        index = ['alien_test', 'cloudy', 'foggy', 'rainy', 'shine', 'sunrise']
        result = index[prediction[0]]
        print(result)
        return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
