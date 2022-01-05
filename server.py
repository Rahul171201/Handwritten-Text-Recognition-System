import pickle
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from numpy import argmax
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import base64
import os
app = Flask(__name__)
CORS(app)


def load_image(filename):
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    img = img_to_array(img)
    img = img.reshape(1, 784)
    return img


DIGIT_MODEL = int(1)
ALPHABET_MODEL = int(2)

model_name_map = {
    DIGIT_MODEL: "digit_model.sav",
    ALPHABET_MODEL: "letter_model.sav",
}

# API route


@app.route('/data', methods=['POST'])
@cross_origin(origin='*')
def postTest():
    a = request.get_json(force=True)
    if not a:
        return jsonify({'msg': 'Missing JSON'}), 400

    requested_model = a.get('id')
    if not requested_model:
        return jsonify({'msg': 'ID is missing'}), 400

    imgname = a.get('name')
    if not imgname:
        return jsonify({'msg': 'Image name is missing or Invalid'}), 400

    img64 = a.get('img')
    if not img64:
        return jsonify({'msg': 'Image is missing or Invalid'}), 400

    imgdata = base64.b64decode(img64)
    filename = imgname  # I assume you have a way of picking unique filenames
    with open('images/'+filename, 'wb') as f:
        f.write(imgdata)

    model_name = model_name_map[requested_model]
    print('ID: ', requested_model)
    print('Image: Received image is saved at images/'+imgname)
    print('Requested model: ', model_name)

    svm = pickle.load(open(model_name, 'rb'))
    prediction = svm.predict(load_image('images/'+imgname))
    try:
        os.remove('images/'+imgname)
    except OSError as error:
        print('', '')
    if requested_model == 1:
        print('\nPrediction: ', prediction[0])
    else:
        print('\nPrediction: ', chr(prediction[0]+64))
    print('--------------------------------------')
    if requested_model == 1:
        return jsonify({'Prediction': str(prediction[0])}), 200
    else:
        return jsonify({'Prediction': chr(prediction[0]+64)}), 200


if __name__ == "__main__":
    try:
        os.mkdir('images')
    except OSError as error:
        print('', '')
    app.run(debug=True)
