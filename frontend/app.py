from flask import Flask, render_template, request, jsonify
import logging
import sys
import json
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from keras.preprocessing.image import img_to_array

app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')

@app.route('/recognizeImage', methods=['POST'])
def recognizeImage():
    requestImage = request.form.get('img')
    imageBase64 = requestImage[22::]
    image = Image.open(BytesIO(base64.b64decode(imageBase64)))
    imageResize = image.resize((150, 150), Image.NEAREST)
    print(orgImage.shape)
    intImage = np.array(orgImage) / 255.0    
    image = np.reshape(intImage, (1,28,28,1)).astype(np.float32)


    print(orgImage)
    print(orgImage.shape)

    return "fsdfa"

if __name__ == '__main__':
    app.run()