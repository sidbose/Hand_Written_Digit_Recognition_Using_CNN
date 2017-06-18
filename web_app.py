# Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
# HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine
# for you automatically.
from flask import Flask, render_template, request
# scientific computing library for saving, reading, and resizing images
from scipy.misc import imread, imresize
# for matrix math
import numpy as np
# for regular expressions, saves time dealing with string data
import re
# system level operations (like loading files)
import sys
# for reading operating system data
import os

import base64

sys.path.append(os.path.abspath('./model'))

from load import *

# init flask app
app = Flask(__name__)

global model
model,graph = init()

# decoding an image from base64 into raw representation
def convertImage(imgData1):
	imgstr = re.search(b'base64,(.*)', imgData1).group(1)
	with open('output.png', 'wb') as output:
		output.write(base64.b64decode(imgstr))


@app.route('/')
def index():
    # initModel()
    # render out pre-built HTML file right on the index page
    return render_template('test.html')

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    imgdata = request.get_data()
    convertImage(imgdata)
    # read the image into memory
    x = imread('output.png', mode='L')
    # compute a bit-wise inversion so black becomes white and vice versa
    x = np.invert(x)
    # make it the right size
    x = imresize(x, (28, 28))
    # convert to a 4D tensor to feed into our model
    x = x.reshape(1, 28, 28, 1)

    # in our computation graph
    with graph.as_default():
        # perform the prediction
        out = model.predict(x)
        # convert the response to a string
        response = np.array_str(np.argmax(out, axis=1))
        return response

    # # perform the prediction
    # out = model.predict(x)
    # response = np.array_str(np.argmax(out, axis=1))
    # return response

if __name__ == '__main__':
    # decide what port to run the app in
    port = int(os.environ.get('PORT', 5000))
    # run the app locally on the given port
    app.run(host='0.0.0.0', port=port)
    # optional if we want to run in debugging mode
    # app.run(debug=True)

