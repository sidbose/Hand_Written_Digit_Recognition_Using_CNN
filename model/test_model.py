import numpy as np
from keras.models import model_from_json
from scipy.misc import imread, imresize

json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights
loaded_model.load_weights("model.h5")
print("Model has been loaded")

# compile and evaluate loaded model
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# reading image
image = imread("9.png", mode='L')
print(image)
image = np.invert(image)

image = imresize(image, (28, 28))
image = image.reshape(1, 28, 28, 1)

# perform the prediction
out = loaded_model.predict(image)
print(out)
response = np.array_str(np.argmax(out, axis=1))
print("response: " + response)

