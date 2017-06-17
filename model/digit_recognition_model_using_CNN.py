import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

# mini batch gradient decent
batch_size = 200

# number of predictable classes from 0 to 9 is 10
num_classes = 10

# epochs number
epochs = 20

# input image dimension
img_rows, img_cols = 28, 28

'''
Part 1 - Data loading and preprocessing
'''

# loading the data and splitting it into training and test dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# more reshaping
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train/255
X_test = X_test/255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

'''
Part 2 - building the CNN model
'''
# K.set_image_dim_ordering('th')

# Initializing the CNN
model = Sequential()

# step 1 - convolutional layer with rectified linear unit activation
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))

# step 2 - Poolong
model.add(MaxPooling2D(pool_size=(2, 2)))

# step 3 - Adding second convolutional layer to improve accuracy
model.add(Conv2D(64, activation='relu', kernel_size=(3, 3)))

# step 4 - second maxpooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# step 5 - adding Dropout that helps prevent overfitting.
model.add(Dropout(0.5))

# step6 - Flattening
model.add(Flatten())

# step 7 - full connection
# Adding hidden layers
# hidden layer
model.add(Dense(units=128, activation='relu'))
# output layer
# output a softmax to squash the matrix into output probabilities
model.add(Dense(units=num_classes, activation='softmax'))

# step 8 - compiling the model
# Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
# categorical ce since we have multiple classes (10)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# step 9 - train model
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs, verbose=1,
          validation_data=(X_test, y_test))

# step 9 - evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])
print("Baseline Error: %.2f%%" % (100-score[1]*100))

# Save the model
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model... Hurray!!!")




