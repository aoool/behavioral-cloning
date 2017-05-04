import os
import zipfile

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D

# extract augmented data archive if not already extracted
if not (os.path.isdir("data_augmented") and os.path.exists("data_augmented.csv")):
    zip_ref = zipfile.ZipFile("data_raw.augmented", 'r')
    zip_ref.extractall(".")
    zip_ref.close()


# construct neural network model
# using the model from
#                http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

# model parameters
steering_angle_correction = 0.2

# the model's type is sequential
model = Sequential()

# crop input image to leave only useful information - the road
# the trees, sky and the car's hood will be removed
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))

# normalize input image
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

model.add(Convolution2D(filters=24, kernel_size=(5, 5)))
model.add(Convolution2D(filters=36, kernel_size=(5, 5)))
model.add(Convolution2D(filters=48, kernel_size=(5, 5)))
model.add(Convolution2D(filters=64, kernel_size=(3, 3)))
model.add(Convolution2D(filters=64, kernel_size=(3, 3)))

model.add(Flatten())

# model has only one output - steering angle
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, verbose=1)

# save model to be able to reuse it in autonomous driving
model.save("model.h5")
