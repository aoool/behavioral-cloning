import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, ELU, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


###
# module parameters
###

DATA_CSV = "data_augmented.csv"
STEERING_ANGLE_CORRECTION = 0.25
BATCH_SIZE = 256
TRAIN_TEST_SPLIT_FACTOR = 0.2
EPOCHS = 10
MODEL_NAME = 'commaai'
MODEL_FILE = 'model.h5'
INPUT_IMAGE_SHAPE = (160, 320, 3)
INPUT_IMAGE_CROPPING = ((70, 25), (0, 0))


###
# module functions
###

def compile_commaai_model():
    model = Sequential()
    model.add(Cropping2D(cropping=INPUT_IMAGE_CROPPING, input_shape=INPUT_IMAGE_SHAPE))
    model.add(Lambda(lambda x: x/127.5 - 1.))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model


def compile_nvidia_model():
    model = Sequential()
    model.add(Cropping2D(cropping=INPUT_IMAGE_CROPPING, input_shape=INPUT_IMAGE_SHAPE))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    # add 5 convolution layers
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    # flatten output of the last convolution layer and add some fully connected layers
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model


def compile_model(model_name):
    mapping = {'commaai': compile_commaai_model,
               'nvidia':  compile_nvidia_model}

    assert (model_name in mapping.keys())

    return mapping[model_name]()


def generator(samples, data, batch_size=128):
    num_samples = len(samples)
    while 1:  # loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                image = cv2.imread(data['IMAGE'][batch_sample])
                angle = float(data['STEERING_ANGLE'][batch_sample])
                images.append(image)
                angles.append(angle)
            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)


def prepare_data(in_data, correction):
    """Prepare data to be feed into (image, steering angle) generator.
    
    Input data maps 3 images per record - center camera, left camera and right camera.
    Output data has 1 image per record. The correction is applied for left and right camera images.
    
    Args:
        in_data: data from csv file (dictionary)
        correction: positive float to be applied to right and left image's steering angles
    """
    assert (correction >= 0.0)

    in_records = len(in_data['STEERING_ANGLE'])
    out_data = {'IMAGE': [], 'STEERING_ANGLE': []}

    for i in range(in_records):
        out_data['IMAGE'].append(in_data['CENTER_IMAGE'])
        out_data['STEERING_ANGLE'].append(in_data['STEERING_ANGLE'])

        out_data['IMAGE'].append(in_data['LEFT_IMAGE'])
        left_angle = in_data['STEERING_ANGLE'] + correction
        left_angle = left_angle if left_angle <= 1.0 else 1.0
        out_data['STEERING_ANGLE'].append(left_angle)

        out_data['IMAGE'].append(in_data['RIGHT_IMAGE'])
        right_angle = in_data['STEERING_ANGLE'] - correction
        right_angle = right_angle if right_angle >= -1.0 else -1.0
        out_data['STEERING_ANGLE'].append(right_angle)

    assert (len(out_data['IMAGE']) == len(out_data['STEERING_ANGLE']))

    return out_data


def history(history_object):
    # print the keys contained in the history object
    print(history_object.history.keys())

    # plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


def train(from_scratch=True):
    """Train the model on data.
    
    Runs data preparation, model architecture compilation and model training.
    
    Args:
        from_scratch: whether to compile a new model or load existing one from file
    """
    print("INFO: read csv file with the data and prepare data dictionary")
    data = prepare_data(pd.read_csv(DATA_CSV).to_dict(orient='list'), correction=STEERING_ANGLE_CORRECTION)
    num_samples = len(data['STEERING_ANGLE'])

    print("INFO: split samples to train and validation sets")
    train_samples, validation_samples = \
        train_test_split(np.array([i for i in range(num_samples)]), test_size=TRAIN_TEST_SPLIT_FACTOR)

    print("INFO: initialize generators for train and validation data sets")
    train_generator = generator(train_samples, data, batch_size=BATCH_SIZE)
    validation_generator = generator(validation_samples, data, batch_size=BATCH_SIZE)

    # compile model or load existing
    if from_scratch:
        print("INFO: compile model", MODEL_NAME)
        model = compile_model(MODEL_NAME)
    else:
        print("INFO: load model", MODEL_FILE)
        model = load_model(MODEL_FILE)

    print("INFO: fit the model")
    checkpoint_callback = ModelCheckpoint("model-%s-{epoch:02d}.h5" % MODEL_NAME, verbose=1)
    history_obj = model.fit_generator(train_generator,
                                      samples_per_epoch=len(train_samples),
                                      validation_data=validation_generator,
                                      nb_val_samples=len(validation_samples),
                                      callbacks=[checkpoint_callback],
                                      nb_epoch=EPOCHS,
                                      verbose=1)

    print("INFO: save the model to model-%s.h5" % MODEL_NAME)
    model.save("model-%s.h5" % MODEL_NAME)

    print("INFO: model summary")
    model.summary()

    print("INFO: draw fit history graph")
    history(history_obj)
