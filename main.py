# import math                      # providing access to the mathematical functions defined by the C standard
# import matplotlib.pyplot as plt  # plotting library
# import scipy                     # scientific computnig and technical computing
# import cv2                       # working with, mainly resizing, images
# import numpy as np               # dealing with arrays
# import glob                      # return a possibly-empty list of path names that match pathname
# import os                        # dealing with directories
# import pandas as pd              # providing data structures and data analysis tools
# import tensorflow as tf
# import itertools
# import random
# from random import shuffle       # mixing up or currently ordered data that might lead our network astray in training.
# from tqdm import tqdm            # a nice pretty percentage bar for tasks. Thanks to viewer Daniel Bühler for this suggestion
# from PIL import Image
# from scipy import ndimage
# from pathlib import Path
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn import metrics
# from IPython import get_ipython
# # get_ipython().magic('matplotlib inline')
# np.random.seed(1)
# import datetime as dt
#
#
# from keras.preprocessing.image import ImageDataGenerator # need
# from keras.callbacks import ReduceLROnPlateau # need
# from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
# from keras.callbacks import ModelCheckpoint, EarlyStopping
# from keras.applications import Xception
# from keras.utils import to_categorical
import cv2
import datetime as dt
import glob
import itertools
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from keras import models, layers, optimizers
from keras.applications import Xception
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dropout, Dense



#===============================================================================
#Steps:
# -
#===============================================================================


#===============================================================================
# Main
#===============================================================================
if __name__ == '__main__':
    print('this is main!')
    cols = ['Label','Latin Name', 'Common Name','Train Images', 'Validation Images']
    labels = pd.read_csv("./10-monkey-species/monkey_labels.txt", names=cols, skiprows=1)

    #paths
    train_dir = Path('./training-resized/')
    test_dir = Path('./validation-resized/')

    #set labels
    labels = labels['Common Name']
    print(labels)

    ####Start traning models
    height=150
    width=150
    channels=3
    seed=1387
    batch_size = 64
    num_classes = 10
    epochs = 200
    data_augmentation = True
    num_predictions = 20


    #=============================================================================================
    # train_datagen is an object of ImageDataGenerator class
    #=============================================================================================
    train_datagen = ImageDataGenerator(rescale=1./255)

    #target_size is the dimensions to which all images found will be resized
    #class_mode creates 2D one-hot encoded labels for us
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                target_size=(height,width),
                                                batch_size=batch_size,
                                                seed=seed,
                                                class_mode='categorical')
    #=============================================================================================
    # train_generator returns a DirectoryIteratro yielding tuples of (x,y)
    # where x is a numpy array containing a batch of images with shape
    # (batch_size, target_size, channels) and y is a numpy array of corresponding labels
    #=============================================================================================



    # Test generator
    test_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(height,width),
                                                  batch_size=batch_size,
                                                  seed=seed,
                                                  class_mode='categorical')

#===============================================================================

    # Initialize the base model
    # used in feature extraction
    base_model = Xception(weights='imagenet',
                          include_top=False,
                          input_shape=(height, width, channels))
    base_model.summary()

    def extract_features(sample_count, datagen):
        start = dt.datetime.now()

        #Keras models are trained on Numpy arrays of input data and labels
        # we initially create numpy arrays containing all zeros of size specified below
        ####MAYBE: filter size???????
        features =  np.zeros(shape=(sample_count, 5, 5, 2048))
        labels = np.zeros(shape=(sample_count,10))
        generator = datagen
        i = 0

        #Each generator contains 2 lists
        ## 1 - features
        ## 2 - labels
        for inputs_batch,labels_batch in generator:
            stop = dt.datetime.now()
            time = (stop - start).seconds
            print('\r',
                  'Extracting features from batch', str(i+1), '/', len(datagen),
                  '-- run time:', time,'seconds',
                  end='')

            features_batch = base_model.predict(inputs_batch)

            features[i * batch_size : (i + 1) * batch_size] = features_batch
            labels[i * batch_size : (i + 1) * batch_size] = labels_batch
            i += 1

            if i * batch_size >= sample_count:
                break

        print("\n")

        return features,labels

    print('working on train')
    train_features, train_labels = extract_features(1095, train_generator)
    print('working on test')
    test_features, test_labels = extract_features(272, validation_generator)


    #need to reshape our matricies becuase Dense function only learns using 2D layers
    flat_dim = 5 * 5 * 2048
    train_features = np.reshape(train_features, (1095, flat_dim))
    test_features = np.reshape(test_features, (272, flat_dim))




    #Reduce learning rate when a metric has stopped improving.

    #Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates.
    #This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs,
    #the learning rate is reduced.
    #USED FOR THE model.fit()
    reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=2,
                                         cooldown=2,
                                         min_lr=0.00001,
                                         verbose=1)

    callbacks = [reduce_learning_rate]

    #create models
    print('working on creating model')
    model = Sequential()

    #add model layers
    #using input_dim because we are using 2D layers aka Dense
    ## Number of hidden layers = 512
    model.add(Dense(512, activation='relu', input_dim=flat_dim))
    #adding the Dropout helps prevent overfitting. rate: float between 0 and 1. Fraction of the input units to drop.
    model.add(Dropout(0.2))
    #we have 10 classes
    model.add(Dense(10, activation='softmax'))

    #compile model using accuracy to measure model performance
    print('working on compiling')
    #before training model, we need to configure the learning process, which is done via the compile method
    #arguments 'loss' is the objective that the model will try to minimize

    #when using the categorical_crossentropy loss, your targets should be in categorical format
    #(e.g. if you have 10 classes,
    ## the target for each sample should be a 10-dimensional vector that is all-zeros
    ## except for a 1 at the index corresponding to the class of the sample)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    #train the model
    ### batch_size: Integer or None. Number of samples per gradient update.
    ### epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
    ###
    print('working on training the model')
    history = model.fit(train_features,
                    train_labels,
                    epochs=50,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_split=0.1)
                    # ,
                    # callbacks=callbacks)


    preds = model.predict(test_features)

    #=============================================================================================
    # Plot data
    #=============================================================================================
    # Change labels from one-hot encoded
    predictions = [i.argmax() for i in preds]
    y_true = [i.argmax() for i in test_labels]

    def plot_confusion_matrix(cm, target_names,title='Confusion matrix',cmap=None,normalize=False):
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy
        if cmap is None:
            cmap = plt.get_cmap('Blues')
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float32') / cm.sum(axis=1)
            cm = np.round(cm,2)


        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel("Predicted label\naccuracy={:0.4f}\n misclass={:0.4f}".format(accuracy, misclass))
        print('Accuracy = ', accuracy)
        plt.show()

    cm = confusion_matrix(y_pred=predictions, y_true=y_true)
    plot_confusion_matrix(cm, normalize=True, target_names=labels)
