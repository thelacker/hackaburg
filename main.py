'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop, Adagrad
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
# from densenet import DenseNet

# Различные колбеки
# Рання остановка в случае "необучения"
# earlystopping = EarlyStopping(monitor='val_f1', min_delta=0.001, patience=2, verbose=0, mode='auto')
# Сохранение чекпоинтов с указанием эпохи, loss и f-меры
checkpointer = ModelCheckpoint(filepath='tmp/weights.{epoch:02d}-{val_loss:.2f}-{val_f1:.2f}.hdf5', verbose=1)

# optimizer = RMSprop(lr=1e-7, rho=0.3, epsilon=None, decay=1e-2)
# optimizer = Adagrad(lr=0.000000001, decay=0.5)

img_width, img_height = 300, 300
input_shape = (img_width, img_height, 3)

train_data_dir = 'data_new/train'
validation_data_dir = 'data_new/validation'

# Количество картинок, которое будет использоваться при обучении
# Если картинок в папке data будет меньше, то применится ImageGenerator
nb_train_samples = 2000
nb_validation_samples = 500

epochs = 20
batch_size = 16


def f1(y_true, y_pred):
    """F1 metric"""

    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def get_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    # model = DenseNet(
    #     nb_classes = 61,
    #     img_dim = (3, img_width, img_height),
    #     depth = 22,
    #     nb_dense_block = 2,
    #     growth_rate = 24,
    #     nb_filter = 32,
    #     dropout_rate=0.25
    # )
    #
    model.load_weights('tmp/weights.17-0.33-0.87.hdf5')
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=[f1, 'accuracy'])
    return model


model = get_model()

# model.load_weights('weights.01-4.41-0.40.hdf5')
# img = load_img('/Users/thelacker/PycharmProjects/logos/res/7.jpg',False,target_size=(300,300))
# x = img_to_array(img)
# x = x/255
# x = np.rollaxis(x, 2, 0)
# x = np.expand_dims(x, axis=0)
# preds = np.argmax(model.predict(x))
# print(preds)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.5,
        zoom_range=0.5,
        horizontal_flip=False,
        fill_mode='nearest',
        rescale=1. / 255,
        # data_format = 'channels_first'
)

test_datagen = ImageDataGenerator(rescale=1. / 255)#, data_format = 'channels_first')

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# img = load_img('/Users/thelacker/PycharmProjects/logos/res/test-8.jpg',False,target_size=(300,300))
# x = img_to_array(img)
# x = x/255
# x = np.expand_dims(x, axis=0)
# preds = model.predict_classes(x)
# prob = model.predict_proba(x)
# print(preds, prob)
#
# img = load_img('/Users/thelacker/PycharmProjects/logos/res/test-13.jpg',False,target_size=(300,300))
# x = img_to_array(img)
# x = x/255
# x = np.expand_dims(x, axis=0)
# preds = model.predict_classes(x)
# prob = model.predict_proba(x)
# print(preds, prob)

# model.fit_generator(
#     train_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=nb_validation_samples // batch_size,
#     callbacks=[checkpointer])
#
# model.save('first_try.h5')