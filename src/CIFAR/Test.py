from matplotlib import pyplot
from scipy.misc import toimage
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
import deepdish as dd
from sklearn.model_selection import train_test_split



def loadTrainData(filename = "./DATA/"):
    train = []
    label = []

    for i in range (0,10):
        for j in range(0,16):
            _data_dir = filename+str(i)+'/data_batch_'+str(i)+"_"+str(j)+".h5"
            _dict = dd.io.load(_data_dir)
            x = _dict['Data']
            y = _dict['Labels']
            train.append(x)
            label.append(y)

    train_set = np.array(train)
    label_set = np.array(label)

    print(train_set.shape)
    print(label_set.shape)

    return train_set,label_set

def loadTestData(filename = "./DATA/test/data_batch_test.h5"):

    _data_dir = filename
    _dict = dd.io.load(_data_dir)

    test_set = np.array(_dict['Data'])
    label_set = np.array(_dict['Labels'])

    print(test_set.shape)
    print(label_set.shape)

    return test_set,label_set

def show_imgs(X):
    pyplot.figure(1)
    k = 0
    for i in range(0, 4):
        for j in range(0, 4):
            pyplot.subplot2grid((4, 4), (i, j))
            pyplot.imshow(toimage(X[k]))
            k = k + 1
    # show the plot
    pyplot.show()

def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    elif epoch > 100:
        lrate = 0.0003
    return lrate

#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#show_imgs(x_test[:16])

train_set,label_set = loadTrainData()
test_set,test_label_set = loadTestData()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

train_set = np.reshape(train_set, (train_set.shape[0]*train_set.shape[1],train_set.shape[2],train_set.shape[3], 1))
test_set = np.reshape(test_set, (test_set.shape[0],test_set.shape[1],test_set.shape[2], 1))
label_set = np.reshape(label_set, (label_set.shape[0]*label_set.shape[1], 1))

X_train, X_test, Y_train, Y_test = train_test_split(train_set, label_set, test_size=0.33, random_state=42)



print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

print("Alishan")

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

print(test_set.shape)
print(test_label_set.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
test_set = test_set.astype('float32')

print(X_train.shape[1:])

# z-score
# mean = np.mean(x_train, axis=(0, 1, 2, 3))
# std = np.std(x_train, axis=(0, 1, 2, 3))
# x_train = (x_train - mean) / (std + 1e-7)
# x_test = (x_test - mean) / (std + 1e-7)

num_classes = 10
Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)
test_label_set = np_utils.to_categorical(test_label_set, num_classes)

weight_decay = 1e-4
model = Sequential()
model.add(
    Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=X_train.shape[1:]))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(X_train)

# training
batch_size = 64
# batch_size = 8
# num_epochs = 125
num_epochs = 10

opt_rms = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), \
                    steps_per_epoch=X_train.shape[0] // batch_size, epochs=num_epochs, \
                    verbose=1, validation_data=(X_test, Y_test), callbacks=[LearningRateScheduler(lr_schedule)])

# save to disk
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')

# testing
scores = model.evaluate(test_set, test_label_set, batch_size=128, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1] * 100, scores[0]))