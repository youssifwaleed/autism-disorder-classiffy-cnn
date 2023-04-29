import csv

import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers

TRAIN_DIR = 'train\\train\\'
TEST_DIR = 'test\\test\\'
IMG_SIZE = 61
LR = 0.001
MODEL_NAME = 'Autism-NonaAutism-cnn'

def create_label(image_name):
    word_label = image_name.split('.')
    if word_label[0][:8] == 'autistic':
        return 1
    elif word_label[0][:12] == 'non_autistic':
        return 0

def create_train_data_autistic():
    training_data = []
    paths = TRAIN_DIR+'autistic'
    for img in tqdm(os.listdir(paths)):
        path = os.path.join(paths, img)
        img_data = cv2.imread(path,0)
        img_data = img_data / 255.0
        img_data = cv2.resize(img_data,(IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])
    shuffle(training_data)
    return training_data

def create_train_data_non_autistic():
    training_data = []
    paths = TRAIN_DIR+'non_autistic'
    for img in tqdm(os.listdir(paths)):
        path = os.path.join(paths, img)
        img_data = cv2.imread(path, 0)
        img_data = img_data / 255.0
        img_data = cv2.resize(img_data,(IMG_SIZE, IMG_SIZE))
        label = create_label(img)
        training_data.append([np.array(img_data),label])
    shuffle(training_data)
    return training_data

def create_test_data():
    testing_data=[]
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path, 0)
        img_data = img_data / 255.0
        im = img_data
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append(np.array(img_data))
    return testing_data

train_data = create_train_data_autistic()
train_data.extend(create_train_data_non_autistic())
np.random.shuffle(train_data)

testdata = create_test_data()
valid_data = np.array([i for i in testdata]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

train_size = int((len(train_data)*70)/100)
train = train_data[:train_size]
test = train_data[train_size:]


X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test]

tf.reset_default_graph()
#CNN MODEL
model = Sequential([
  #Input_Layer
  layers.InputLayer(input_shape=[IMG_SIZE, IMG_SIZE, 1], name='input'),
  #1st Layer
  layers.Conv2D(32, 5, padding='same', activation='relu'),
  layers.MaxPooling2D(5,padding='same'),
  #2d
  layers.Conv2D(64, 5, padding='same', activation='relu'),
  layers.MaxPooling2D(5,padding='same'),
  #3rd
  layers.Conv2D(128, 5, padding='same', activation='relu'),
  layers.MaxPooling2D(5,padding='same'),
  #4th
  layers.Conv2D(64, 5, padding='same', activation='relu'),
  layers.MaxPooling2D(5,padding='same'),
  #5th
  layers.Conv2D(32, 5, padding='same', activation='relu'),
  layers.MaxPooling2D(5,padding='same'),
  layers.Flatten(),
  layers.Dense(1024, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(2,activation='softmax')
])

model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
test_loss, test_acc = model.evaluate(X_test,y_test)
print("\n" + str(test_acc))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.1, 1])
plt.legend(loc='lower right')
plt.show()

out = model.predict(np.array(valid_data))
val = 0
with open('Submit.csv', 'a') as fd:
    for i in range(len(out)):
        if out[i][0] >= 0.51:
            val = 1
        else:
            val = 0
        print(val)
        myCsvRow = [str(i) + ".jpg", val]
        writer = csv.writer(fd)
        writer.writerow(myCsvRow)
    fd.close()
