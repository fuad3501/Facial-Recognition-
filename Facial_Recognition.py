# Import Standard Dependencies
import cv2
import os
import uuid
from time import sleep
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten

# File Structure:
# anchor => Input Image
# positive => Verification Image of a positive match
# negative => Verification Image of a positive match

# Setup Directory Paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# Make Directories
# os.makedirs(POS_PATH)
# os.makedirs(NEG_PATH)
# os.makedirs(ANC_PATH)

# Labelled Faces in Wild Dataset: http://vis-www.cs.umass.edu/lfw/
# Move Labelled Wild Faces to Data Folder

# for directory in os.listdir('lfw'):
#     for file in os.listdir(os.path.join('lfw', directory)):
#         EX_PATH = os.path.join('lfw', directory, file)
#         NEW_PATH = os.path.join(NEG_PATH, file)
#         os.replace(EX_PATH, NEW_PATH)


# Image Capture Module
# collect anchor and positive images using webcam (via cv2): 250x250px
cap = cv2.VideoCapture(0)  # define capture object that establishes connection to webcam

while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120 + 250, 200:200 + 250, :]  # slice image to format into 250x250px

    if cv2.waitKey(1) & 0XFF == ord('a'):  # waits 1ms and checks keyboard if a is hit, collects anchor image
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)

    if cv2.waitKey(1) & 0XFF == ord('p'):  # waits 1ms and checks keyboard if p is hit, collects positive image
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)

    if cv2.waitKey(1) & 0XFF == ord('q'):  # waits 1ms and checks keyboard if q is hit to close all windows
        break

    cv2.imshow("Image Collection", frame)  # shows image back to screen
# switch off webcam once finished capturing
cap.release()
cv2.destroyAllWindows()

# 3) *** Load and Preprocess Images ***

# Data Pipeline:
# grabs all images from directories and returns their files paths (300 times)
anchor = tf.data.Dataset.list_files(ANC_PATH + '\*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH + '\*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH + '\*.jpg').take(300)


# Scale and Resize Images
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)  # reads image from file path
    img = tf.io.decode_jpeg(byte_img)  # decodes jpeg image to number values
    img = tf.image.resize(img, (100, 100))  # resize to 100x100x3 px (3 channel colours)
    img = img / 255.0  # scale to 0-1

    return img


# create labelled dataset
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(
    tf.ones(len(anchor)))))  # passing through a positive and anchor image should return a 1 label
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(
    tf.ones(len(anchor)))))  # passing through a negative and anchor image should return a 0 label
data = positives.concatenate(negatives)


# Build train-test partition
def preprocess_twin(input_image, validation_image, label):
    return preprocess(input_image), preprocess(validation_image), label


# Build data-loader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)  # returns shuffled dataset tensor (3 channels) of anchor + positive/negative sample + label

# Training Partition
train_data = data.take(round(len(data) * .7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)  # pre-fetches next set of images to not bottleneck NN

# Testing Partition
test_data = data.skip(round(len(data) * .7))
test_data = test_data.take(round(len(data) * .3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


# 4) *** Model Engineering ***

# Build Embedding Layer

def make_embedding():
    inp = Input(shape=(100, 100, 3), name='input_image')  # defines shape of input tensor
    # First Block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)  # defines first convolution layer with 64 filters of size (10, 10)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)  # defines first maximum pooling layer with 64 filters of size (2, 2)
    # Second Block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)
    # Third Block
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)
    # Fourth Block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)  # Final convolution layer
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)  # Final feature vector of size 4096x1

    return Model(inputs=[inp], outputs=[d1], name='embedding')  # returns the model function with the parameters inserted


# Build Siamese L1 Distance Layer Class (compares two separate embedding layer - siamese element)
class L1Dist(Layer):                # class inherits Layer class

    def __init__(self, **kwargs):
        super().__init__()

    # Performs Similarity Calculation by Subtraction
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)     # return the difference of the two layers


# Build Siamese Model
def make_siamese_model():

    embedding = make_embedding()
    input_image = Input(name='input_image', shape=(100, 100, 3))       # anchor image
    validation_image = Input(name='validation_image', shape=(100, 100, 3))  # poqqqqqqqqqqsitive image

    # combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    # Classification Layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name="SiameseNN")


model = make_siamese_model()
print(model.summary())

# 5) *** Training and Testing the Model ***

# 6) *** Model Evaluation ***

# 7) *** Real-Time Test ***





