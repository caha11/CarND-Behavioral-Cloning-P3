import cv2
import csv
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import keras
import sklearn
from random import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D
from keras.layers.pooling import MaxPooling2D

class Pipeline:
    def __init__(self, model = None, epochs = 2):
        self.csv_lines = []
        self.train_samples = []
        self.valid_samples = []
        self.model = model
        self.epochs = epochs

    # Add lines from cvsfile
    def retrieve_data(self):
        with open('./data/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            # The csv file from Udacity, has a first line that needs to be ignored.
            next(reader)
            for line in reader:
                self.csv_lines.append(line)

    # Middle, Right, Left Image. Resize, Crop and Flip. 
    def process_batch_sample(self, batch_sample):
        images = []
        measurements = []

        # Read left, centre and right image.
        for image_selection in range(0,3):
            filename = batch_sample[image_selection].split('/')[-1]
            local_path = './data/IMG/' + filename
            image = cv2.imread(local_path)
            if image is None:
                print("Image was not read : ", local_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            #Crop and scale down the image to reduce params. 
            image = image[60:130, :]
            image = cv2.resize(image, (160, 70));
            images.append(image)			

        # Adding correction factor for the measurements of the left and right images
        measurement = float(batch_sample[3])
        measurements.append(measurement)
        measurements.append(measurement + 0.2)
        measurements.append(measurement - 0.2)

        augmented_images = []
        augmented_measurements = []

        # Flip the images so that it works for the right turns as well.
        for image, measurement in zip(images, measurements):
            augmented_images.append(image)
            augmented_measurements.append(measurement)
            flipped_image = cv2.flip(image, 1)
            flipped_measurement =  -1.0 * float(measurement)
            augmented_images.append(flipped_image)
            augmented_measurements.append(flipped_measurement)

        return augmented_images, augmented_measurements
  
    # To prevent loading all the images in the data at once.
    def batch_generator(self, lines, batch_size=128):
        num_samples = len(lines)
        while 1:
            shuffle(lines)
            for offset in range(0, num_samples, batch_size):
                batch_samples = lines[offset:offset + batch_size]
                images = []
                measurements = []

                for batch_sample in batch_samples:
                    images, measurements = self.process_batch_sample(batch_sample)

                X_train, y_train = np.array(images), np.array(measurements)
                yield sklearn.utils.shuffle(X_train, y_train)

    # Split samples into train and valid bins
    def split_samples(self):
        self.train_samples, self.valid_samples =  train_test_split(self.csv_lines, test_size = 0.2)

    # Generator for the training data
    def train_gen(self, batch_size = 128):
        return self.batch_generator(self.train_samples, batch_size)

    # Generator for the validation data
    def valid_gen(self, batch_size = 128):
        return self.batch_generator(self.valid_samples, batch_size)

    def run(self):
        self.split_samples()
        self.model.fit_generator(self.train_gen(),
                                 steps_per_epoch = len(self.train_samples) * 2,
                                 epochs = self.epochs,
                                 validation_data = self.valid_gen(),
                                 validation_steps = len(self.valid_samples))
        self.model.save('model.h5')

# Model based on the Nvidia's paper: https://devblogs.nvidia.com/deep-learning-self-driving-cars/
def model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape = (70, 160, 3)))
    model.add(Conv2D(filters = 24, kernel_size = 5, strides = 2, activation = 'relu'))
    model.add(Conv2D(filters = 36, kernel_size = 5, strides = 2, activation = 'relu'))
    model.add(Conv2D(filters = 48, kernel_size = 5, strides = 2, activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = 3, activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = 3, activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer='adam')

    return model

def main():
    
    pipeline = Pipeline(model(), epochs = 2)
    pipeline.retrieve_data()
    pipeline.run()

if __name__ == '__main__':
    main()
