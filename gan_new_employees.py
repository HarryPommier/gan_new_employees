import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.datasets import fashion_mnist
from keras.optimizers import Adam
import os
from keras.layers import Activation, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
from keras.backend.tensorflow_backend import set_session


def build_generator_conv(z_dim, size4):
    model = Sequential()

    model.add(Dense(size4*size4*256, input_dim = z_dim))
    model.add(Reshape((size4, size4, 256)))

    model.add(Conv2DTranspose(128, kernel_size = 3, strides = 2, padding = "same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.01))

    model.add(Conv2DTranspose(64, kernel_size = 3, strides = 1, padding = "same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.01))

    model.add(Conv2DTranspose(1, kernel_size = 3, strides = 2, padding = "same"))
    model.add(Activation("tanh"))

    return model

def build_discriminator_conv(img_shape):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides = 2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha = 0.001))

    model.add(Conv2D(64, kernel_size = 3, strides = 2, input_shape = img_shape, padding = "same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.001))

    model.add(Conv2D(128, kernel_size = 3, strides = 2, input_shape = img_shape, padding = "same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.001))

    model.add(Flatten())
    model.add(Dense(1, activation = "sigmoid"))

    return model

def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

def train_gan(X_train, generator, discriminator, gan, iterations, batch_size, sample_step, z_dim, metrics, suffix):
    X_train = X_train/127.5 - 1
    X_train = np.expand_dims(X_train, axis=3)
    real_label = np.ones((batch_size, 1))
    fake_label = np.zeros((batch_size, 1))

    for iteration in range(iterations):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        z = np.random.normal(0, 1, (batch_size, z_dim))
        fake_imgs = generator.predict(z)

        d_loss_real = discriminator.train_on_batch(real_imgs, real_label)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_label)
        d_loss, accuracy = 0.5*np.add(d_loss_real, d_loss_fake)

        z = np.random.normal(0, 1, (batch_size, z_dim))
        g_loss = gan.train_on_batch(z, real_label)

        if (iteration + 1)%sample_step == 0 or iteration == 0:
            metrics["losses"].append((d_loss, g_loss))
            metrics["accuracies"].append(100*accuracy)
            metrics["it_checkpoints"].append(iteration + 1)
            print("{} [D loss: {}, acc: {}] [G loss: {}]".format(iteration+1, d_loss, 100*accuracy, g_loss))
            it1 = iteration + 1
            image_sample(generator, it1, z_dim, suffix)

def image_sample(generator, it, z_dim, suffix, img_per_row=4, img_per_col=4):
    z = np.random.normal(0, 1, (img_per_col*img_per_row, z_dim))
    img_gen = generator.predict(z)
    img_gen = 0.5*img_gen + 0.5
    fig, ax = plt.subplots(img_per_row, img_per_col, dpi=400, figsize=(img_per_row, img_per_col), sharex = True, sharey = True)
    cpt = 0
    for i in range(img_per_row):
        for j in range(img_per_col):
            ax[i, j].imshow(img_gen[cpt,:,:,0], cmap = "gray")
            ax[i, j].axis("off")
            cpt += 1
    plt.savefig("samples{}/it_{}.jpg".format(suffix, it), format="jpg")


if __name__ == "__main__":
    size = 64 #image size must be a multiple of 4 

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)

    #prepare trainig set
    os.system("rm raw_data/*.jpg raw_data/*.png")
    os.system("rm zenika_faces/*")
    os.system("cp raw_data/raw_zenika_faces_cropped/* raw_data")
    os.system("bash script/resize_faces.sh {}".format(size))
    os.system("cp raw_data/*.jpg raw_data/*.png zenika_faces")
    time.sleep(5)

    #load trainng set
    zenika_faces_dir = "zenika_faces/"
    images_path = []
    for dirs, subdirs, files in os.walk(zenika_faces_dir):
        images_path.append((dirs, files))
    images_path = np.array(images_path[0][1])

    faces = []
    for img_name in images_path:
        im = cv2.imread(zenika_faces_dir + img_name)
        im = np.mean(im, axis=-1)
        faces.append(im)
    X_train = np.stack(faces, axis=0)
    print(X_train.shape)


    sample_filename_suffix = "" 
    os.makedirs("./samples{}".format(sample_filename_suffix), exist_ok = True)
    z_dim = 100 
    channels = 1
    img_shape = (size, size, channels)
    size4 = int(size/4)

    discriminator_conv = build_discriminator_conv(img_shape)
    discriminator_conv.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])
    discriminator_conv.trainable = False
    generator_conv = build_generator_conv(z_dim, size4)
    gan_conv = build_gan(generator_conv, discriminator_conv)
    gan_conv.compile(loss = "binary_crossentropy", optimizer=Adam())

    iterations = 20000
    batch_size = 8 
    sample_step = 1000
    metrics = {"losses":[], "accuracies":[], "it_checkpoints":[]}
    train_gan(X_train, generator_conv, discriminator_conv, gan_conv, iterations, batch_size, sample_step, z_dim, metrics, sample_filename_suffix)