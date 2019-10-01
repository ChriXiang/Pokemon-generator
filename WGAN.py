import matplotlib
matplotlib.use('Agg')

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Convolution2D, Conv2DTranspose
from keras.models import Sequential, Model, load_model
from keras.optimizers import RMSprop

import keras.backend as K

from imageio import imread, imsave
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import cv2
import math

class WGAN():
    def __init__(self):
        self.in_rows = 128
        self.in_cols = 128
        self.channels = 3
        self.input_size = (self.in_rows, self.in_cols, self.channels)
        self.noise_dim = 100
        
        #hypper parameters
        self.n_dis = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)
        
        self.discriminator = self.Discriminator()
        self.generator = self.Generator()
        
        self.discriminator.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])
        
        #feed generator noize
        z = Input(shape=(self.noise_dim,))
        img = self.generator(z)
        
        #only train the generator in stacked model
        self.discriminator.trainable = False

        # discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The stacked model  (stacked generator and discriminator)
        self.stack_model = Model(z, valid)
        self.stack_model.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])
        
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
    
    def Generator(self):
        
        model = Sequential()
        
        model.add(Dense((256 * 4 * 4), input_dim =self.noise_dim))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Activation("relu"))
        #reshape 4*4*256
        if K.image_data_format() == 'channels_first':
            model.add(Reshape((256, 4, 4), input_shape=(256 * 4 * 4,)))
        else:
            model.add(Reshape((4, 4, 256), input_shape=(256 * 4 * 4,)))
        #8*8*128
        
        model.add(Conv2DTranspose(128, (3, 3), strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Activation("relu"))
        #16*16*64
        
        model.add(Conv2DTranspose(64, (3, 3), strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Activation("relu"))
        #32*32*32
        
        model.add(Conv2DTranspose(32, (3, 3), strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Activation("relu"))
        #64*64*16
   
        model.add(Conv2DTranspose(16, (3, 3), strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Activation("relu"))
        #128*128*3
        model.add(Conv2DTranspose(self.channels, (5, 5), strides=2, padding='same'))
        model.add(Activation("tanh"))

        #model.summary()
        
        noise = Input(shape=(self.noise_dim,))
        img = model(noise)

        return Model(noise, img)
        
    def Discriminator(self):
        
        model = Sequential()
        #1
        model.add(Conv2D(16, kernel_size=(3, 3), strides=2, input_shape=self.input_size, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        #2
        model.add(Conv2D(32, kernel_size=(3, 3), strides=2,  padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        #3
        model.add(Conv2D(64, kernel_size=(3, 3), strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        #4
        model.add(Conv2D(128, kernel_size=(3, 3), strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        #flatten
        model.add(Flatten())
        model.add(Dense(1))
        
        #model.summary()

        img = Input(shape=self.input_size)
        validity = model(img)

        return Model(img, validity)      
    
    def train(self,  interval, check_point):
        
        BATCH_SIZE = 32
        EPOCHS = 60001
        
        current_dir = os.getcwd()
        #pokemon_dir = os.path.join(current_dir, '128_poke')
        pokemon_dir = './training_poke'
        all_images = []
        for file in os.listdir(pokemon_dir):
            if file in ['.DS_S_h.png','.DS_S_v.png','.floyddata']:
                continue

            all_images.append(cv2.imread(os.path.join(pokemon_dir,file)))

        all_images = np.asarray(all_images)
        all_images = (all_images.astype(np.float32) - 127.5) / 127.5
        
        valid = -np.ones((BATCH_SIZE, 1)) #-1
        fake = np.ones((BATCH_SIZE, 1)) #1
        
        for epoch in range(48001,EPOCHS):

            for _ in range(self.n_dis):
                #train discriminator first
                
                #select minibatches
                idx = np.random.randint(0, all_images.shape[0], BATCH_SIZE)
                
                imgs = all_images[idx]
                
                # use generator's prediction to train discriminator
                noise = np.random.normal(0, 1, (BATCH_SIZE, self.noise_dim))
                #noise = np.random.uniform(-1.0, 1.0,size=[BATCH_SIZE, self.noise_dim]).astype(np.float32)
                gen_imgs = self.generator.predict(noise)

                #train discriminator with both real and fake images
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                #use mean of losses
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip discriminator weights
                for layer in self.discriminator.layers:
                    weights = layer.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    layer.set_weights(weights)
            
            
            #train stacked generator and discriminator (discriminator won't be trained)
            g_loss = self.stack_model.train_on_batch(noise, valid)

            # show progress
            print ("%d: [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % interval == 0:
                self.save_progress(epoch)

            # save models every 2000 epoches
            if epoch % check_point == 0 and epoch >= check_point:
            	self.save_check(epoch)

                
    def save_progress(self, epoch):
        row, col = 5, 5
        noise = np.random.normal(0, 1, (row * col, self.noise_dim))
        #noise = np.random.uniform(-1.0, 1.0,size=[row * col, self.noise_dim]).astype(np.float32)
        gen_imgs = self.generator.predict(noise).astype(np.float32)

        # Rescale images 0 - 1
        gen_imgs = np.clip(0.5 * gen_imgs + 0.5, 0.0, 1.0)

        fig, axs = plt.subplots(row, col)
        
        count = 0
        for i in range(row):
            for j in range(col):
                axs[i,j].imshow(gen_imgs[count, :,:])
                axs[i,j].axis('off')
                count += 1
        axs[0,2].set_title('epoch:  %d' % epoch)

        # if not os.path.exists('mydirectory'):
        #     os.makedirs('mydirectory')
        save_path = './result_2'

        fig.savefig(save_path+"/_"+str(epoch)+".jpg")
        plt.close()
        print("reuslt of epoch %d saved" % epoch)

    def save_check(self, epoch):
        self.generator.save('./model_2/generator_%d' % epoch)
        self.discriminator.save('./model_2/discriminator_%d' % epoch)
        print("model of epoch %d saved" % epoch)

#########################
#load model section start
########################
def wasserstein_loss( y_true, y_pred):
    return K.mean(y_true * y_pred)

generator = load_model('./model_2/generator_48000')
discriminator = load_model('./model_2/discriminator_48000', custom_objects={'wasserstein_loss': wasserstein_loss})

#preprocess with loaded models
optimizer = RMSprop(lr=0.00005)
discriminator.compile(loss=wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])
#feed generator noize
z = Input(shape=(100,))
img = generator(z)
        
#only train the generator in stacked model
discriminator.trainable = False

valid = discriminator(img)

# The stacked model  (stacked generator and discriminator)
stack_model = Model(z, valid)
stack_model.compile(loss=wasserstein_loss,
    optimizer=optimizer,
    metrics=['accuracy'])
##########################
#load model section finish
#########################

wgan = WGAN()
#model replace below
wgan.generator = generator
wgan.discriminator = discriminator
wgan.stack_model = stack_model
#model replace above
wgan.train(interval=100 , check_point = 2000)


