import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, Model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os
import pathlib
import random
import nibabel as nib
import imageio
from skimage.metrics import structural_similarity as ssim
import os, random, json, PIL, shutil, re, imageio, glob
from PIL import ImageDraw
import glob
import os
from keras import Input
from keras.applications import VGG19
from keras.callbacks import TensorBoard
from keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Dense
from keras.layers import Conv2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
import random
from numpy import asarray
from itertools import repeat
import imageio
from imageio import imread
from PIL import Image
from skimage.transform import resize as imresize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
tf.__version__


BUFFER_SIZE = 5000
HEIGHT = 256
WIDTH = 256
HEIGHT_RESIZE = 128
WIDTH_RESIZE = 128
CHANNELS = 1
# BATCH_SIZE = 4
# EPOCHS = 25
BATCH_SIZE = 4
EPOCHS = 80
TRANSFORMER_BLOCKS = 6
GENERATOR_LR = 2e-4
DISCRIMINATOR_LR = 2e-4

import math
import os

import cv2
import numpy as np

def psnr(original, contrast):
    mse = np.mean((original - contrast) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    print('mse: ',mse)
    return PSNR
  
def preprocess_image_T2(image):
    image= tf.image.decode_png(image, channels=1)
    image = tf.image.pad_to_bounding_box(image, offset_height=0, offset_width=0, target_height=256, target_width=256)
    image = tf.image.resize(image, [256, 256])
    image = tf.image.rot90(image, k=3) # rotate 270º
    image = tf.image.flip_up_down(image)
    image = (image-127.5)/127.5
    return image

def preprocess_image(image):
    image= tf.image.decode_png(image, channels=1)
    image = tf.image.pad_to_bounding_box(image, offset_height=0, offset_width=60, target_height=256, target_width=256)
    image = tf.image.resize(image, [256, 256])
    image = (image-127.5)/127.5
    return image

def preprocess_image_test(image):
    image= tf.image.decode_png(image, channels=1)
    image = tf.image.pad_to_bounding_box(image, offset_height=0, offset_width=0, target_height=256, target_width=256)
    image = tf.image.resize(image, [256, 256])
    image = (image-127.5)/127.5
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def load_and_preprocess_image_test(path):
    image = tf.io.read_file(path)
    return preprocess_image_test(image)

def load_and_preprocess_image_T2(path):
    image = tf.io.read_file(path)
    return preprocess_image_T2(image)
  
data_root = pathlib.Path('../input/ixi-t1/image slice-T1')
all_image_paths_T1 = list(data_root.glob('*/*'))
all_image_paths_T1.sort()
print(len(all_image_paths_T1))
new_all_image_paths_T1 = []
for i in range(len(all_image_paths_T1)):
    if (i+5)%10 == 0:
        new_all_image_paths_T1.append(all_image_paths_T1[i])
all_image_paths_T1 = new_all_image_paths_T1
all_image_paths_T1 = [str(path) for path in all_image_paths_T1[:500]]
image_count= len(all_image_paths_T1)
ds_T1 = tf.data.Dataset.from_tensor_slices((all_image_paths_T1))
dataset_T1 = ds_T1.map(load_and_preprocess_image).batch(BATCH_SIZE).repeat().shuffle(512)
dataset_T1 = dataset_T1.cache()
dataset_T1_test = ds_T1.map(load_and_preprocess_image).batch(4)

len(list(dataset_T1_test))
len(new_all_image_paths_T1)

data_root = pathlib.Path('../input/ixit2-slices/image slice-T2')
all_image_paths_T2 = list(data_root.glob('*/*'))
all_image_paths_T2.sort()
print(len(all_image_paths_T2))
new_all_image_paths_T2 = []
for i in range(len(all_image_paths_T2)):
    if (i+4)%10 == 0:
        new_all_image_paths_T2.append(all_image_paths_T2[i])
all_image_paths_T2 = new_all_image_paths_T2
all_image_paths_T2 = [str(path) for path in all_image_paths_T2[:1000]]
#random.shuffle(all_image_paths)
print(len(all_image_paths_T2))
ds_T2 = tf.data.Dataset.from_tensor_slices((all_image_paths_T2))
dataset_T2 = ds_T2.map(load_and_preprocess_image_T2).batch(BATCH_SIZE).repeat().shuffle(512)
dataset_T2 = dataset_T2.cache()
dataset_T2_test = ds_T2.map(load_and_preprocess_image_T2).batch(2)

len(list(dataset_T2_test))
iteration = iter (dataset_T2)
image = next(iteration )
plt.imshow(image[0],cmap = 'gray')
plt.grid(False)

iteration2 = iter (dataset_T1)
image2 = next(iteration2 )
plt.imshow(image2[0],cmap = 'gray')
plt.grid(False)

conv_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
gamma_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    
def encoder_block(input_layer, filters, size=3, strides=2, apply_instancenorm=True, activation=layers.ReLU(), name='block_x'):
    block = layers.Conv2D(filters, size, 
                     strides=strides, 
                     padding='same', 
                     use_bias=False, 
                     kernel_initializer=conv_initializer, 
                     name=f'encoder_{name}')(input_layer)

    if apply_instancenorm:
        block = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(block)
        
    block = activation(block)

    return block

def transformer_block(input_layer, size=3, strides=1, name='block_x'):
    filters = input_layer.shape[-1]
    
    block = layers.Conv2D(filters, size, strides=strides, padding='same', use_bias=False, 
                     kernel_initializer=conv_initializer, name=f'transformer_{name}_1')(input_layer)
#     block = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(block)
    block = layers.ReLU()(block)
    
    block = layers.Conv2D(filters, size, strides=strides, padding='same', use_bias=False, 
                     kernel_initializer=conv_initializer, name=f'transformer_{name}_2')(block)
#     block = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(block)
    
    block = layers.Add()([block, input_layer])

    return block

def decoder_block(input_layer, filters, size=3, strides=2, apply_instancenorm=True, name='block_x'):
    block = layers.Conv2DTranspose(filters, size, 
                              strides=strides, 
                              padding='same', 
                              use_bias=False, 
                              kernel_initializer=conv_initializer, 
                              name=f'decoder_{name}')(input_layer)

    if apply_instancenorm:
        block = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(block)

    block = layers.ReLU()(block)
    
    return block

# Resized convolution
# def decoder_rc_block(input_layer, filters, size=3, strides=1, apply_instancenorm=True, name='block_x'):
#     block = tf.image.resize(images=input_layer, method='bilinear', 
#                             size=(input_layer.shape[1]*2, input_layer.shape[2]*2))
    
# #     block = tf.pad(block, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC") # Works only with GPU
# #     block = L.Conv2D(filters, size, strides=strides, padding='valid', use_bias=False, # Works only with GPU
#     block = layers.Conv2D(filters, size, 
#                      strides=strides, 
#                      padding='same', 
#                      use_bias=False, 
#                      kernel_initializer=conv_initializer, 
#                      name=f'decoder_{name}')(block)

#     if apply_instancenorm:
#         block = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(block)

#     block = layers.ReLU()(block)
    
#     return block

def generator_fn(height=HEIGHT, width=WIDTH, channels=CHANNELS, transformer_blocks=TRANSFORMER_BLOCKS):
    OUTPUT_CHANNELS = 1
    inputs = layers.Input(shape=[height, width, channels], name='input_image')

    # Encoder
    enc_1 = encoder_block(inputs, 64,  7, 1, apply_instancenorm=False, activation=layers.ReLU(), name='block_1') # (bs, 256, 256, 64)
    enc_2 = encoder_block(enc_1, 128, 3, 2, apply_instancenorm=True, activation=layers.ReLU(), name='block_2')   # (bs, 128, 128, 128)
    enc_3 = encoder_block(enc_2, 256, 3, 2, apply_instancenorm=True, activation=layers.ReLU(), name='block_3')   # (bs, 64, 64, 256)
    
    # Transformer
    x = enc_3
    for n in range(transformer_blocks):
        x = transformer_block(x, 3, 1, name=f'block_{n+1}') # (bs, 64, 64, 256)

    # Decoder
    x_skip = layers.Concatenate(name='enc_dec_skip_1')([x, enc_3]) # encoder - decoder skip connection
    
    dec_1 = decoder_block(x_skip, 128, 3, 2, apply_instancenorm=True, name='block_1') # (bs, 128, 128, 128)
    x_skip = layers.Concatenate(name='enc_dec_skip_2')([dec_1, enc_2]) # encoder - decoder skip connection
    
    dec_2 = decoder_block(x_skip, 64,  3, 2, apply_instancenorm=True, name='block_2') # (bs, 256, 256, 64)
    x_skip = layers.Concatenate(name='enc_dec_skip_3')([dec_2, enc_1]) # encoder - decoder skip connection

    outputs = last = layers.Conv2D(OUTPUT_CHANNELS, 7, 
                              strides=1, padding='same', 
                              kernel_initializer=conv_initializer, 
                              use_bias=False, 
                              activation='tanh', 
                              name='decoder_output_block')(x_skip) # (bs, 256, 256, 3)

    generator = Model(inputs, outputs)
    
    return generator
  
 def discriminator_fn(height=HEIGHT, width=WIDTH, channels=CHANNELS):
    inputs = layers.Input(shape=[height, width, channels], name='input_image')
    #inputs_patch = L.experimental.preprocessing.RandomCrop(height=70, width=70, name='input_image_patch')(inputs) # Works only with GPU

    # Encoder    
    x = encoder_block(inputs, 64,  4, 2, apply_instancenorm=False, activation=layers.LeakyReLU(0.2), name='block_1') # (bs, 128, 128, 64)
    x = encoder_block(x, 128, 4, 2, apply_instancenorm=True, activation=layers.LeakyReLU(0.2), name='block_2')       # (bs, 64, 64, 128)
    x = encoder_block(x, 256, 4, 2, apply_instancenorm=True, activation=layers.LeakyReLU(0.2), name='block_3')       # (bs, 32, 32, 256)
    x = encoder_block(x, 512, 4, 1, apply_instancenorm=True, activation=layers.LeakyReLU(0.2), name='block_4')       # (bs, 32, 32, 512)

    outputs = layers.Conv2D(1, 4, strides=1, padding='valid', kernel_initializer=conv_initializer)(x)                # (bs, 29, 29, 1)
    
    discriminator = Model(inputs, outputs)
    
    return discriminator
  
 T1_generator = generator_fn() # transforms T2 to T1
T1_discriminator = discriminator_fn() # differentiates real T1 and generated T1
# T1_generator = tf.keras.models.load_model('../input/dcgan/T1/generateT1_3_25')
# T1_discriminator = tf.keras.models.load_model('../input/dcgan/T1/discriminateT1_3_25')
T1_generator.summary()

T1_discriminator.summary()

class DCGan(keras.Model):
    def __init__(
        self,
        T1_generator,
        T1_discriminator,
        lambda_cycle=10,
    ):
        super(DCGan, self).__init__()
        self.T1_gen = T1_generator
        self.T1_disc = T1_discriminator
        self.lambda_cycle = lambda_cycle
        
    def compile(
        self,
        T1_gen_optimizer,
        T1_disc_optimizer,
        gen_loss_fn,
        disc_loss_fn,
        identity_loss_fn
    ):
        super(DCGan, self).compile()
        self.T1_gen_optimizer = T1_gen_optimizer
        self.T1_disc_optimizer = T1_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.identity_loss_fn = identity_loss_fn
        
    def train_step(self, batch_data):
        real_T1, real_T2 = batch_data
        
        with tf.GradientTape(persistent=True) as tape:
            # photo to monet back to photo
            fake_T1 = self.T1_gen(real_T2, training=True)

            # generating itself
            same_T1 = self.T1_gen(real_T1, training=True)

            # discriminator used to check, inputing real images
            disc_real_T1 = self.T1_disc(real_T1, training=True)

            # discriminator used to check, inputing fake images
            disc_fake_T1 = self.T1_disc(fake_T1, training=True)
            
            # evaluates generator loss
            T1_gen_loss = self.gen_loss_fn(disc_fake_T1)


            # evaluates total generator loss
            total_T1_gen_loss = T1_gen_loss + self.identity_loss_fn(real_T1, same_T1, self.lambda_cycle)
        

            # evaluates discriminator loss
            T1_disc_loss = self.disc_loss_fn(disc_real_T1, disc_fake_T1)

        # Calculate the gradients for generator and discriminator
        T1_generator_gradients = tape.gradient(total_T1_gen_loss,
                                                  self.T1_gen.trainable_variables)
       

        T1_discriminator_gradients = tape.gradient(T1_disc_loss,
                                                      self.T1_disc.trainable_variables)
       
        # Apply the gradients to the optimizer
        self.T1_gen_optimizer.apply_gradients(zip(T1_generator_gradients,
                                                 self.T1_gen.trainable_variables))

        self.T1_disc_optimizer.apply_gradients(zip(T1_discriminator_gradients,
                                                  self.T1_disc.trainable_variables))

        
        return {
            "T1_gen_loss": total_T1_gen_loss,   
            "T1_disc_loss": T1_disc_loss,
        }
      
def discriminator_loss(real, generated):
     real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real), real)

     generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated), generated)

     total_disc_loss = real_loss + generated_loss

     return total_disc_loss * 0.5
    
def generator_loss(generated):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated), generated)
  
def identity_loss(real_image, same_image, LAMBDA):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss
  
@tf.function
def linear_schedule_with_warmup(step):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    lr_start   = 2e-4
    lr_max     = 2e-4
    lr_min     = 0.
    
    steps_per_epoch = int(max(500, 500)//BATCH_SIZE)
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = 1
    hold_max_steps = total_steps * 0.8
    
    if step < warmup_steps:
        lr = (lr_max - lr_start) / warmup_steps * step + lr_start
    elif step < warmup_steps + hold_max_steps:
        lr = lr_max
    else:
        lr = lr_max * ((total_steps - step) / (total_steps - warmup_steps - hold_max_steps))
        if lr_min is not None:
            lr = tf.math.maximum(lr_min, lr)

    return lr

steps_per_epoch = int(max(500, 500)//BATCH_SIZE)
total_steps = EPOCHS * steps_per_epoch
rng = [i for i in range(0, total_steps, 50)]
y = [linear_schedule_with_warmup(x) for x in rng]

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(20, 6))
plt.plot(rng, y)
print(f'{EPOCHS} total epochs and {steps_per_epoch} steps per epoch')
print(f'Learning rate schedule: {y[0]:.3g} to {max(y):.3g} to {y[-1]:.3g}')
print(steps_per_epoch)

@tf.function
def linear_schedule_with_warmup(step):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    lr_start   = 2e-4
    lr_max     = 2e-4
    lr_min     = 0.
    
    steps_per_epoch = int(max(500, 500)//BATCH_SIZE)
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = 1
    hold_max_steps = total_steps * 0.8
    
    if step < warmup_steps:
        lr = (lr_max - lr_start) / warmup_steps * step + lr_start
    elif step < warmup_steps + hold_max_steps:
        lr = lr_max
    else:
        lr = lr_max * ((total_steps - step) / (total_steps - warmup_steps - hold_max_steps))
        if lr_min is not None:
            lr = tf.math.maximum(lr_min, lr)

    return lr

steps_per_epoch = int(max(500, 500)//BATCH_SIZE)
total_steps = EPOCHS * steps_per_epoch
rng = [i for i in range(0, total_steps, 50)]
y = [linear_schedule_with_warmup(x) for x in rng]

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(20, 6))
plt.plot(rng, y)
print(f'{EPOCHS} total epochs and {steps_per_epoch} steps per epoch')
print(f'Learning rate schedule: {y[0]:.3g} to {max(y):.3g} to {y[-1]:.3g}')

lr_T1_gen = lambda: linear_schedule_with_warmup(tf.cast(T1_generator_optimizer.iterations, tf.float32))

T1_generator_optimizer = optimizers.Adam(learning_rate=lr_T1_gen, beta_1=0.5)


    # Create discriminators
lr_T1_disc = lambda: linear_schedule_with_warmup(tf.cast(T1_discriminator_optimizer.iterations, tf.float32))

T1_discriminator_optimizer = optimizers.Adam(learning_rate=lr_T1_disc, beta_1=0.5)


    
    # Create GAN
gan_model = DCGan(T1_generator, 
                  T1_discriminator)

gan_model.compile(    T1_gen_optimizer=T1_generator_optimizer,
                      T1_disc_optimizer=T1_discriminator_optimizer,
                      gen_loss_fn=generator_loss,
                      disc_loss_fn=discriminator_loss,
                      identity_loss_fn=identity_loss)

# Callbacks
class GANMonitor(Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, num_img=4, T1_path='T1', T2_path='T2'):
        self.num_img = num_img
        self.T1_path = T1_path
        self.T2_path = T2_path
        # Create directories to save the generate images
        if not os.path.exists(self.T1_path):
            os.makedirs(self.T1_path)
        if not os.path.exists(self.T2_path):
            os.makedirs(self.T2_path)

    def on_epoch_end(self, epoch, logs=None):
        # Monet generated images
        fig = plt.figure(figsize=(8,4))
        for i, img in enumerate(dataset_T2_test.take(self.num_img)):
            prediction = T1_generator(img, training=False).numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            plt.subplot(2,4,i +1)
            plt.imshow(img[0], cmap = 'gray')
            plt.axis('off')
            plt.subplot(2,4,i +5)
            plt.imshow(prediction[0], cmap = 'gray')
            plt.axis('off')
#             plt.savefig('img')
            plt.savefig(f'{self.T1_path}/visualiation_{i}_{epoch+1}')
        plt.show()
#         plt.savefig(f'{self.T1_path}/visualiation_{i}_{epoch+1}')
        
        if epoch%6 == 0:
            
            tf.keras.models.save_model(T1_generator, f'{self.T1_path}/generateT1_{i}_{epoch+1}')
            tf.keras.models.save_model(T1_discriminator, f'{self.T1_path}/discriminateT1_{i}_{epoch+1}')
            
gan_ds = tf.data.Dataset.zip((dataset_T1, dataset_T2))

 history =   gan_model.fit(  gan_ds,
                             epochs=EPOCHS,
                             callbacks=[GANMonitor()],
                             steps_per_epoch=(max(500, 500)//BATCH_SIZE) ).history    
  
 from skimage.io import imread, imshow

old=imread('../input/1-image-for-psnr/old1.png')
imshow(old)
new=imread('../input/1-image-for-psnr/new1.png')
imshow(new)

old1 = tf.image.resize(old, [256, 256])
new1 = tf.image.resize(new, [256, 256])

print("-- First Test --")
# old11 = np.squeeze(old1)
# new11=np.squeeze(new1)
# s = ssim(old1, new1)
print(f"PSNR value is {psnr(old1, new1)} dB")
# print(s)

def compute_ssim(original_image, generated_image):
    
    original_image = tf.convert_to_tensor(original_image, dtype=tf.float32)
    generated_image = tf.convert_to_tensor(generated_image, dtype=tf.float32)
    ssim = tf.image.ssim(original_image, generated_image, max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)

    return tf.math.reduce_mean(ssim, axis=None, keepdims=False, name=None)
  
ss = compute_ssim(old1, new1) 
print(ss)

from skimage.io import imread, imshow

old=imread('../input/theimg/Screenshot (6513).png')
imshow(old)
new=imread('../input/theimg/Screenshot (6514).png')
imshow(new)

old1 = tf.image.resize(old, [256, 256])
new1 = tf.image.resize(new, [256, 256])

print("-- First Test --")
# old11 = np.squeeze(old1)
# new11=np.squeeze(new1)
# s = ssim(old1, new1)
print(f"PSNR value is {psnr(old1, new1)} dB")
# print(s)

ss = compute_ssim(old1, new1) 
print(ss)

from skimage.io import imread, imshow

old=imread('../input/theimg/Screenshot (6513).png')
imshow(old)
new=imread('../input/newimgg/6.png')
imshow(new)

old1 = tf.image.resize(old, [256, 256])
new1 = tf.image.resize(new, [256, 256])

print("-- First Test --")
# old11 = np.squeeze(old1)
# new11=np.squeeze(new1)
# s = ssim(old1, new1)
print(f"PSNR value is {psnr(old1, new1)} dB")
# print(s)

ss = compute_ssim(old1, new1) 
print(ss)

import shutil
shutil.make_archive('./res', 'zip', './')

dataroot = pathlib.Path('../input/new-data')
allimagepaths_res = list(dataroot.glob('*/*'))
allimagepaths_res.sort()
print(len(allimagepaths_res))
newallimagepaths_res = []
for i in range(len(allimagepaths_res)):
    newallimagepaths_res.append(allimagepaths_res[i])
allimagepaths_res = newallimagepaths_res
allimagepaths_res = [str(path) for path in allimagepaths_res[:80]]
len(allimagepaths_res)
ds_res = tf.data.Dataset.from_tensor_slices((allimagepaths_res))
dataset_res = ds_res.map(load_and_preprocess_image).batch(BATCH_SIZE).repeat().shuffle(512)
dataset_res = dataset_res.cache()
dataset_res_test = ds_res.map(load_and_preprocess_image).batch(4)

data_path = dataroot

epochs = 500

# batch size equals to 8 (due to RAM limits)
batch_size = 8

# define the shape of low resolution image (LR) 
low_resolution_shape = (64, 64, 3)

# define the shape of high resolution image (HR) 
high_resolution_shape = (256, 256, 3)

# optimizer for discriminator, generator 
common_optimizer = Adam(0.0002, 0.5)

# use seed for reproducible results
SEED = 2020 
tf.random.set_seed(SEED)

def find_img_dims(image_list):
    
    min_size = []
    max_size = []
    
    for i in range(len(image_list)):
        im = Image.open(image_list[i])
        min_size.append(min(im.size))
        max_size.append(max(im.size))
    
    return min(min_size), max(max_size)
  
min_size, max_size = find_img_dims(allimagepaths_res)
print('The min and max image dims are {} and {} respectively.'
      .format(min_size, max_size))

def compute_psnr(original_image, generated_image):
    
    original_image = tf.convert_to_tensor(original_image, dtype=tf.float32)
    generated_image = tf.convert_to_tensor(generated_image, dtype=tf.float32)
    psnr = tf.image.psnr(original_image, generated_image, max_val=1.0)

    return tf.math.reduce_mean(psnr, axis=None, keepdims=False, name=None)
  
def plot_psnr(psnr):
    
    psnr_means = psnr['psnr_quality']
    plt.figure(figsize=(10,8))
    plt.plot(psnr_means)    
    plt.xlabel('Epochs')
    plt.ylabel('PSNR') 
    plt.title('PSNR')
    
def compute_ssim(original_image, generated_image):
    
    original_image = tf.convert_to_tensor(original_image, dtype=tf.float32)
    generated_image = tf.convert_to_tensor(generated_image, dtype=tf.float32)
    ssim = tf.image.ssim(original_image, generated_image, max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)

    return tf.math.reduce_mean(ssim, axis=None, keepdims=False, name=None)
  
def plot_ssim(ssim):
    
    ssim_means = ssim['ssim_quality']

    plt.figure(figsize=(10,8))
    plt.plot(ssim_means)
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.title('SSIM')
    
 def plot_loss(losses):

    d_loss = losses['d_history']
    g_loss = losses['g_history']
    
   
    plt.figure(figsize=(10,8))
    plt.plot(d_loss, label="Discriminator loss")
    plt.plot(g_loss, label="Generator loss")
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Loss")    
    plt.legend()
    
def sample_images(image_list, batch_size, high_resolution_shape, low_resolution_shape):
    
    """
    Pre-process a batch of training images
    """
    
    # image_list is the list of all images
    # ransom sample a batch of images
    images_batch = np.random.choice(image_list, size=batch_size)
    
    lr_images = []
    hr_images = []
    

    for img in images_batch:
  
        img1 = imread(img, as_gray=False, pilmode='RGB')
        #img1 = imread(img, pilmode='RGB')
        img1 = img1.astype(np.float32)
        
        # change the size     
        img1_high_resolution = imresize(img1, high_resolution_shape)
        img1_low_resolution = imresize(img1, low_resolution_shape)
                

        # do a random horizontal flip
        if np.random.random() < 0.5:
            img1_high_resolution = np.fliplr(img1_high_resolution)
            img1_low_resolution = np.fliplr(img1_low_resolution)
       
        hr_images.append(img1_high_resolution)
        lr_images.append(img1_low_resolution)
        
   
    # convert lists into numpy ndarrays
    return np.array(hr_images), np.array(lr_images)    
  
def save_images(original_image, lr_image, r_image, path):
    
    """
    Save LR, HR (original) and generated R
    images in one panel 
    """
    
    fig, ax = plt.subplots(1,3, figsize=(10, 6))

    images = [original_image, lr_image, r_image]
    titles = ['HR', 'LR','R - generated']

    for idx,img in enumerate(images):
        # (X + 1)/2 to scale back from [-1,1] to [0,1]
        ax[idx].imshow((img + 1)/2.0, cmap='gray')
        ax[idx].axis("off")
    for idx, title in enumerate(titles):    
        ax[idx].set_title('{}'.format(title))
        
    plt.savefig(path)    
    
def residual_block(x):

    filters = [64, 64]
    kernel_size = 3
    strides = 1
    padding = "same"
    momentum = 0.8
    activation = "relu"

    res = Conv2D(filters=filters[0], kernel_size=kernel_size, strides=strides, padding=padding)(x)
    res = Activation(activation=activation)(res)
    res = BatchNormalization(momentum=momentum)(res)

    res = Conv2D(filters=filters[1], kernel_size=kernel_size, strides=strides, padding=padding)(res)
    res = BatchNormalization(momentum=momentum)(res)

    res = Add()([res, x])
    
    return res
  
def build_generator():
    
    # use 16 residual blocks in generator
    residual_blocks = 16
    momentum = 0.8
    
    # input LR dimension: 4x downsample of HR
    input_shape = (64, 64, 3)
    
    # input for the generator
    input_layer = Input(shape=input_shape)
    
    # pre-residual block: conv layer before residual blocks 
    gen1 = Conv2D(filters=64, kernel_size=9, strides=1, padding='same', activation='relu')(input_layer)
    
    # add 16 residual blocks
    res = residual_block(gen1)
    for i in range(residual_blocks - 1):
        res = residual_block(res)
    
    # post-residual block: conv and batch-norm layer after residual blocks
    gen2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(res)
    gen2 = BatchNormalization(momentum=momentum)(gen2)
    
    # take the sum of pre-residual block(gen1) and post-residual block(gen2)
    gen3 = Add()([gen2, gen1])
    
    # upsampling
    gen4 = UpSampling2D(size=2)(gen3)
    gen4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen4)
    gen4 = Activation('relu')(gen4)
    
    # upsampling
    gen5 = UpSampling2D(size=2)(gen4)
    gen5 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen5)
    gen5 = Activation('relu')(gen5)
    
    # conv layer at the output
    gen6 = Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(gen5)
    output = Activation('tanh')(gen6)
    
    # model 
    model = Model(inputs=[input_layer], outputs=[output], name='generator')

    return model
  
generator = build_generator()

def build_discriminator():
    
    # define hyperparameters
    leakyrelu_alpha = 0.2
    momentum = 0.8
    
    # the input is the HR shape
    input_shape = (256, 256, 3)
    
    # input layer for discriminator
    input_layer = Input(shape=input_shape)
    
    # 8 convolutional layers with batch normalization  
    dis1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer)
    dis1 = LeakyReLU(alpha=leakyrelu_alpha)(dis1)

    dis2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(dis1)
    dis2 = LeakyReLU(alpha=leakyrelu_alpha)(dis2)
    dis2 = BatchNormalization(momentum=momentum)(dis2)

    dis3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(dis2)
    dis3 = LeakyReLU(alpha=leakyrelu_alpha)(dis3)
    dis3 = BatchNormalization(momentum=momentum)(dis3)

    dis4 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(dis3)
    dis4 = LeakyReLU(alpha=leakyrelu_alpha)(dis4)
    dis4 = BatchNormalization(momentum=0.8)(dis4)

    dis5 = Conv2D(256, kernel_size=3, strides=1, padding='same')(dis4)
    dis5 = LeakyReLU(alpha=leakyrelu_alpha)(dis5)
    dis5 = BatchNormalization(momentum=momentum)(dis5)

    dis6 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(dis5)
    dis6 = LeakyReLU(alpha=leakyrelu_alpha)(dis6)
    dis6 = BatchNormalization(momentum=momentum)(dis6)

    dis7 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(dis6)
    dis7 = LeakyReLU(alpha=leakyrelu_alpha)(dis7)
    dis7 = BatchNormalization(momentum=momentum)(dis7)

    dis8 = Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(dis7)
    dis8 = LeakyReLU(alpha=leakyrelu_alpha)(dis8)
    dis8 = BatchNormalization(momentum=momentum)(dis8)
    
    # fully connected layer 
    dis9 = Dense(units=1024)(dis8)
    dis9 = LeakyReLU(alpha=0.2)(dis9)
    
    # last fully connected layer - for classification 
    output = Dense(units=1, activation='sigmoid')(dis9)   
    
    model = Model(inputs=[input_layer], outputs=[output], name='discriminator')
    
    return model
  
discriminator = build_discriminator()
discriminator.trainable = True
discriminator.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])
# VGG19_base = VGG19(weights="imagenet")
# def build_VGG19():
    
#     input_shape = (280, 220, 3)
#     VGG19_base.outputs = [VGG19_base.get_layer('block5_conv2').output]
#     input_layer = Input(shape=input_shape)
#     features = VGG19_base(input_layer)
#     model = Model(inputs=[input_layer], outputs=[features])
    
#     return model
# fe_model = build_VGG19()
# fe_model.trainable = False
# fe_model.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

# def build_adversarial_model(generator, discriminator, feature_extractor):
def build_adversarial_model(generator, discriminator):
    
    # input layer for high-resolution images
    input_high_resolution = Input(shape=high_resolution_shape)

    # input layer for low-resolution images
    input_low_resolution = Input(shape=low_resolution_shape)

    # generate high-resolution images from low-resolution images
    generated_high_resolution_images = generator(input_low_resolution)

    # extract feature maps from generated images
#     features = feature_extractor(generated_high_resolution_images)
    
    # make a discriminator non-trainable 
    discriminator.trainable = False
    discriminator.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

    # discriminator will give us a probability estimation for the generated high-resolution images
    probs = discriminator(generated_high_resolution_images)

    # create and compile 
#     adversarial_model = Model([input_low_resolution, input_high_resolution], [probs, features])
    adversarial_model = Model([input_low_resolution, input_high_resolution])
    adversarial_model.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=common_optimizer)
    
    return adversarial_model

# adversarial_model = build_adversarial_model(generator, discriminator, fe_model)
adversarial_model = build_adversarial_model(generator, discriminator)

# initialize 

losses = {"d_history":[], "g_history":[]}
psnr = {'psnr_quality': []}
ssim = {'ssim_quality': []}

# training loop

for epoch in range(epochs):

    d_history = []
    g_history = []
    
#     image_list = get_train_images(data_path)
    
    """
    Train the discriminator network
    """
    
    hr_images, lr_images = sample_images(allimagepaths_res, 
                                         batch_size=batch_size,
                                         low_resolution_shape=low_resolution_shape,
                                         high_resolution_shape=high_resolution_shape)
    
    
    # normalize the images
    hr_images = hr_images / 127.5 - 1.
    lr_images = lr_images / 127.5 - 1.
    
    # generate high-resolution images from low-resolution images
    generated_high_resolution_images = generator.predict(lr_images)
    
    # generate a batch of true and fake labels 
    real_labels = np.ones((batch_size, 16, 16, 1))
    fake_labels = np.zeros((batch_size, 16, 16, 1))
    
 
    d_loss_real = discriminator.train_on_batch(hr_images, real_labels)
    d_loss_real =  np.mean(d_loss_real)
    d_loss_fake = discriminator.train_on_batch(generated_high_resolution_images, fake_labels)
    d_loss_fake =  np.mean(d_loss_fake)
    
    # calculate total loss of discriminator as average loss on true and fake labels
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    losses['d_history'].append(d_loss)
   

    """
        Train the generator network
    """
      
    # sample a batch of images    
    hr_images, lr_images = sample_images(allimagepaths_res, 
                                         batch_size=batch_size,
                                         low_resolution_shape=low_resolution_shape,
                                         high_resolution_shape=high_resolution_shape)
    
    
    # normalize the images
    hr_images = hr_images / 127.5 - 1.
    lr_images = lr_images / 127.5 - 1.
    
    
    
    # extract feature maps for true high-resolution images
#     image_features = fe_model.predict(hr_images)


    
    # train the generator
#     g_loss = adversarial_model.train_on_batch([lr_images, hr_images],
#                                                [real_labels, image_features])
    g_loss = adversarial_model.train_on_batch([lr_images, hr_images])
    
    losses['g_history'].append(0.5 * (g_loss[1]))
    
    
    
    # calculate the psnr  
    ps = compute_psnr(hr_images, generated_high_resolution_images) 
    psnr['psnr_quality'].append(ps)
            
    # calculate the ssim 
    ss = compute_ssim(hr_images, generated_high_resolution_images)   
    ssim['ssim_quality'].append(ss)

    
  
    """
        save and print image samples
    """
    
    if epoch % 500 == 0:
        
        hr_images, lr_images = sample_images(allimagepaths_res, 
                                             batch_size=batch_size,
                                             low_resolution_shape=low_resolution_shape,
                                             high_resolution_shape=high_resolution_shape)
    
    
        # normalize the images
        hr_images = hr_images / 127.5 - 1.
        lr_images = lr_images / 127.5 - 1.
    
    
        generated_images = generator.predict_on_batch(lr_images)
    
        for index, img in enumerate(generated_images):
            if index < 3:   # comment this line to display all the images
                save_images(hr_images[index], lr_images[index], img,
                            path="/kaggle/working/img_{}_{}".format(epoch, index))  
              
# plots - post training

plot_loss(losses)
plot_psnr(psnr)
plot_ssim(ssim)
# save model weights

generator.save_weights("/kaggle/working/r_generator.h5")
discriminator.save_weights("/kaggle/working/r_discriminator.h5")
