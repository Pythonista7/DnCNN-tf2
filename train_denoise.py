"""
Learnt a lot about building the input pipeline in tf2 from here
https://financial-engineering.medium.com/tensorflow-2-0-load-images-to-tensorflow-897b8b067fc2

"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from DnCNN import DnCNN

data_dir = "./BSDS300/images/"

BATCH_SIZE = 32
IMG_HEIGHT = 321
IMG_WIDTH = 481

CHANNELS = 1


def decode_img(img,channels):
    img = tf.image.decode_jpeg(img, channels=channels) #color/greyscale images
    img = tf.image.convert_image_dtype(img, tf.float32) 
    #convert unit8 tensor to floats in the [0,1]range
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT]) 

def process_path(file_path):
    clean_img = tf.io.read_file(file_path)
    clean_img = decode_img(clean_img,1) #Setting CHANNELS=1 
    noisy_img = clean_img + np.random.normal(0,25/255.0,size=clean_img.shape)
    return noisy_img, clean_img



# Incase u dont have the dataset you can get it by running the following 2 commands in the dir of this file
# wget https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz
# tar -xvzf BSDS300-images.tgz 

# Setup data by reading files for input and testing 
train_list_ds = tf.data.Dataset.list_files(str(data_dir+'train/*'))
test_list_ds = tf.data.Dataset.list_files(str(data_dir+'test/*'))

# Load up the files for the model
train_ds = train_list_ds.map(process_path)
test_ds = test_list_ds.map(process_path)

# Build the model , compile and fit it to the data
model = DnCNN(depth=5).model()

opt = tf.keras.optimizers.Adam(lr=0.0001,beta_1=0.9)
loss_fn = tf.losses.mse

model.compile(optimizer= opt,loss=loss_fn,metrics=["accuracy"])

model.fit(train_ds,epochs=1,batch_size=32)

# Lets now see how the model performs
test_noise,test_clean = next(iter(test_ds)) #Picking up a sample from test set
# Make a prediction using the model
prediction = model.predict(test_noise)

#Plotting out all the 3 images
fig, (ax1, ax2,ax3) = plt.subplots(1,3,figsize=(7,7))
fig.suptitle('1) Clean img 2) input noisy img  3) Model output' )
ax1.imshow( test_clean.numpy().reshape((test_noise.shape[0],test_noise.shape[1])) , cmap = 'gray')
ax2.imshow(test_noise.numpy().reshape((test_noise.shape[0],test_noise.shape[1])) ,  cmap='gray')
ax3.imshow(prediction.reshape((prediction.shape[0],prediction.shape[1])),cmap='gray', )
