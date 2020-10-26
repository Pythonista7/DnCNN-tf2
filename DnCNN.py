import tensorflow as tf 
"""
https://danijar.com/structuring-models/
"""

class DnCNN(tf.keras.Model):
  
    def __init__(self,depth = 5,grayscale=True):
        super(DnCNN,self).__init__()
        # Network params
        self.channels = 1 if grayscale else 3
        self.depth = depth

    def call(self,input_tensor,training=True):
        # First Convolution Layer with Conv and ReLU
        x = tf.keras.layers.Conv2D(64,(3,3),padding="same",kernel_initializer='Orthogonal')(input_tensor)
        x = tf.keras.activations.relu(x)

        # Add Conv+Batch_Norm+ReLU for layers 2 to (depth-1)
        for _ in range(self.depth - 1):
            x = tf.keras.layers.Conv2D(64,(3,3),padding="same",kernel_initializer='Orthogonal')(x)
            x =  tf.keras.layers.BatchNormalization(epsilon=0.0001)(x,training=training)
            x = tf.keras.activations.relu(x)

        # The final conv layer will use only 1 filter to recontruct the original image
        x = tf.keras.layers.Conv2D(1,(3,3),padding="same",kernel_initializer='Orthogonal')(x)

        # Subtract the predicted noise from the noisy input image
        x = tf.keras.layers.Subtract()([input_tensor,x]) #input - noise

        return x
    
    def model(self):
        # Funtion to build the model
        x = tf.keras.Input(shape=(None,None,self.channels))
        return tf.keras.Model(inputs=[x],outputs= self.call(x) )




# Simple code to instantiate , compile and print the summary of the model architecture 
# if __name__ == "__main__":
#     model = DnCNN(depth=5).model()
#     model.compile(optimizer='Adam',loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=["accuracy"])
#     print(model.summary())
