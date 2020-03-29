import numpy as np 
import matplotlib.pyplot as plt 
from random import randint 
from keras import backend as K 
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Softmax, Dropout, Flatten 
from keras.models import Model 
from keras.datasets import fashion_mnist
from keras.callbacks import TensorBoard 
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.losses import categorical_crossentropy
from sklearn.metrics import classification_report
from keras.utils import plot_model
from keras.layers.normalization import BatchNormalization

num_classes=10
(X_train, y_train), (X_test, y_test) =fashion_mnist.load_data() 

input_img = Input(shape = (28, 28, 1))

X_train = X_train.astype('float32') / 255
X_train = np.reshape(X_train, (len(X_train), 28, 28, 1)) 
X_test = X_test.astype('float32') / 255
X_test = np.reshape(X_test, (len(X_test), 28, 28, 1)) 
# data for encoder
train_X,valid_X,train_ground,valid_ground = train_test_split(X_train,
                                                             X_train,
                                                             test_size=0.2,
                                                             random_state=13)



encoding_conv_layer_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
encoding_pooling_layer_1 = MaxPooling2D((2, 2), padding='same')(encoding_conv_layer_1)
encoding_conv_layer_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoding_pooling_layer_1)
encoding_pooling_layer_2 = MaxPooling2D((2, 2), padding='same')(encoding_conv_layer_2)
encoding_conv_layer_3 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoding_pooling_layer_2)
encode_layer = MaxPooling2D((2, 2), padding='same')(encoding_conv_layer_3)
        
# Decoding
decodging_conv_layer_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(encode_layer)
decodging_upsampling_layer_1 = UpSampling2D((2, 2))(decodging_conv_layer_1)
decodging_conv_layer_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(decodging_upsampling_layer_1)
decodging_upsampling_layer_2 = UpSampling2D((2, 2))(decodging_conv_layer_2)
decodging_conv_layer_3 = Conv2D(64, (3, 3), activation='relu')(decodging_upsampling_layer_2)
decodging_upsampling_layer_3 = UpSampling2D((2, 2))(decodging_conv_layer_3)
output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decodging_upsampling_layer_3)

autoencoder = Model(input_img, output_layer)
autoencoder.summary()

autoencoder.compile(optimizer=Adam(1e-3), loss='mean_squared_error', metrics=['accuracy', 'mse'])

autoencoder.fit(train_X, train_ground,epochs=2,batch_size=5,shuffle=True,validation_data=(valid_X, valid_ground))

# to the last layer (should contain labels)
# ---part2 for last layer (softmax)

train_X,valid_X,train_label,valid_label = train_test_split(X_train,y_train,test_size=0.2,random_state=13)
num_classes=10

dp = Dropout(0.25)(encode_layer)
flat = Flatten()(dp)

den = Dense(64, activation='relu')(flat)
output = Dense(num_classes, activation='softmax')(den)
autoencoder = Model(input_img,output)
#plot_model(autoencoder, 'ae.png', show_shapes=True)

autoencoder.compile(optimizer=Adam(1e-3), loss='mean_squared_error', metrics=['accuracy', 'mse'])
y_train =to_categorical(train_label, num_classes)
y_test = to_categorical(valid_label, num_classes)
autoencoder.fit(train_X,y_train,epochs=2,batch_size=5, verbose=1,validation_data=(valid_X, y_test))

score = autoencoder.evaluate(valid_X, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# making a prediction
autoencoder.save('autoencoder-betteresult.h5')
