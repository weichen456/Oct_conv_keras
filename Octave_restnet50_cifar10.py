from Octave_conv import *
import keras
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau,EarlyStopping
import numpy as np
import os
from keras.layers import Conv2D,Activation,BatchNormalization,AveragePooling2D
from utils import compose
from utils import reader_reator
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Flatten,Dense

#training parameters
batch_size=32
epoch=100
num_class=10
alpha=0.25
k_sec={2:3,
       3:4,
       4:6,
       5:3
       }

#load cifar10 data
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

#input image demensions

input_shape=x_train.shape[1:]

#Normalize the data
x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

#Subtract the pixel mean

x_train_mean=np.mean(x_train,axis=0)
x_train-=x_train_mean
x_test-=x_train_mean

#Conver class vertors to binary class matrices
y_train=keras.utils.to_categorical(y_train,num_class)
y_test=keras.utils.to_categorical(y_test,num_class)

def conv_BN_AC(data,num_filter,kernel,stride=(1,1),name=None,padding='same'):
    return compose(Conv2D(num_filter,
                          kernel_size=kernel,
                          strides=stride,
                          name=name,
                          padding=padding,
                          use_bias=False),
                          BatchNormalization(momentum=0.9),
                          Activation('relu'))(data)

def oct_restnet50(input_shapes):
    '''
    relize the restnet with octave
    :param input_shape:tuple the input data shape
    :return:keras Model
    '''

    inputs=Input(shape=input_shapes)
    #conv1
    #conv1=ZeroPadding2D(padding=(3,3))(image_shape)

    conv1=conv_BN_AC(data=inputs,
                     num_filter=64,
                     kernel=(7,7),
                     name='conv1',
                     stride=(1,1),
                     padding='same')
    pool1=MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='pool1')(conv1)

    #conv2
    num_in=32
    num_mid=64
    num_out=256
    i=1
    hf_conv2_x, lf_conv2_x = Residual_unit_first(
        data=pool1,
        alpha=alpha,
        num_in=(num_in if i == 1 else num_out),
        num_mid=num_mid,
        num_out=num_out,
        name=('conv2_B%02d' % i),
        first_block=(i == 1),
        stride = ((1, 1) if (i == 1) else (1,1)) )

    for i in range(2, k_sec[2] + 1):

        hf_conv2_x, lf_conv2_x = Residual_unit(
            hf_data=( hf_conv2_x),
            lf_data=( lf_conv2_x),
            alpha=alpha,

            num_mid=num_mid,
            num_out=num_out,
            name=('conv2_B%02d' % i),
            first_block=(i == 1),
            stride = ((1, 1) if (i == 2) else (1,1)) )

    #conv3
    num_mid=int(num_mid*2)
    num_out=int(num_out*2)
    for i in range(1, k_sec[3] + 1):
        hf_conv3_x, lf_conv3_x = Residual_unit(
            hf_data=(hf_conv2_x if i == 1 else hf_conv3_x),
            lf_data=(lf_conv2_x if i == 1 else lf_conv3_x),
            alpha=alpha,
            num_mid=num_mid,
            num_out=num_out,
            name=('conv3_B%02d' % i),
            first_block=(i == 1),
            stride=((2, 2) if (i == 1) else (1, 1)))

    #conv4
    num_mid = int(num_mid * 2)
    num_out = int(num_out * 2)
    for i in range(1, k_sec[4] + 1):
        hf_conv4_x, lf_conv4_x = Residual_unit(
            hf_data=(hf_conv3_x if i == 1 else hf_conv4_x),
            lf_data=(lf_conv3_x if i == 1 else lf_conv4_x),
            alpha=alpha,
            num_mid=num_mid,
            num_out=num_out,
            name=('conv4_B%02d' % i),
            first_block=(i == 1),
            stride=((2, 2) if (i == 1) else (1, 1)))

    #conv5
    num_mid = int(num_mid * 2)
    num_out = int(num_out * 2)
    i = 1
    conv5_x = Residual_unit_last(
        hf_data=(hf_conv4_x  ),
        lf_data=(lf_conv4_x  ),
        alpha=alpha,
        num_mid=num_mid,
        num_out=num_out,
        name=('conv5_B%02d' % i),
        first_block=(i == 1),
        stride=((2, 2) if (i == 1) else (1, 1)))

    for i in range(2, k_sec[5] + 1):
        conv5_x = Residual_unit_norm(data=( conv5_x),
                                     num_mid=num_mid,
                                     num_out=num_out,
                                     alpha=alpha,
                                     name=('conv5_B%02d' % i),
                                     first_block=(i == 1),
                                     stride=((2, 2) if (i == 1) else (1, 1)))
    outputs=AveragePooling2D(name='global-pool')(conv5_x)

    outputs=Flatten()(outputs)
    outputs=Dense(num_class,
                  activation='softmax',
                  kernel_initializer='he_normal')(outputs)

    #return Model
    inputs=inputs
    model=Model(inputs=inputs,outputs=outputs)
    return model

def main():
    '''
    define main function

    '''
    model=oct_restnet50(input_shapes=input_shape)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])
    model.summary()

    #Prepare model saving directory
    save_dir=os.path.join(os.getcwd(),'save_model')
    model_name='cifar_model.{epoch:03d}.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    file_path=os.path.join(save_dir,model_name)

    #Prepare callbacks for model saving and for learning rate adjustment
    checkpoint=ModelCheckpoint(filepath=file_path,
                               monitor='val_acc',
                               verbose=1,
                               period=5,
                               save_best_only=True)
    reduce_lr=ReduceLROnPlateau(monitor='val_acc',
                                verbose=1,
                                factor=0.1,
                                patience=6)
    early_stopping=EarlyStopping(monitor='val_acc',
                                 min_delta=0,
                                 patience=10,
                                 verbose=1)
    callbacks=[checkpoint,reduce_lr,early_stopping]

    model.fit(x_train,y_train,
              batch_size=batch_size,
              epochs=epoch,
              validation_data=(x_test,y_test),
              shuffle=True,
              callbacks=callbacks
              )
    scores=model.evaluate(x_test,y_test,verbose=1)
    print('test loss:',scores[0])
    print('test accuracy:',scores[1])

if __name__=="__main__":
    main()



