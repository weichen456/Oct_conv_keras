from keras.layers import AveragePooling2D,BatchNormalization,Conv2D,Activation
from keras.layers import ZeroPadding2D,UpSampling2D
from functools import wraps
from utils import compose
from keras.layers import Add

#--------------------------
#Elements

@wraps(Conv2D)
def conv2D(*args,**kwargs):
    conv2D_kwargs={'use_bias':False}
    Conv2D_kwargs={'kernel_initializer':'he_normal'}
    conv2D_kwargs.update(kwargs)

    return Conv2D(*args,**conv2D_kwargs)


def BN_AC(*args,**kwargs):
    return compose(
                   BatchNormalization(momentum=0.9),
                   Activation('relu'))

def firstOctConv(input,alpha_out,ch_in,ch_out,name,kernel=(1,1),padding='same',stride=(1,1)):

    #hf_ch_in=int(ch_in*(1-alpha_in))
    hf_ch_out=int(ch_out*(1-alpha_out))

    #lf_ch_in=ch_in-hf_ch_in
    lf_ch_out=ch_out-hf_ch_out

    hf_data=input

    if stride==(2,2):
        hf_data=AveragePooling2D(
                                 strides=(2,2),
                                 )(hf_data)
    hf_conv=conv2D(hf_ch_out,
                   kernel_size=kernel,
                   padding=padding,
                   strides=stride,
                   )(hf_data)
    hf_pool=AveragePooling2D(
                             strides=(2,2),
                             )(hf_data)
    hf_pool_conv=conv2D(filters=lf_ch_out,
                        kernel_size=kernel,
                        padding=padding,
                        strides=stride
                        )(hf_pool)

    out_h=hf_conv
    out_l=hf_pool_conv

    return out_h,out_l

def lastOctConv(hf_data,lf_data,alpha_out,ch_out,name,kernel=(1,1),padding='same',stride=(1,1)):
    hf_ch_out=int(ch_out*(1-alpha_out))

    if stride==(2,2):
        hf_data=AveragePooling2D(strides=(2,2))(hf_data)
    hf_conv=conv2D(hf_ch_out,kernel_size=kernel,padding=padding,strides=(1,1))(hf_data)
    lf_conv=conv2D(hf_ch_out,kernel_size=kernel,padding=padding,strides=(1,1))(lf_data)
    out_h=Add()([hf_conv,lf_conv])

    return out_h

def OctConv(hf_data,lf_data,alpha_out,ch_out,name,kernel=(1,1),padding='same',stride=(1,1)):
    hf_ch_out=int(ch_out*(1-alpha_out))

    lf_ch_out=ch_out-hf_ch_out

   # print('1',hf_data.shape,'2',lf_data.shape)
    #print(stride)

    if stride==(2,2):
        hf_data=AveragePooling2D(
                                 strides=(2,2),
                                 )(hf_data)

    hf_conv=conv2D(hf_ch_out,
                   kernel_size=kernel,
                   padding=padding,
                   )(hf_data)
    hf_pool=AveragePooling2D(
                             strides=(2,2),
                             )(hf_data)
    hf_pool_conv=conv2D(lf_ch_out,
                        kernel_size=kernel,
                        padding=padding,
                       )(hf_pool)

    lf_conv=conv2D(hf_ch_out,
                   kernel_size=kernel,
                   padding='same',
                   strides=(1,1),
                   )(lf_data)

    if stride==(2,2):
        lf_upsample=lf_conv
        lf_down=AveragePooling2D(strides=(2,2),
                                 )(lf_data)
    else:
        lf_upsample=UpSampling2D(2)(lf_conv)
        lf_down=lf_data
    lf_down_conv=conv2D(lf_ch_out,
                        kernel_size=kernel,
                        padding=padding,
                       )(lf_down)
   # print(hf_conv.shape,lf_upsample.shape)
    out_h=Add()([hf_conv,lf_upsample])
    out_l=Add()([hf_pool_conv,lf_down_conv])

    return out_h,out_l

def firstOctConv_BN_AC(data,alpha,num_filters_in,num_filters_out,kernel,padding,stride=(1,1),name=None):
     hf_data,lf_data=firstOctConv(data,alpha,ch_in=num_filters_in,ch_out=num_filters_out,name=name,kernel=kernel,padding=padding,stride=stride)
     out_hf=BN_AC()(hf_data)
     out_lf=BN_AC()(lf_data)

     return out_hf,out_lf

def lastOctConv_BN_AC(hf_data,lf_data,alpha,num_filter_out,kernel,padding,stride=(1,1),name=None):
     conv=lastOctConv(hf_data=hf_data,lf_data=lf_data,alpha_out=alpha,kernel=kernel,ch_out=num_filter_out,name=name,padding=padding,stride=stride)
     out=BN_AC()(conv)

     return out

def octConv_BN_AC(hf_data,lf_data,alpha,num_filster_out,kernel,padding,stride=(1,1),name=None):
    hf_data,lf_data=OctConv(hf_data=hf_data,
                            lf_data=lf_data,
                            alpha_out=alpha,
                            ch_out=num_filster_out,
                            name=name,
                            kernel=kernel,
                            padding=padding,
                            stride=stride
                            )
    out_hf=BN_AC()(hf_data)
    out_lf=BN_AC()(lf_data)
    return out_hf,out_lf

def firstOctConc_BN(data,alpha,num_filter_out,num_filter_in ,kernel,padding, stride=(1,1),name=None):
    hf_data,lf_data=firstOctConv(data,alpha_out=alpha,ch_out=num_filter_out,ch_in=num_filter_in,name=name,kernel=kernel,padding=padding,stride=stride)
    out_hf=BatchNormalization(momentum=0.9)(hf_data)
    out_lf=BatchNormalization(momentum=0.9)(lf_data)

    return out_hf,out_lf

def lastOctConv_BN(hf_data,lf_data,alpha,num_filter_out,kernel,padding,stride=(1,1),name=None):
    conv=lastOctConv(hf_data=hf_data,lf_data=lf_data,alpha_out=alpha,ch_out=num_filter_out,name=name,kernel=kernel,padding=padding,stride=stride)
    out=BatchNormalization(momentum=0.9)(conv)

    return out

def octConv_BN(hf_data,lf_data,alpha,num_filter_out,kernel,padding,stride=(1,1),name=None):

    hf_data, lf_data = OctConv(hf_data=hf_data,
                               lf_data=lf_data,
                               alpha_out=alpha,
                               ch_out=num_filter_out,
                               name=name,
                               kernel=kernel,
                               padding=padding,
                               stride=stride
                               )
    out_hf=BatchNormalization(momentum=0.9)(hf_data)
    out_lf=BatchNormalization(momentum=0.9)(lf_data)

    return out_hf,out_lf


'''
residual_unit

'''

def Residual_unit_norm(data,alpha,num_mid,num_out,name,first_block=False,stride=(1,1)):
    conv_m1=compose(conv2D(num_mid,kernel_size=(1,1),padding='same'),
                    BatchNormalization(momentum=0.9),
                    Activation('relu'))(data)
    conv_m2=compose(conv2D(num_mid,kernel_size=(3,3),padding='same',strides=stride),
                    BatchNormalization(momentum=0.9),
                    Activation('relu'))(conv_m1)
    conv_m3=compose(conv2D(int(num_out*(1-alpha)),kernel_size=(1,1),padding='same'),
                    BatchNormalization(momentum=0.9),
                    Activation('relu'))(conv_m2)

    if first_block:
            data = compose(
                           conv2D(int(num_out*(1-alpha)), kernel_size=(3, 3), padding='same', strides=stride),
                           BatchNormalization(momentum=0.9))(data)

    outputs=Add()([data,conv_m3])
    return Activation('relu')(outputs)

def Residual_unit_last(hf_data,lf_data,alpha,num_mid,num_out,name=None,first_block=False,stride=(1,1),g=1):
    hf_data_m,lf_data_m=octConv_BN_AC(hf_data=hf_data,lf_data=lf_data,alpha=alpha,num_filster_out=num_mid,kernel=(1,1),padding='same')
    conv_m2=lastOctConv_BN_AC(hf_data=hf_data_m,lf_data=lf_data_m,alpha=alpha,num_filter_out=num_mid,kernel=(3,3),padding='same',stride=stride)
    conv_m3=compose(conv2D(int(num_out*(1-alpha)),kernel_size=(1,1),padding='same'),
                    BatchNormalization(momentum=0.9))(conv_m2)

    if first_block:


        data=lastOctConv_BN(hf_data=hf_data,
                            lf_data=lf_data,
                            alpha=alpha,
                            num_filter_out=num_out,
                            kernel=(1,1),
                            padding='same',
                            stride=stride)
    outputs=Add()([data,conv_m3])
    outputs=Activation('relu')(outputs)
    return outputs

def Residual_unit_first(data,alpha,num_in,num_mid,num_out,name,first_block=False,stride=(1,1)):

    hf_data_m,lf_data_m=firstOctConv_BN_AC(data=data ,
                                           alpha=alpha,
                                           num_filters_in=num_in,
                                           num_filters_out=num_mid,
                                           kernel=(1,1),
                                           padding='same',
                                           )

    hf_data_m,lf_data_m=octConv_BN_AC(hf_data=hf_data_m,
                                      lf_data=lf_data_m,
                                      alpha=alpha,
                                      num_filster_out=num_mid,
                                      kernel=(3,3),
                                      padding='same',
                                      stride=stride
                                      )
    #print(hf_data_m.shape, lf_data_m.shape)

    hf_data_m,lf_data_m=octConv_BN(hf_data=hf_data_m,
                                   lf_data=lf_data_m,
                                   alpha=alpha,
                                   num_filter_out=num_out,
                                   kernel=(1,1),
                                   padding='same',
                                   name=('{}_conv_m3'.format(name))
                                   )


    if first_block:

        hf_data, lf_data = firstOctConc_BN(data=data,
                                               alpha=alpha,
                                               num_filter_in=num_mid,
                                               num_filter_out=num_out,
                                               kernel=(1, 1),
                                               padding='same',
                                               stride=stride
                                               )


    hf_outputs=Add()([hf_data,hf_data_m])
    lf_outputs=Add()([lf_data,lf_data_m])

    hf_outputs=Activation('relu')(hf_outputs)
    lf_outputs=Activation('relu')(lf_outputs)

    return hf_outputs,lf_outputs

def Residual_unit(hf_data,lf_data,alpha,num_mid,num_out,name,first_block=False,stride=(1,1)):


    hf_data_m,lf_data_m=octConv_BN_AC(hf_data=hf_data,
                                      lf_data=lf_data,
                                      alpha=alpha,
                                      num_filster_out=num_mid,
                                      kernel=(1,1),
                                      padding='same'
                                      )
    hf_data_m,lf_data_m=octConv_BN_AC(hf_data=hf_data_m,
                                      lf_data=lf_data_m,
                                      alpha=alpha,
                                      num_filster_out=num_mid,
                                      kernel=(3,3),

                                      padding='same',
                                      stride=stride,
                                      name=('{}_conv_m2'.format(name)))
    hf_data_m,lf_data_m=octConv_BN(hf_data=hf_data_m,
                                   lf_data=lf_data_m,
                                   alpha=alpha,
                                   num_filter_out=num_out,
                                   kernel=(1,1),
                                   padding='same'
                                   )

    if first_block:
        hf_data,lf_data=octConv_BN(hf_data=hf_data,
                                   lf_data=lf_data,
                                   alpha=alpha,
                                   num_filter_out=num_out,
                                   kernel=(1,1),
                                   padding='same',
                                   stride=stride
                                   )

    hf_outputs=Add()([hf_data,hf_data_m])
    lf_outputs=Add()([lf_data,lf_data_m])

    hf_outputs=Activation('relu')(hf_outputs)
    lf_outputs=Activation('relu')(lf_outputs)
    return hf_outputs,lf_outputs






