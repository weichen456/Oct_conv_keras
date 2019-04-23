from functools import reduce
import pickle
import numpy as np
import os
from keras.utils import to_categorical
'''
compose function

'''
'''with open('cifar10/data_batch_1','rb') as f:
    datadict=pickle.load(f,encoding='bytes')
    x=datadict.keys()
    x=datadict[b'data']
    #y=datadict['labels']
    print(x)'''

def compose(*funcs):
    '''
    compose many functions from left to right
    :param funcs: functons
    :return: composed functions

    '''
    if funcs:
        return reduce(lambda f,g:lambda *a,**kw:f(g(*a,**kw)),funcs)
    else:
        raise ValueError("Compositon of empty sequence is not supported")
def reader_reator(root,istrain=True,cycle=True,subtract_pixel_mean=True):
    def load_cifar_batch(filename):
        '''

        load singel batch of cifar10
        :param filename: string,the path of single batch
        :return: data,label


        '''
        with open (filename,'rb') as f:
            datadict=pickle.load(f,encoding='bytes')
            X=datadict[b'data']
            Y=datadict[b'labels']

            #(N C H W) transpose to (N H W C)
            if subtract_pixel_mean:
              X=X.reshape(10000,3,32,32).transpose(0,2,3,1).astype('float32')
              X=np.array(X)
              X=X/255.0
              Y=np.array(Y)
              Y=to_categorical(Y,10)
              X_mean=np.mean(X)
              X-=X_mean

              return X,Y
            else:
                X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32')
                Y = np.array(Y)
                return X,Y
    while True:
          X = []
          Y = []
          if istrain:
            for b in range(1,6):
                   f=os.path.join(root,'data_batch_{}'.format(b))
                   image,labels=load_cifar_batch(f)
                   X.append(image)
                   Y.append(labels)
            X=np.concatenate(X)
            Y=np.concatenate(Y)

            yield [X,Y]

          if not cycle:
                break
          else:
                  f=os.path.join(root,'test_batch')
                  X,Y=load_cifar_batch(f)
                  yield [X,Y]

          if not cycle:
                break

def data_generator_wraper(root,istrain=True,cycle=False,subtract_pixel_mean=True):
    return reader_reator(root,istrain=True,cycle=False,subtract_pixel_mean=True)

