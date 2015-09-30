#!/usr/bin/env python

import os
import numpy as np
import amazon
import neon
from neon.data import DataIterator
from neon import NervanaObject
from neon.backends import gen_backend
from neon.layers import Affine, GeneralizedCost, Conv, Pooling, Dropout, LSTM
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary, CrossEntropyMulti, Misclassification, Softmax, Tanh
from neon.initializers import Gaussian, Constant, Uniform
from neon.models import Model
from neon.callbacks.callbacks import Callbacks

import sys
sys.path.append('/root/data/pcallier/sunny-side-up/')
import src.datasets.amazon_neon as amazon_neon

num_epochs = 50
text_length = 1014
vocab_size = len(amazon.vocabulary)
frame_size=256
batch_size=64

amazon.truncate_length = text_length
amazon.num_entries_max = 10000

be = gen_backend(backend="gpu",batch_size=batch_size,rng_seed=888)

def amazon_to_dataiterator(am):
    amazon_train = list(am)
    amazon_in, amazon_out = zip(*amazon_train)
    #print amazon_in[0].shape
    amazon_in_2 = np.array(amazon_in).reshape(len(amazon_in), -1)
    amazon_out_2 = np.ndarray((len(amazon_out), 2))
    amazon_out_2[:, :] = amazon_out
    #print amazon_in_2.shape
    #print amazon_out_2.shape
    data_it = DataIterator(amazon_in_2, amazon_out_2, 
                           lshape=(1, text_length, vocab_size),
                           make_onehot=False)
    return data_it


#amazon_gen_train = amazon.load_character_encoded_data("/root/data/amazon/aggressive_dedup.json", record_limit=amazon.num_entries_max)
#amazon_train = amazon_to_dataiterator(amazon_gen_train)
#amazon_gen_test = amazon.load_character_encoded_data("/root/data/amazon/aggressive_dedup.json",
#                                                    which_set="test", record_limit=amazon.num_entries_max)
#amazon_test = amazon_to_dataiterator(amazon_gen_test)

am_train = "/root/data/amazon/reviews_Sports_and_Outdoors.json.gz"
am_test = "/root/data/amazon/reviews_Health_and_Personal_Care.json.gz"
h5file_train_path = '/root/data/pcallier/amazon/temp_train.hd5'
h5file_test_path = '/root/data/pcallier/amazon/temp_test.hd5'
azbw_train = amazon_neon.AmazonBatchWriter(am_train ,h5file_train_path)
azbw_test = amazon_neon.AmazonBatchWriter(am_test ,h5file_test_path)

amazon_train = amazon_neon.AmazonDataIterator(h5file_train_path)
amazon_test = amazon_neon.AmazonDataIterator(h5file_test_path)

# setup weight initialization function
init_norm = Gaussian(loc=0.0, scale=0.01)
init1 = Gaussian(scale=0.09)
init_unif = Uniform(low=-0.08, high=0.08)

relu = Rectlin()
sigm = Logistic()
soft = Softmax()

#model (CNN)
def am_CNN():
    layers = []
    layers.append(Conv((vocab_size, 7, frame_size), 
                       init=init1, bias=Constant(0), activation=relu))
    layers.append(Pooling((1,3)))

    layers.append(Conv((1, 7, frame_size), 
                       init=init1, bias=Constant(0), activation=relu))
    layers.append(Pooling((1,3)))

    layers.append(Conv((1, 3, frame_size), 
                      init=init1, bias=Constant(0), activation=relu))

    layers.append(Conv((1, 3, frame_size), 
                       init=init1, bias=Constant(0), activation=relu))

    layers.append(Conv((1, 3, frame_size), 
                       init=init1, bias=Constant(0), activation=relu))

    layers.append(Conv((1, 3, frame_size), 
                       init=init1, bias=Constant(0), activation=relu))
    layers.append(Pooling((1,3)))

    layers.append(Affine(nout=1024, init=init1, bias=Constant(1), activation=relu))
    layers.append(Dropout(keep=0.5))

    layers.append(Affine(nout=1024, init=init1, bias=Constant(1), activation=relu))
    layers.append(Dropout(keep=0.5))

    layers.append(Affine(nout=1, init=init1, bias=Constant(0), activation=sigm))
    return layers

def am_LSTM(hidden_size=100):
    rlayer = LSTM(hidden_size, init1, Logistic(), Tanh())
    layers = [ rlayer,
               Affine(nout=1, init=init1, bias=init_unif, activation=Logistic()) ]
    return layers

# setup cost function as CrossEntropy
cost = GeneralizedCost(costfunc=CrossEntropyBinary())
optimizer = GradientDescentMomentum(0.001, momentum_coef=0.9)
mlp = Model(layers=am_CNN())
callbacks = Callbacks(mlp, amazon_train, valid_set=amazon_test, valid_freq=1, progress_bar=True)
callbacks.add_serialize_callback(1, "/root/data/pcallier/amazon/saved_model.pkl")

mlp.fit(amazon_train, optimizer=optimizer, num_epochs=num_epochs, cost=cost, callbacks=callbacks)

# do eval stuff
print 'Misclassification error = %.1f%%' % (mlp.eval(amazon_test, metric=Misclassification())*100)

