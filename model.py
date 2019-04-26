# load variables
import numpy as np
import librosa
from librosa.display import specshow
from librosa.util import softmask
import matplotlib.pyplot as plt
import IPython.display as ipd
import sys, os
import types
%matplotlib inline

# load tensorflow and keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Reshape, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, ReLU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence
from tensorflow.keras.losses import mean_squared_error
import tensorflow.keras.backend as K
tf.__version__


# variables
path = '/home/jangryga/source-separation/MSD100/'  # your path to the MSD100
rate = 44100                                       # sampling rate at which songs are loaded
hop_length = 256                                   # hop_length for the fft
n_fft = 1024                                       # window length of the fft
list_titles = os.listdir(os.path.join(os.path.join(path,'Mixtures'),'Dev'))        # train file list with titels
list_titles_test = os.listdir(os.path.join(os.path.join(path,'Mixtures'),'Test'))  # test file list with titles
time_len = 30                                      # time length of a batch
n_frames = 32                                      # number of examples in a single batch
overlap = 15                                       # overlap between examples
normalization = np.sqrt(1024)                      # sqrt of the frame size
scale_mag = 0.3                                    # further normalization
batch = np.empty(shape=(n_frames, int(n_fft/2+1), time_len))    # init empty batch
a,b,c,d,e = (0,0,0,0,0)                            # init variables a,b,c,d,e

# perameters for the NN
steps_per_epoch = 4810                             # total num of batches
n_epochs=1                                         # number of training epochs
number = steps_per_epoch * n_epochs                # total number of batches to load for training
max_queue = 200                                    # how many batches to load into memory at once
multiprocessing = True
validation_steps = 467                             # number of validation batches
n_workers=3
epsilon=1e-8                                       # ...
alpha=0.001                                        # ...
beta=0.01                                          # ...
beta_voc=0.03                                      # ...
rand_num = np.random.uniform(size=(32,513,30,1))

# generator to load tracks for training

def file_gen(iterable):

    saved = []   # list that generator iterates over after first epoch

    for element in range(len(iterable)):

        # load mixture tracks
        path_ = os.path.join(os.path.join(path,'Mixtures'), 'Dev')
        os.chdir(os.path.join(path_,iterable[element]))
        y, _ = librosa.load('mixture.wav', sr=rate)


        # load vocal tracks & possible other tracks
        path_vox = os.path.join(os.path.join(path,'Sources'), 'Dev')
        os.chdir(os.path.join(path_vox,iterable[element]))
        x, _ = librosa.load('vocals.wav', sr=rate)
        w, _ = librosa.load('bass.wav', sr=rate)
        z, _ = librosa.load('drums.wav', sr=rate)
        h, _ = librosa.load('other.wav', sr=rate)
        yield y, x, w, z, h

        # append elements to the saved list
        saved.append(list_titles[element])

        # generator iterates indefinitely over the saved list
    while saved:
        for element in saved:
            path_ = os.path.join(os.path.join(path,'Mixtures'), 'Dev')
            os.chdir(os.path.join(path_,element))
            y, _ = librosa.load('mixture.wav', sr=rate)

            path_vox = os.path.join(os.path.join(path,'Sources'), 'Dev')
            os.chdir(os.path.join(path_vox,element))
            x, _ = librosa.load('vocals.wav', sr=rate)
            w, _ = librosa.load('bass.wav', sr=rate)
            z, _ = librosa.load('drums.wav', sr=rate)
            h, _ = librosa.load('other.wav', sr=rate)

            yield y, x, w, z, h
"""
generator to load files for testing
to do:   code to be optimised to avoid defining seperate generators
         e.g., generator could take test/train as a condition variable

"""

def file_gen_test(iterable):

    saved = []

    for element in range(len(iterable)):
        path_ = os.path.join(os.path.join(path,'Mixtures'), 'Test')
        os.chdir(os.path.join(path_,iterable[element]))
        y, _ = librosa.load('mixture.wav', sr=rate)

        path_vox = os.path.join(os.path.join(path,'Sources'), 'Test')
        os.chdir(os.path.join(path_vox,iterable[element]))
        x, _ = librosa.load('vocals.wav', sr=rate)
        w, _ = librosa.load('bass.wav', sr=rate)
        z, _ = librosa.load('drums.wav', sr=rate)
        h, _ = librosa.load('other.wav', sr=rate)
        yield y, x, w, z, h

        saved.append(list_titles_test[element])

    while saved:
        for element in saved:
            path_ = os.path.join(os.path.join(path,'Mixtures'), 'Test')
            os.chdir(os.path.join(path_,element))
            y, _ = librosa.load('mixture.wav', sr=rate)

            path_vox = os.path.join(os.path.join(path,'Sources'), 'Test')
            os.chdir(os.path.join(path_vox,element))
            x, _ = librosa.load('vocals.wav', sr=rate)
            w, _ = librosa.load('bass.wav', sr=rate)
            z, _ = librosa.load('drums.wav', sr=rate)
            h, _ = librosa.load('other.wav', sr=rate)
            yield y, x, w, z, h

# initialize generators
gen_file = file_gen(list_titles)
gen_file_test = file_gen_test(list_titles_test)


def spectrogram(b):
    # compute a spectrogram of b
    return librosa.stft(b, n_fft=n_fft, hop_length=hop_length)

def mag_phase(b):
    # return matude and phase arrays
    a, b = librosa.magphase(b)
    return a, b

def softmask(x, x_ref):
    # create softmask for the x and x_ref
    return librosa.util.softmask(x,x_ref)


"""
generator that takes files and loads
to do:
- catch when initialised with parameters outside of constraints
- cobmibe to into a single generator+wrapper for testing and training
- return the phase as option
"""
def return_batch(num, a, b, c, d, e):
    """
    a - mix magnitude spectrogram
    b - vocal magnitude spectrogram
    num - number of batches - after it goes to 0, load new files and update itself -> wrapper function
    """
    memory = num
    batch_a = np.empty((32,513,30))
    batch_b = np.empty((32,513,30))
    batch_c = np.empty((32,513,30))
    batch_d = np.empty((32,513,30))
    batch_e = np.empty((32,513,30))

    index = 0
    batch_len = (n_frames-1)*(time_len-overlap) + time_len

    # transform tracks into
    while memory > 0:
        for n in range(32):
            batch_a[n] = a[:,index*batch_len + n*(time_len-overlap): index*batch_len + n*(time_len-overlap)+time_len]
        for n in range(32):
            batch_b[n] = b[:,index*batch_len + n*(time_len-overlap): index*batch_len + n*(time_len-overlap)+time_len]
        for n in range(32):
            batch_c[n] = c[:,index*batch_len + n*(time_len-overlap): index*batch_len + n*(time_len-overlap)+time_len]
        for n in range(32):
            batch_d[n] = d[:,index*batch_len + n*(time_len-overlap): index*batch_len + n*(time_len-overlap)+time_len]
        for n in range(32):
            batch_e[n] = e[:,index*batch_len + n*(time_len-overlap): index*batch_len + n*(time_len-overlap)+time_len]

        yield batch_a, batch_b, batch_c, batch_d, batch_e

        #update memory and index
        memory -= 1
        index +=1

# function that lets the inside generator iterate indefinitely
def wrapper():
    global batch_tr
    try:
        ok = next(batch_tr)
        return ok
    except StopIteration:

        # load new file using gen_file generator
        a, b, c, d, e = next(gen_file)

        # transform audio data into magnitude spectrograms, forgo the phase
        a, _ = mag_phase(spectrogram(a))
        b, _ = mag_phase(spectrogram(b))
        c, _ = mag_phase(spectrogram(c))
        d, _ = mag_phase(spectrogram(d))
        e, _ = mag_phase(spectrogram(e))
        # memory - number of batches in the file
        memory = a.shape[1]/((n_frames-1)*(time_len-overlap) + time_len)
        batch = return_batch(memory, a, b, c, d, e)
        ok = next(batch)
        return ok

# outside generator for fit_model
def gen_train(num_batch):
    while num_batch > 0:

        # call wrapper function that can iterate indefinitely
        x,y,z,w,h = wrapper()

        # reshape the file for the output
        x = np.reshape(x,(32,513,30,1))
        y = np.reshape(y,(32,513,30,1))
        z = np.reshape(z,(32,513,30,1))
        w = np.reshape(w,(32,513,30,1))
        h = np.reshape(h,(32,513,30,1))

        y = np.append(y, z, axis=3)
        y = np.append(y, w, axis=3)
        y = np.append(y, h, axis=3)

        #normalize
        y = y / normalization
        y = scale_mag*y.astype(np.float32)

        x = x / normalization
        x = scale_mag*x.astype(np.float32)

        yield x,y


def return_batch_test(num, a, b, c, d, e):

    memory_test = num
    batch_a = np.empty((32,513,30))
    batch_b = np.empty((32,513,30))
    batch_c = np.empty((32,513,30))
    batch_d = np.empty((32,513,30))
    batch_e = np.empty((32,513,30))
    index_test = 0
    batch_len = (n_frames-1)*(time_len-overlap) + time_len


    while memory_test > 0:
        for n in range(32):
            batch_a[n] = a[:,index_test*batch_len + n*(time_len-overlap): index_test*batch_len + n*(time_len-overlap)+time_len]
        for n in range(32):
            batch_b[n] = b[:,index_test*batch_len + n*(time_len-overlap): index_test*batch_len + n*(time_len-overlap)+time_len]
        for n in range(32):
            batch_c[n] = c[:,index*batch_len + n*(time_len-overlap): index*batch_len + n*(time_len-overlap)+time_len]
        for n in range(32):
            batch_d[n] = d[:,index*batch_len + n*(time_len-overlap): index*batch_len + n*(time_len-overlap)+time_len]
        for n in range(32):
            batch_e[n] = e[:,index*batch_len + n*(time_len-overlap): index*batch_len + n*(time_len-overlap)+time_len]

        yield batch_a, batch_b, batch_c, batch_d, batch_e
        memory_test -= 1
        index_test +=1


def wrapper_test():
    global batch_test
    try:
        ok = next(batch_test)
        return ok
    except StopIteration:
        a, b, c, d, e = next(gen_file_test)
        a, _ = mag_phase(spectrogram(a))
        b, _ = mag_phase(spectrogram(b))
        c, _ = mag_phase(spectrogram(c))
        d, _ = mag_phase(spectrogram(d))
        e, _ = mag_phase(spectrogram(e))
        memory = a.shape[1]/((n_frames-1)*(time_len-overlap) + time_len)
        batch_test = return_batch_test(memory, a, b, c, d, e)
        ok = next(batch_test)
        return ok

def gen_test(num_batch):
    while num_batch > 0:
        # call wrapper function that can iterate indefinitely
        x,y,z,w,h = wrapper_test()

        # reshape the file for the output
        x = np.reshape(x,(32,513,30,1))
        y = np.reshape(y,(32,513,30,1))
        z = np.reshape(z,(32,513,30,1))
        w = np.reshape(w,(32,513,30,1))
        h = np.reshape(h,(32,513,30,1))

        y = np.append(y, z, axis=3)
        y = np.append(y, w, axis=3)
        y = np.append(y, h, axis=3)

        #normalize
        y = y / normalization
        y = scale_mag*y.astype(np.float32)

        x = x / normalization
        x = scale_mag*x.astype(np.float32)

        yield x,y

# initialize inside generators - 0 parameter prompts the except route that loads new data
batch_test = return_batch_test(0, a, b, c, d, e)
batch_tr = return_batch(0, a, b, c, d, e)

# initialize outside generators for fit_model
gen_ts = gen_test(number)
gen_tr = gen_train(number)

#model

inp = Input(shape=(513,30,1),batch_size=32)
layer_conv1 = Conv2D(filters=50, kernel_size=(513,1), padding='valid')(inp)
layer_conv2 = Conv2D(filters=50, kernel_size=(1,15), padding='valid')(layer_conv1)
layer_flat = Flatten()(layer_conv2)
layer_dense = Dense(units=128, activation='relu')(layer_flat)

b1 = Dense(units=int(layer_flat.shape[1]), activation='relu')(layer_dense)
b1 = Reshape(target_shape=(int(layer_conv2.shape[1]),int(layer_conv2.shape[2]),int(layer_conv2.shape[3])))(b1)
b1 = Conv2DTranspose(filters=50, kernel_size=(1,15), padding='valid')(b1)
b1 = Conv2DTranspose(filters=1, kernel_size=(513,1), padding='valid')(b1)

b2 = Dense(units=int(layer_flat.shape[1]), activation='relu')(layer_dense)
b2 = Reshape(target_shape=(int(layer_conv2.shape[1]),int(layer_conv2.shape[2]),int(layer_conv2.shape[3])))(b2)
b2 = Conv2DTranspose(filters=50, kernel_size=(1,15), padding='valid')(b2)
b2 = Conv2DTranspose(filters=1, kernel_size=(513,1), padding='valid')(b2)


b3 = Dense(units=int(layer_flat.shape[1]), activation='relu')(layer_dense)
b3 = Reshape(target_shape=(int(layer_conv2.shape[1]),int(layer_conv2.shape[2]),int(layer_conv2.shape[3])))(b3)
b3 = Conv2DTranspose(filters=50, kernel_size=(1,15), padding='valid')(b3)
b3 = Conv2DTranspose(filters=1, kernel_size=(513,1), padding='valid')(b3)


b4 = Dense(units=int(layer_flat.shape[1]), activation='relu')(layer_dense)
b4 = Reshape(target_shape=(int(layer_conv2.shape[1]),int(layer_conv2.shape[2]),int(layer_conv2.shape[3])))(b4)
b4 = Conv2DTranspose(filters=50, kernel_size=(1,15), padding='valid')(b4)
b4 = Conv2DTranspose(filters=1, kernel_size=(513,1), padding='valid')(b4)


out = Concatenate(axis=3)([b1,b2,b3,b4])
out = ReLU()(out)

model = Model(inputs=inp, outputs=out)


def loss_func(y_true, y_pred):

    global alpha, beta, beta_voc, rand_num


    voc =  y_pred[:,:,:,0:1] + epsilon * rand_num
    bass = y_pred[:,:,:,1:2] + epsilon * rand_num
    dru = y_pred[:,:,:,2:3] + epsilon * rand_num
    oth = y_pred[:,:,:,3:4] + epsilon * rand_num

    mask_vox = voc/(voc+bass+dru+oth)
    mask_bass = bass/(voc+bass+dru+oth)
    mask_drums = dru/(voc+bass+dru+oth)
    mask_oth = oth/(voc+bass+dru+oth)

    vocals = mask_vox * inp
    bass = mask_bass * inp
    drums = mask_drums * inp
    other = mask_oth * inp

    train_loss_vocals = mean_squared_error(y_true=y_true[:,:,:,0:1],y_pred=vocals)
    alpha_component = alpha*mean_squared_error(y_true=y_true[:,:,:,1:2],y_pred=vocals)
    alpha_component += alpha*mean_squared_error(y_true=y_true[:,:,:,2:3],y_pred=vocals)
    train_loss_recon_neg_voc = beta_voc*mean_squared_error(y_true=y_true[:,:,:,3:4],y_pred=vocals)

    train_loss_bass = mean_squared_error(y_true=y_true[:,:,:,1:2],y_pred=bass)
    alpha_component += alpha*mean_squared_error(y_true=y_true[:,:,:,0:1],y_pred=bass)
    alpha_component += alpha*mean_squared_error(y_true=y_true[:,:,:,2:3],y_pred=bass)
    train_loss_recon_neg = beta*mean_squared_error(y_true=y_true[:,:,:,3:4],y_pred=bass)

    train_loss_drums = mean_squared_error(y_true=y_true[:,:,:,2:3],y_pred=drums)
    alpha_component += alpha*mean_squared_error(y_true=y_true[:,:,:,0:1],y_pred=drums)
    alpha_component += alpha*mean_squared_error(y_true=y_true[:,:,:,1:2],y_pred=drums)
    train_loss_recon_neg += beta*mean_squared_error(y_true=y_true[:,:,:,3:4],y_pred=drums)

    vocals_error= K.sum(train_loss_vocals)
    drums_error= K.sum(train_loss_drums)
    bass_error= K.sum(train_loss_bass)
    negative_error= K.sum(train_loss_recon_neg)
    negative_error_voc= K.sum(train_loss_recon_neg_voc)
    alpha_component= K.sum(alpha_component)

    loss=K.abs(vocals_error+drums_error+bass_error-negative_error-alpha_component-negative_error_voc)

    return loss

# train and compile
model.compile(loss=loss_func, optimizer="adam")

checkpointer = ModelCheckpoint(filepath='/home/jangryga/source-separation/checkpoints/weights.hdf5', verbose=1, save_best_only=True)

model.fit_generator(generator=gen_tr, steps_per_epoch=4810, epochs=n_epochs, max_queue_size=max_queue,
                    use_multiprocessing=multiprocessing, validation_data=gen_ts,
                    validation_steps=467, workers=n_workers)
