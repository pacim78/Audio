import pyaudio, wave, librosa, librosa.display, librosa.feature, librosa.feature.inverse
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from keras.preprocessing import sequence

from frames import *
from preproc.txt_to_phoneme import *

def encode_single_sample(name,label):
    ##  Process the Audio
    name = pad_shave_frame(name)[0]
    name = tf.convert_to_tensor(name,dtype='float32')
    ##  Process the label
    label = phoneme(label)
    label = tf.convert_to_tensor(label,dtype='int64')
    return name,label
    
def make_dataset():
    df_train = pd.read_csv('rec_data/dataset.csv',encoding='utf8')
    df_train = df_train[['audiofile','transcript']]
    df_train.transcript = [x.replace('\r\n',' ').replace('\n',' ') for x in df_train.transcript]
    df_train.audiofile = ['rec_data/' + x for x in df_train.audiofile]
    # Define the training dataset
    inputs, labels = [], []
    for i,row in df_train.iterrows():
        input,label = encode_single_sample(row[0],row[1])
        inputs.append(input)
        labels.append(label)
    train_dataset = tf.data.Dataset.from_tensor_slices((list(inputs), list(labels))).padded_batch(2).prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_dataset

def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=128):
    input_spectrogram = layers.Input((None, input_dim), name="input")
    x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 41],
        strides=[2, 2],
        padding="same",
        use_bias=False,
        name="conv_1",
    )(x)
    x = layers.BatchNormalization(name="conv_1_bn")(x)
    x = layers.ReLU(name="conv_1_relu")(x)
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 21],
        strides=[1, 2],
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x)
    x = layers.BatchNormalization(name="conv_2_bn")(x)
    x = layers.ReLU(name="conv_2_relu")(x)
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    for i in range(1, rnn_layers + 1):
        recurrent = layers.GRU(
            units=rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        )
        x = layers.Bidirectional(
            recurrent, name=f"bidirectional_{i}", merge_mode="concat"
        )(x)
        if i < rnn_layers:
            x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
    x = layers.ReLU(name="dense_1_relu")(x)
    x = layers.Dropout(rate=0.5)(x)
    output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
    '''
    final = ['']
    for i in range(len(x)):
        if x[i] == 'e':
            pass
        elif x[i] == final[-1]:
            pass
        else:
            final.append(x[i])
    final = final[1:]
    '''
    model = keras.Model(input_spectrogram, output, name="model")
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss=CTCLoss)
    return model