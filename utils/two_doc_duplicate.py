# -*- coding: utf-8 -*-
# vim:tabstop=4:shiftwidth=4:expandtab

# 複数のモデルの計算結果の傾向より、分類を予測するメタモデル

import os
import numpy as np

from keras.models import Model
from keras import Input
from keras import layers

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence


train = df[:30000]
val = df[30000:]

max_feature = 100
max_len = 60
batch_size = 32
max_words = 10000

# tokenizer

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(np.r_[train['question1'].values, train['question2'].values, val['question1'].values, val['question1'].values])
train_q1 = tokenizer.texts_to_sequences(train['question1'])
train_q2 = tokenizer.texts_to_sequences(train['question2'])
val_q1 = tokenizer.texts_to_sequences(val['question1'])
val_q2 = tokenizer.texts_to_sequences(val['question2'])

input_train_q1 = sequence.pad_sequences(train_q1, maxlen=max_len)
input_train_q2 = sequence.pad_sequences(train_q2, maxlen=max_len)
input_val_q1 = sequence.pad_sequences(val_q1, maxlen=max_len)
input_val_q2 = sequence.pad_sequences(val_q2, maxlen=max_len)


word_index = tokenizer.word_index

glove_dir = '/Users/$(whoami)/Downloads/glove.6B'


embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# enbedding

embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# model

lstm = layers.LSTM(32)
emb_layer = layers.Embedding(
    input_dim=max_words,
    output_dim=max_feature,
    #     input_length=max_len,
    weights=[embedding_matrix],
    trainable=False
)

left_input = Input(shape=(max_len, ))
left_emd = emb_layer(left_input)
left_output = lstm(left_emd)

right_input = Input(shape=(max_len, ))
right_emd = emb_layer(right_input)
right_output = lstm(right_emd)

merged = layers.concatenate([left_output, right_output])
predictions = layers.Dense(1, activation='sigmoid')(merged)

model = Model([left_input, right_input], predictions)
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc'],
)


# train and validation

history = model.fit(
    [input_train_q1, input_train_q2],
    train['is_duplicate'],
    epochs=20,
    batch_size=128,
    validation_data=([input_val_q1, input_val_q2], val['is_duplicate'])
)


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Training val_acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Training val_loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
