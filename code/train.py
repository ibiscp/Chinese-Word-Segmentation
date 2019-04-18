import tensorflow.keras as K
from preprocess import load_data
from gridSearch import *

# Just disables the warning
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def keras_model(vocab_size=1000, sentence_size=100, embedding_size=64, hidden_size=256, mergeMode='concat'):

    model = K.models.Sequential()

    # Embedding
    model.add(K.layers.Embedding(vocab_size, embedding_size, input_length=sentence_size, mask_zero=True))

    # Bidirectional LSTM
    model.add(
        K.layers.Bidirectional(K.layers.LSTM(
            hidden_size, dropout=0.2, recurrent_dropout=0.2, return_sequences=True), merge_mode=mergeMode))
    model.add(
        K.layers.Bidirectional(K.layers.LSTM(
            hidden_size, dropout=0.2, recurrent_dropout=0.2, return_sequences=True), merge_mode=mergeMode))

    # Dense layer
    model.add(K.layers.Dense(5, activation='softmax'))

    # Optimizer
    optimizer = K.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model

# Load data
dataset, dictionary = load_data()
train_x = dataset['train_x']
dev_x = dataset['dev_x']
train_y = dataset['train_y']
dev_y = dataset['dev_y']
vocabulary_size = dictionary['vocab_size']
sentenceSize = dictionary['sentence_size']
print('\nTraining dataset information')
print('X shape:', train_x.shape)
print('Y shape:', train_y.shape)
print('Vocabulary size:', vocabulary_size)
print('Sentence size:', sentenceSize)

# Define the grid search parameters
batch_size = [32, 64]
epochs = [1, 3]
merge = ['concat', 'sum']
param_grid = dict(batch_size=batch_size, epochs=epochs, mergeMode=merge)

# Train
grid = gridSearch(build_fn=keras_model, param_grid=param_grid, vocab_size=vocabulary_size, sentence_size=sentenceSize)
cbk = K.callbacks.TensorBoard("../resources/logging/keras_model")
grid.fit(train_x, train_y, dev_x, dev_y, callbacks=[cbk])

# Print grid search summary
grid.summary()