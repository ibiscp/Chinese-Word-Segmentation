import tensorflow.keras as K
from argparse import ArgumentParser
from preprocess import load_data
from gridSearch import *

# Just disables the warning
# import warnings
# import os
# warnings.filterwarnings('ignore')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("resources_path", nargs='?', default='../resources/', help="The path of the resources needed to load your model")
    parser.add_argument("sentence_size", nargs='?', const=626, type=int, default=626, help="The size of the maximum sentence")

    return parser.parse_args()

def keras_model(vocab_size, sentence_size, mergeMode, lstmLayers, embedding_size, hidden_size=256, dropout=0.2, recurrent_dropout=0.2):

    model = K.models.Sequential()

    # Embedding
    model.add(K.layers.Embedding(vocab_size, embedding_size, input_length=sentence_size, mask_zero=True))

    # Bidirectional LSTM
    for i in range(lstmLayers):
        model.add(
            K.layers.Bidirectional(K.layers.LSTM(
                hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True), merge_mode=mergeMode))

    # Dense layer
    model.add(K.layers.Dense(5, activation='softmax'))

    # Optimizer
    optimizer = K.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model

if __name__ == '__main__':
    args = parse_args()

    # Load data
    dataset, dictionary = load_data(path=args.resources_path, sentence_size=args.sentence_size)
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
    batchSize = [64]
    epochs = [10]
    mergeMode = ['sum']
    lstmLayers = [2]
    embedding_size = [16, 32]
    param_grid = dict(batchSize=batchSize, epochs=epochs, mergeMode=mergeMode, lstmLayers=lstmLayers, embedding_size=embedding_size)

    # Train
    grid = gridSearch(build_fn=keras_model, param_grid=param_grid, vocab_size=vocabulary_size, sentence_size=sentenceSize)
    grid.fit(train_x, train_y, dev_x, dev_y)

    # Print grid search summary
    grid.summary()