import tensorflow.keras as K
from preprocess import load_data
# from sklearn.model_selection import GridSearchCV
# from scikit_learn_multi_input import KerasClassifier
# from sklearn.externals import joblib
# import pickle
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import ParameterGrid
from gridSearch import *
import warnings
# warnings.filterwarnings('ignore')

# Just disables the warning, doesn't enable AVX/FMA
import os
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
    model.add(K.layers.Dense(4, activation='softmax'))

    # Optimizer
    optimizer = K.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model

train_x, dev_x, train_y, dev_y, vocabulary_size = load_data()
sentenceSize = train_x.shape[1]

print('X shape:', train_x.shape)
print('Y shape:', train_y.shape)
print('Vocabulary size:', vocabulary_size)
print('Sentence size:', sentenceSize)

# Define the grid search parameters
batch_size = [32]#, 64, 128]
epochs = [1]#, 50, 10]
merge = ['concat', 'sum']
param_grid = dict(batch_size=batch_size, epochs=epochs, mergeMode=merge)

grid = gridSearch(build_fn=keras_model, param_grid=param_grid, vocab_size=vocabulary_size, sentence_size=sentenceSize)

cbk = K.callbacks.TensorBoard("../resources/logging/keras_model")
grid.fit(train_x, train_y, dev_x, dev_y, callbacks=[cbk])

grid.summary()

# # Define model
# model = KerasClassifier(build_fn=keras_model, vocab_size=vocabulary_size, sentence_size=sentenceSize, verbose=1)


# # param_grid = dict(mergeMode=merge)
# # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=None)
#
#
#
# # Train
# print("\nStarting training...")
# cbk = K.callbacks.TensorBoard("../resources/logging/keras_model")
# grid_result = grid.fit(train_x, train_y, validation_data=(dev_x, dev_y), callbacks=[cbk])
# # grid_result = grid.fit(train_x, train_y)#, batch_size=32, epochs=1, shuffle=True, validation_data=(dev_x, dev_y), callbacks=[cbk])
#
# # Summarize results
# print("\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#
# # Saving best estimator
# # joblib.dump(grid.best_estimator_, '../resources/estimator.pkl')
# filename = '../resources/finalized_model.sav'
# pickle.dump(grid.best_estimator_, open(filename, 'wb'))
#
# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(dev_x, dev_y)
# print(result)
