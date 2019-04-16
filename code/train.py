import tensorflow.keras as K
from preprocess import load_data
from sklearn.model_selection import GridSearchCV
from scikit_learn_multi_input import KerasClassifier
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def keras_model(vocab_size=1000, maxlen=100, embedding_size=64, hidden_size=256):

    print("Creating KERAS model")

    model = K.models.Sequential()

    # Embedding
    model.add(K.layers.Embedding(vocab_size, embedding_size, input_length=maxlen, mask_zero=True))

    # Bidirectional LSTM
    model.add(
        K.layers.Bidirectional(K.layers.LSTM(
            hidden_size, dropout=0.2, recurrent_dropout=0.2, return_sequences=True), merge_mode='concat'))
    model.add(
        K.layers.Bidirectional(K.layers.LSTM(
            hidden_size, dropout=0.2, recurrent_dropout=0.2, return_sequences=True), merge_mode='concat'))

    # Dense layer
    model.add(K.layers.Dense(5, activation='softmax'))

    # we are going to use the Adam optimizer which is a really powerful optimizer.
    optimizer = K.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model

train_x, dev_x, train_y, dev_y, vocabulary_size = load_data()

print(train_x.shape)
print(train_y.shape)
print(vocabulary_size)

sentence_size = train_x.shape[1]
print(sentence_size)

model = KerasClassifier(build_fn=keras_model, vocab_size=vocabulary_size, verbose=1)

# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [2, 10, 20]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=None)

cbk = K.callbacks.TensorBoard("logging/keras_model")

print("\nStarting training...")
grid_result = grid.fit(train_x, train_y, shuffle=True, validation_data=(dev_x, dev_y), callbacks=[cbk])

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# model = create_keras_model(vocabulary_size, maxlen=sentence_size)
# # Let's print a summary of the model
# model.summary()
#
# cbk = K.callbacks.TensorBoard("logging/keras_model")
# print("\nStarting training...")
# model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
#           shuffle=True, validation_data=(dev_x, dev_y), callbacks=[cbk])
# print("Training complete.\n")