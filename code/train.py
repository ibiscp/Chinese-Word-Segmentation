import tensorflow.keras as K
from preprocess import load_data

def create_keras_model(vocab_size, maxlen, embedding_size=64, hidden_size=256):

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
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model

train_x, dev_x, train_y, dev_y, vocabulary_size = load_data()

print(train_x.shape)
print(train_y.shape)

sentence_size = train_x.shape[1]
print(sentence_size)

batch_size = 32
epochs = 3

model = create_keras_model(vocabulary_size, maxlen=sentence_size)
# Let's print a summary of the model
model.summary()

cbk = K.callbacks.TensorBoard("logging/keras_model")
print("\nStarting training...")
model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
          shuffle=True, validation_data=(dev_x, dev_y), callbacks=[cbk])
print("Training complete.\n")