from sklearn.model_selection import ParameterGrid
import tensorflow.keras as K

class gridSearch:

    def __init__(self, build_fn, param_grid, vocab_size, sentence_size):
        self.build_fn = build_fn
        self.param_grid = param_grid
        self.best_score = 0
        self.best_params = None
        self.results = []
        self.vocab_size = vocab_size
        self.sentence_size = sentence_size

    def fit(self, X, y, X_test, y_test):

        for g in ParameterGrid(self.param_grid):
            model = self.build_fn(vocab_size=self.vocab_size, sentence_size=self.sentence_size, mergeMode=g['mergeMode'])

            print('\nUsing parameters:', g)
            callback_str = '_'.join(['%s-%s' % (key, str(value)) for (key, value) in g.items()])
            cbk = K.callbacks.TensorBoard("../resources/logging/" + callback_str)
            model.fit(X, y, batch_size=g['batchSize'], epochs=g['epochs'], shuffle=True, callbacks=[cbk])

            print('Evaluating')
            loss, acc = model.evaluate(X_test, y_test, verbose=1)
            print('Loss: %f - Accuracy: %f' % (loss, acc))

            self.results.append({'loss':loss, 'acc':acc, 'params':g})

            if acc > self.best_score:
                self.best_score = acc
                self.best_params = g

                # Save model
                print("Saving model")
                model.save("../resources/model.h5")

    def summary(self):
        # Summarize results
        print('\nSummary')
        print("Best: %f using %s" % (self.best_score, self.best_params))
        for res in self.results:
            print("Loss: %f - Accuracy: %f - Parameters: %r" % (res['loss'], res['acc'], res['params']))

        with open('../resources/results.txt', "w+") as f:
            f.write('Summary\n')
            f.write("Best: %f using %s\n" % (self.best_score, self.best_params))
            for res in self.results:
                f.write("Loss: %f - Accuracy: %f - Parameters: %r\n" % (res['loss'], res['acc'], res['params']))
