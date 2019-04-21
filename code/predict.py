from argparse import ArgumentParser
from tensorflow.keras.models import load_model
from preprocess import load
from preprocess import file2BIES
from preprocess import processX
import numpy as np
from score import score


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", nargs='?', default='../resources/dataset/gold/cityu_test_gold.utf8', help="The path of the input file")
    parser.add_argument("output_path", nargs='?', default='../resources/dataset/predicted/cityu_test_gold.utf8', help="The path of the output file")
    parser.add_argument("resources_path", nargs='?', default='../resources/', help="The path of the resources needed to load your model")

    return parser.parse_args()


def predict(input_path, output_path, resources_path):
    """
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the BIES format.
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """

    model = load_model(resources_path + '/model.h5')

    model.summary()

    dictionary = load(resources_path + 'dictionary')
    word2id = dictionary['word2id']
    id2label = dictionary['id2label']

    X_chinese, y, characters, sizes = file2BIES(input_path)

    # Process X
    X_processed = processX(X_chinese, word2id, sentence_size=626)

    y_pred = model.predict(X_processed)

    prediction = []

    arg = np.argmax(y_pred, axis=2)

    for i in range(len(arg)):
        sentence = arg[i]
        labels = []
        num_char = np.count_nonzero(X_processed[i])
        for char in sentence[0:num_char]:
            labels.append(id2label[char])

        prediction.append(labels)

    score(prediction, y, verbose=True)

    with open(output_path, "w+") as f:
        for line in prediction:
            f.write(''.join(str(e) for e in line))
            f.write('\n')

    pass


if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)
