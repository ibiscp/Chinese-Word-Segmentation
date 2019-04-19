import glob
from hanziconv import HanziConv
import zhon.hanzi
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("resources_path", nargs='?', default='../resources/', help="The path of the resources needed to load your model")
    parser.add_argument("sentence_size", nargs='?', const=626, type=int, default=626, help="The size of the maximum sentence")

    return parser.parse_args()

def line2BIES(data):
    encoded = []
    for i in data:
        # if len(i) > 2 and '<>' in i:
        #     print(i)
        if len(i) == 1:
            encoded.append('S')
        elif len(i) == 2:
            encoded.append('BE')
        else:
            size_i = len(i)-2
            encoded.append('B' + 'I'*size_i + 'E')

    return list(''.join(encoded)), ''.join(data)

def file2BIES(filename, characters={}):

    X_chinese = []
    y = []
    sizes = []

    print(filename)

    for line in open(filename, encoding="utf-8"):
        simplified = HanziConv.toSimplified(line)
        # clean = re.sub(p, '<>', simplified)

        data = simplified.split()
        if len(data) > 0:
            encoded, data_chinese = line2BIES(data)
            X_chinese.append(data_chinese)
            y.append(encoded)
            sizes.append(len(encoded))

            for d in data_chinese:
                if d not in characters:
                    characters[d] = 1
                else:
                    characters[d] = characters[d] + 1

    return X_chinese, y, characters, sizes

# Save dictionary to file
def save(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

# Load dictionary from file
def load(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def processX(X_chinese, word2id, sentence_size):

    # Convert sentences to id
    X = []
    for sentence in X_chinese:
        x = []
        for char in sentence:
            try:
                x.append(word2id[char])
            except:
                x.append(word2id["<UNK>"])
        X.append(x)

    # print(len(word2id))
    # print(max(sizes))
    # print(min(sizes))
    # print(sum(sizes) / len(sizes))

    # Padding
    return pad_sequences(X, truncating='post', padding='post', maxlen=sentence_size)

def processY(y, label2id, sentence_size):

    # Convert label to hot encoded
    y_encoded = []
    for a in y:
        temp = []
        for c in a:
            temp.append(label2id[c])
        y_encoded.append(temp)

    y_encoded = pad_sequences(y_encoded, truncating='post', padding='post', maxlen=sentence_size, value=0)
    return to_categorical(y_encoded)

def load_data(path='../resources/', sentence_size=630):

    dictionaryFound = False

    # Check if dictionary exists
    if glob.glob(path + 'dictionary.pkl'):
        print('\nDictionary found!')
        dictionaryFound = True
        dictionary = load(path + 'dictionary')

        word2id = dictionary['word2id']
        id2word = dictionary['id2word']
        label2id = dictionary['label2id']
        id2label = dictionary['id2label']
    else:
        print('\nDictionary not found!')

    print('\nBuilding dataset from files')

    files = [f for f in glob.glob(path + 'dataset/train/' + "*.utf8", recursive=True)]

    X_chinese = []
    y = []
    characters = {}
    sizes = []
    # p = re.compile('|'.join(list(zhon.hanzi.punctuation)))

    for filename in files:
        X_chinese_, y_, characters, sizes_ = file2BIES(filename, characters)
        X_chinese += X_chinese_
        y += y_
        sizes += sizes_

    if not dictionaryFound:
        # BIES dictionary
        label2id = {'B': 1, 'I': 2, 'E': 3, 'S': 4}
        id2label = {v:k for k,v in label2id.items()}

        # Character dictionary
        word2id = dict()
        word2id["<PAD>"] = 0 #zero is not casual!
        word2id["<UNK>"] = 1 #OOV are mapped as <UNK>
        #word2id["<PKT>"] = 3
        index = 2
        for key, value in characters.items():
            if value > 20:
                word2id[key] = index
                index += 1
        id2word = {v:k for k,v in word2id.items()}

    # Process X and y
    X_final = processX(X_chinese, word2id, sentence_size)
    y_final = processY(y, label2id, sentence_size)

    # Train test split
    train_x, dev_x, train_y, dev_y = train_test_split(X_final, y_final, test_size=.3)

    # Dataset
    dataset = {'train_x': train_x, 'dev_x': dev_x, 'train_y': train_y, 'dev_y': dev_y, 'sizes': sizes}

    # Save dictionary
    if not dictionaryFound:
        dictionary = {'word2id': word2id, 'id2word': id2word, 'label2id': label2id, 'id2label': id2label,
                      'vocab_size': len(word2id), 'sentence_size': sentence_size}
        save(dictionary, path + 'dictionary')

    return dataset, dictionary


if __name__ == '__main__':
    args = parse_args()

    # Load data
    _, _ = load_data(path=args.resources_path, sentence_size=args.sentence_size)
