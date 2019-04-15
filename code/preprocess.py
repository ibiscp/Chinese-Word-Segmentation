import glob
from hanziconv import HanziConv
import zhon.hanzi
import re
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

def encoding(data):
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

# Save dictionary to file
def save_dic(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# Load dictionary from file
def load_dic(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_data(dataset='../resources/dataset/', dic='dictionary'):

    # Check if dictionary exists
    if glob.glob(dataset + dic + '.pkl'):
        print('Dictionary file found!')
        [train_x, dev_x, train_y, dev_y, word_to_id, id_to_word, label2id, id2label, sizes] = load_dic(dataset + dic)

    else:
        print('Dictionary file not found!')

        files = [f for f in glob.glob(dataset + "*.utf8", recursive=True)]

        X_chinese = []
        y = []
        characters = {}
        sizes = []
        p = re.compile('|'.join(list(zhon.hanzi.punctuation)))

        for filename in files:
            print(filename)
            for line in open(filename, encoding="utf-8"):
                simplified = HanziConv.toSimplified(line)
                # clean = re.sub(p, '<>', simplified)

                data = simplified.split()
                if len(data)>0:
                    encoded, data_chinese = encoding(data)
                    X_chinese.append(data_chinese)
                    y.append(encoded)
                    sizes.append(len(encoded))

                    for d in data_chinese:
                        if d not in characters:
                            characters[d] = 1
                        else:
                            characters[d] = characters[d] + 1

        label2id = {'B': 1, 'I': 2, 'E': 3, 'S': 4}
        id2label = {v:k for k,v in label2id.items()}

        word_to_id = dict()
        word_to_id["<PAD>"] = 0 #zero is not casual!
        word_to_id["<UNK>"] = 1 #OOV are mapped as <UNK>
        #word_to_id["<PKT>"] = 3

        index = 2
        for key, value in characters.items():
            if value > 10:
                word_to_id[key] = index
                index += 1

        id_to_word = {v:k for k,v in word_to_id.items()}

        X = []
        for sentence in X_chinese:
            x = []
            for char in sentence:
                try:
                    x.append(word_to_id[char])
                except:
                    x.append(word_to_id["<UNK>"])
            X.append(x)

        # print(word_to_id)
        # print(id_to_word)
        #
        # print(len(word_to_id))
        # print(max(sizes))
        # print(min(sizes))
        # print(sum(sizes) / len(sizes))

        X = pad_sequences(X, truncating='post', padding='post', maxlen=100)
        # print(X)

        # Convert label to hot encoded
        y_encoded = []
        for a in y:
            temp = []
            for c in a:
                temp.append(label2id[c])
            y_encoded.append(temp)


        # transfomed_label = MultiLabelBinarizer().fit_transform(y)
        # print(transfomed_label)

        y_encoded = pad_sequences(y_encoded, truncating='post', padding='post', maxlen=100, value=0)
        # print(y_encoded)
        y_l = to_categorical(y_encoded)
        # print(y_l)
        #
        # print(y[0])
        # print(y_encoded[0])
        # print(y_l[0])

        train_x, dev_x, train_y, dev_y = train_test_split(X, y_l, test_size=.2)

        save_dic([train_x, dev_x, train_y, dev_y, word_to_id, id_to_word, label2id, id2label, sizes], dataset + dic)

    vocabulary_size = len(word_to_id)

    return train_x, dev_x, train_y, dev_y, vocabulary_size

# train_x, dev_x, train_y, dev_y, vocabulary_size= load_data()