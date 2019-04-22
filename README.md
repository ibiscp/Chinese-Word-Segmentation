# Chinese Word Segmentation

The goal of this project is to train a model based on Bidirectional LSTM to separate chinese words in a sentence.

The dataset used for the training was the concatenation of four different datasets: AS (Traditional Chinese), CITYU (Traditional Chinese), MSR (Simplified Chinese) and PKU (Simplified Chinese).

The training was done using a Google Compute Engine instance running a Tesla K80 GPU.

## Instructions

* Generate dictionary

`python preprocess.py [resources_path] [sentence_size]`


* Train

`python train.py [resources_path] [sentence_size]`

* Predict

`python train.py [input_path] [output_path] [resources_path]`

* Score

`python train.py [prediction_file] [gold_file]`