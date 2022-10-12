import shutil
import urllib.request
import zipfile
import os
import subprocess
import emoji
import torch
import numpy as np

from util import *
from bayes import NaiveBayes


#### READ DATASET ####

SEM_EVAL_FOLDER = 'SemEval2018-Task3'
TRAIN_FILE = SEM_EVAL_FOLDER + '/datasets/train/SemEval2018-T3-train-taskA_emoji.txt'
TEST_FILE = SEM_EVAL_FOLDER + '/datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt'


def download_dataset():
    if os.path.exists(SEM_EVAL_FOLDER):
        return
    else:
        try:
            git('clone', 'https://github.com/Cyvhee/SemEval2018-Task3.git')
            return
        except OSError:
            # Do nothing. Continue try downloading zip file
            pass

    print('Downloading dataset')
    path_to_zip_file = 'SemEval2018-Task3.zip'
    # Download the dataset zipped file
    urllib.request.urlretrieve('https://github.com/Cyvhee/SemEval2018-Task3/archive/master.zip',
                               path_to_zip_file)
    # Unzip the dataset
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall('./')
    # Delete any existing sem eval folder
    if os.path.exists(SEM_EVAL_FOLDER):
        shutil.rmtree(SEM_EVAL_FOLDER)
    # Remove zip file
    if os.path.exists(path_to_zip_file):
        os.remove(path_to_zip_file)
    # Rename sem eval folder name
    shutil.move(SEM_EVAL_FOLDER + '-master', SEM_EVAL_FOLDER)


def read_dataset_file(file_path):
    with open(file_path, 'r', encoding='utf8') as ff:
        rows = [line.strip().split('\t') for line in ff.readlines()[1:]]
        _, labels, texts = zip(*rows)
    clean_texts = [emoji.demojize(tex) for tex in texts]
    return clean_texts, labels


def load_datasets():
    # download dataset from git
    download_dataset()

    # read the datasets
    train_texts, train_labels = read_dataset_file(TRAIN_FILE)
    test_texts, test_labels = read_dataset_file(TEST_FILE)

    return train_texts, train_labels, test_texts, test_labels


def all_predicts(test_preds, test_labs):
    print('Accuracy:', accuracy(np.array(test_preds), np.array(test_labs)))
    print('Precision:', precision(np.array(test_preds), np.array(test_labs), which_label='1'))
    print('Recall:', recall(np.array(test_preds), np.array(test_labs), which_label='1'))
    print('F1-score:', f1_score(np.array(test_preds), np.array(test_labs), which_label='1'))


if __name__ == '__main__':
    train_t, train_labels, test_t, test_labels = load_datasets()

    train_t_processed = [t.split() for t in train_t]
    test_t_processed = [t.split() for t in test_t]
    #
    # ### Baseline: Naive Bayes ###
    nb = NaiveBayes()
    nb.fit(train_t_processed, train_labels)
    _, train_probs = nb.predict(train_t_processed)

    t_predictions, _ = nb.predict(test_t_processed)

    print('Baseline: Naive Bayes Classifier')
    all_predicts(t_predictions, test_labels)
