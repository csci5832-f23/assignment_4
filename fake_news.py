import os
from irony import git
import shutil
import urllib
import zipfile
import spacy
from bayes import NaiveBayes


FAKE_NEWS_FOLDER = 'siamese-BERT-fake-news-detection-LIAR'
TRAIN_FILE = './siamese-BERT-fake-news-detection-LIAR/LIAR-PLUS/dataset/train2.tsv'
VALID_FILE = './siamese-BERT-fake-news-detection-LIAR/LIAR-PLUS/dataset/val2.tsv'
TEST_FILE = './siamese-BERT-fake-news-detection-LIAR/LIAR-PLUS/dataset/test2.tsv'


def download_dataset():
    if os.path.exists(FAKE_NEWS_FOLDER):
        return
    else:
        try:
            git('clone', 'https://github.com/manideep2510/siamese-BERT-fake-news-detection-LIAR.git')
            return
        except OSError:
            # Do nothing. Continue try downloading zip file
            pass

    print('Downloading dataset')
    path_to_zip_file = 'SemEval2018-Task3.zip'
    # Download the dataset zipped file
    urllib.request.urlretrieve('https://github.com/manideep2510/siamese-BERT-fake-news-detection-LIAR/master.zip',
                               path_to_zip_file)
    # Unzip the dataset
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall('./')
    # Delete any existing sem eval folder
    if os.path.exists(FAKE_NEWS_FOLDER):
        shutil.rmtree(FAKE_NEWS_FOLDER)
    # Remove zip file
    if os.path.exists(path_to_zip_file):
        os.remove(path_to_zip_file)
    # Rename sem eval folder name
    shutil.move(FAKE_NEWS_FOLDER + '-master', FAKE_NEWS_FOLDER)


def load_fake_news(file_path):
    texts, meta_texts, labels = [], [], []
    with open(file_path) as vf:
        rows = [line.strip().split('\t') for line in vf.readlines()]
        for curr_row in rows:
            meta_ = ' '.join(['party:' + curr_row[8],'title:' + curr_row[6].replace(' ', '_'), curr_row[5], 'loc:' + curr_row[7], 'topic:' + curr_row[4]])
            meta_texts.append(meta_)
            labels.append(curr_row[2])
            texts.append(curr_row[3])
    return texts, labels, meta_texts


if __name__ == '__main__':
    train_t, train_labels, train_meta= load_fake_news(TRAIN_FILE)
    val_t, val_labels, val_meta = load_fake_news(VALID_FILE)

    from bayes import featurize

    my_nlp = spacy.load('en_core_web_sm')
    tokenized_data_train = featurize(train_t, my_nlp)
    tokenized_data_val = featurize(val_t, my_nlp)

    train_with_meta = []
    for train_d, meta_d in zip(tokenized_data_train, train_meta):
        train_with_meta.append(train_d + meta_d.split())
    val_with_meta = []
    for val_d, meta_d in zip(tokenized_data_val, val_meta):
        val_with_meta.append(val_d + meta_d.split())

    naive_b = NaiveBayes()
    naive_b.fit(train_with_meta, train_labels)
    val_pred, _ = naive_b.predict(val_with_meta)


    def acc(pred, labs):
        true_count = 0
        for p, l in zip(pred, labs):
            true_count += int(p == l)
        return true_count / len(labs)

    print('Naive Bayes Baseline')
    print('Accuracy:', acc(val_pred, val_labels))
