from typing import List
import numpy as np
import spacy
from collections import Counter
from scipy.sparse import lil_matrix, vstack
from tqdm import tqdm
from collections import defaultdict


UNK_TOKEN = '<unk>'


def featurize(texts, nlp):
    """
    Remove stop words and lemmatize tokens in the text and return Lisl[List[str]]
    :param texts: List[str]
    :param nlp: spacy.Language
    :return: List[List[str]]
    """
    all_texts = []
    for doc in tqdm(nlp.pipe(texts), total=len(texts), desc='Featurizing Text'):
        tokenized_text = []
        for token in doc:
            if not token.is_stop:
                tokenized_text.append(token.lemma_)

        all_texts.append(tokenized_text)
    return all_texts


class NaiveBayes:
    def __init__(self, nlp, vocab=None, add_unk=False):
        """
        Initialize the Naive Bayes classifier
        :param nlp: spacy.Language
            The spaCy model
        :param vocab: dict
            A predefined dictionary of words in the data
        :return:
        """
        self.nlp = nlp
        self.prior = None
        self.i2label = None
        self.class_word_logprobs = None
        self.add_unk = add_unk
        if vocab is not None:
            self.vocab = vocab
            if self.add_unk:
                self.vocab[UNK_TOKEN] = len(self.vocab)
        else:
            self.vocab = None

    def vectorize(self, tokenized_data, total):
        text_vectors = []
        for tokens in tqdm(tokenized_data, total=total, desc='Vectorizing Text'):
            w_counts = Counter(tokens).items()
            curr_vector = lil_matrix((1, len(self.vocab)), dtype=int)
            for word, count in w_counts:
                if word not in self.vocab and self.add_unk:
                    curr_vector[0, self.vocab[UNK_TOKEN]] += count
                elif word in self.vocab:
                    curr_vector[0, self.vocab[word]] += count
            text_vectors.append(curr_vector)
        return vstack(text_vectors)

    def fit(self, texts, labels):
        tokenized_data = featurize(texts, self.nlp)
        self.i2label = sorted(list(set(labels)))
        label2i = {lab: i for i, lab in enumerate(self.i2label)}

        if self.vocab is None:
            all_words = sorted(list(set([w for ws in tokenized_data for w in ws])))
            self.vocab = {w: i for i, w in enumerate(all_words)}
            if self.add_unk:
                self.vocab[UNK_TOKEN] = len(self.vocab)

        text_vectors = self.vectorize(tokenized_data, len(texts))

        label_vector = []
        for label in labels:
            curr_vector = lil_matrix((1, len(self.i2label)), dtype=float)
            curr_vector[0, label2i[label]] = 1
            label_vector.append(curr_vector)
        label_vector = vstack(label_vector)

        class_word_probs = label_vector.T.dot(text_vectors).toarray()
        class_word_probs += 1  # smoothing
        self.class_word_logprobs = np.log(class_word_probs / class_word_probs.sum(axis=1).reshape((-1, 1)))

        self.prior = np.zeros(len(self.i2label))
        label_counter = Counter(labels)
        self.prior = np.array([label_counter[self.i2label[i]] for i in self.i2label])
        self.prior = np.log(self.prior/self.prior.sum()).reshape((1, -1))

    def predict(self, texts):
        tokenized_data = featurize(texts, self.nlp)
        text_vectors = self.vectorize(tokenized_data, len(texts))
        forward_log_probs = text_vectors.dot(self.class_word_logprobs.transpose()) + self.prior
        predictions = [self.i2label[index] for index in np.argmax(forward_log_probs, axis=1)]
        forward_probs = np.exp(forward_log_probs)
        forward_probs = forward_probs/np.sum(forward_probs, axis=1).reshape((-1, 1))
        probabilities = np.max(forward_probs, axis=1)
        return predictions, probabilities


class NaiveBayesHW2:
    def __init__(self, num_classes, nlp, vocab=None):
        self.num_classes = num_classes
        self.label_word_counter = {}
        self.label_count = {}
        self.vocab = None
        self.nlp = nlp
        if vocab is not None:
            self.vocab = vocab

    def fit(self, texts: List[str], labels: List[int]):
        """
        1. Group samples by their labels
        2. Preprocess each text
        3. Count the words of the text for each label
        """
        preprocess_texts = featurize(texts, self.nlp)
        self.vocab = {w: i for i, w in enumerate(sorted(list(set([w for ws in preprocess_texts for w in ws]))))}
        label_texts = {}
        self.label_count = Counter(labels)
        for tokens, label in zip(preprocess_texts, labels):
            if label not in label_texts:
                label_texts[label] = list(self.vocab.keys())  # smoothing: start count with Vocab
            label_texts[label].extend(tokens)

        for label, tokens in label_texts.items():
            self.label_word_counter[label] = Counter(tokens)

    def predict(self, texts: List[str]):
        """
        1. Preprocess the texts
        2. Predict the class by using the likelihood with Bayes Method and Laplace Smoothing
        """
        preprocess_texts = featurize(texts, self.nlp)
        predictions = []
        pred_probs = []
        for tokens in tqdm(preprocess_texts, desc='Predicting'):
            class_log_probs = {}
            for c in self.label_count:
                c_prior = np.log(self.label_count[c]/sum(self.label_count.values()))
                valid_tokens = [w for w in tokens if w in self.vocab]
                c_w_log_prob = np.sum([
                    np.log(self.label_word_counter[c][w]/sum(self.label_word_counter[c].values()))
                    for w in valid_tokens
                ])
                class_log_probs[c] = c_prior + c_w_log_prob
            best_c = max(class_log_probs.keys(), key=lambda x: class_log_probs[x])
            class_probs = np.array([np.exp(v) for v in class_log_probs.values()])
            class_probs = class_probs/np.sum(class_probs)
            predictions.append(best_c)
            pred_probs.append(np.max(class_probs))

        return predictions, pred_probs


if __name__ == '__main__':
    my_texts = ['This is good sentence', 'This is a bad sentence', 'good and amazing words', 'bad and worse words',
                'something', 'other thing', 'yes']
    my_labels = [1, 0, 1, 0, 1, 0, 0]
    my_nlp = spacy.load('en_core_web_sm')

    naive_bayes = NaiveBayes(my_nlp, add_unk=False)
    naive_bayes.fit(my_texts, my_labels)

    preds = naive_bayes.predict(['good sentence', 'worse sentence', 'bad', 'good and amazing', 'what?', ''])
    print('my Naive:', preds)

    naive_bayes_hw = NaiveBayesHW2(2, my_nlp)
    naive_bayes_hw.fit(my_texts, my_labels)
    preds = naive_bayes_hw.predict(['good sentence', 'worse sentence', 'bad', 'good and amazing', 'what?', ''])
    print('hw2 Naive:', preds)

