from typing import List
import numpy as np
import spacy
from collections import Counter
from scipy.sparse import lil_matrix, vstack
from tqdm import tqdm


UNK_TOKEN = '<unk>'


class NaiveBayes:
    def __init__(self, nlp, vocab=None):
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
        if vocab is not None:
            self.vocab = vocab
            self.vocab[UNK_TOKEN] = len(self.vocab)
        else:
            self.vocab = None

    def featurize(self, texts):
        """
        Remove stop words and lemmatize tokens in the text and return Lisl[List[str]]
        :param texts: List[str]
        :return: List[List[str]]
        """
        all_texts = []
        for doc in tqdm(self.nlp.pipe(texts), total=len(texts), desc='Featurizing Text'):
            tokenized_text = []
            for token in doc:
                if not token.is_stop:
                    tokenized_text.append(token.lemma_)

            all_texts.append(tokenized_text)
        return all_texts

    def vectorize(self, tokenized_data, total, add_unk=False):
        text_vectors = []
        for tokens in tqdm(tokenized_data, total=total, desc='Vectorizing Text'):
            w_counts = Counter(tokens).items()
            curr_vector = lil_matrix((1, len(self.vocab)), dtype=int)
            for word, count in w_counts:
                if word not in self.vocab and add_unk:
                    curr_vector[0, self.vocab[UNK_TOKEN]] += count
                else:
                    curr_vector[0, self.vocab[word]] += count
            text_vectors.append(curr_vector)
        return vstack(text_vectors)

    def fit(self, texts, labels, add_unk=True):
        tokenized_data = self.featurize(texts)
        self.i2label = sorted(list(set(labels)))
        label2i = {lab: i for i, lab in enumerate(self.i2label)}

        if self.vocab is None:
            all_words = sorted(list(set([w for ws in tokenized_data for w in ws])))
            self.vocab = {w: i for i, w in enumerate(all_words)}
            self.vocab[UNK_TOKEN] = len(self.vocab)

        text_vectors = self.vectorize(tokenized_data, len(texts), add_unk=add_unk)

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

    def predict(self, texts, add_unk=True):
        tokenized_data = self.featurize(texts)
        text_vectors = self.vectorize(tokenized_data, len(texts), add_unk=add_unk)
        forward_log_probs = text_vectors.dot(self.class_word_logprobs.transpose()) + self.prior
        predictions = [self.i2label[index] for index in np.argmax(forward_log_probs, axis=1)]
        forward_probs = np.exp(forward_log_probs)
        forward_probs = forward_probs/np.sum(forward_probs, axis=1).reshape((-1, 1))
        probabilities = np.max(forward_probs, axis=1)
        return predictions, probabilities


class NaiveBayesHW2:
    pass


if __name__ == '__main__':
    my_texts = ['This is good sentence', 'This is a bad sentence', 'good and amazing words', 'bad and worse words',
                'something', 'other thing']
    my_labels = [1, 0, 1, 0, 1, 0]
    my_nlp = spacy.load('en_core_web_sm')

    naive_bayes = NaiveBayes(my_nlp)
    naive_bayes.fit(my_texts, my_labels)

    preds = naive_bayes.predict(['good sentence', 'worse sentence', 'bad', 'what?', ''])
    print(preds)

