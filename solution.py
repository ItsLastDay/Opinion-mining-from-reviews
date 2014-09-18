from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import label_binarize
from nltk import word_tokenize

class Solution:
    _ngram = 3

    def __init__(self):
        self._opinion_to_number = dict()
        self._ngram_to_number = dict()
        self._n_fearures = 0
        self._clf = OneVsRestClassifier(MultinomialNB(), n_jobs=-1) 
    
    def get_params(self):
        return self._clf.get_params(deep=True)

    @staticmethod
    def _normalize_text(text):
        return text.lower()

    def _ngr_add(self, ngram):
        if ngram not in self._ngram_to_number:
            self._ngram_to_number[ngram] = len(self._ngram_to_number)

    def _encode_opinions(self, op_list):
        ''' 
            Transforms opinion list like [[('ggg', positive), 
            ('fff', negative)], [('qqq', eutral)]] into
            [[1, 1, 0], [0, 0, 1]]
        '''
        n_ops = 0
        converted_op_list = []
        for ops in op_list:
            for op in ops:
                if op not in self._opinion_to_number:
                    self._opinion_to_number[op] = n_ops
                    n_ops += 1

            converted_op_list.append(tuple(map(lambda x: \
                    self._opinion_to_number[x], ops)))

        return label_binarize(converted_op_list, classes=range(n_ops))
        
    def train(self, train_corp):
        texts = train_corp[0]

        target = self._encode_opinions(train_corp[1])
        features_list = []

        token_list = []
        for text in texts:
            ntext = Solution._normalize_text(text)
            
            tokens = word_tokenize(ntext)
            token_list.append([])
            for token in tokens:
                tk = '^' + token + '$'
                token_list[-1].append(tk)
                for i in xrange(len(tk) - Solution._ngram + 1):
                    self._ngr_add(tk[i:i + Solution._ngram])

        n_features = len(self._ngram_to_number)
        for tokens in token_list:
            features_list.append([0 for i in xrange(n_features)])

            for tk in tokens:
                features_list[-1][self._ngram_to_number[tk]] += 1

        self._clf.fit(features_list, target)


    def fit(self, x, y):
        self.train((x, y))
        return self

    def predict(self, text):
        pass

    def getClasses(self, texts):
        pass

