from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import label_binarize

class Solution:
    def __init__(self):
        self._opinion_to_number = dict()
        self._ngram_to_number = dict()
        self._n_fearures = 0
        self._clf = OneVsRestClassifier(MultinomialNB(), n_jobs=-1) 
    
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


        for text in texts:
            pass

        return self

    def fit(self, x, y):
        self.train((x, y))
        return self

    def predict(self, text):
        pass

    def getClasses(self, texts):
        pass

