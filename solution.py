from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import label_binarize
from nltk import word_tokenize
from string import digits, punctuation

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn import svm


class Transformer:
    def __init__(self):
        self._clf = DecisionTreeClassifier()
        self._idx = None

    def fit(self, X, y):
        X = np.array(X)
        self._clf.fit(X, y)

        self._idx = filter(lambda x: self._clf.feature_importances_[x] > 0, \
                range(len(self._clf.feature_importances_)))

        return [X[i][self._idx] for i in xrange(len(X))]

    def transform(self, features):
        return features[self._idx]

class Solution:
    _ngram = 5

    def __init__(self, debug=False):
        self._opinion_to_number = dict()
        self._ngram_to_number = dict()
        self._feature_transformer = Transformer()
        self._debug = debug
        #xxx = AdaBoostClassifier(MultinomialNB(),\
        #        n_estimators=500, learning_rate=1)
        #xxx = RandomForestClassifier(n_estimators=20, min_samples_split=1)
        #xxx = svm.LinearSVC(dual=False)
        xxx = MultinomialNB()
        self._clf = OneVsRestClassifier(xxx, n_jobs=-1) 
    
    @staticmethod
    def _normalize_text(text):
        cnt_digits = 0
        for dig in digits:
            cnt_digits += text.count(dig)
            text = text.replace(dig, '')

        cnt_punct = 0
        for punct in punctuation:
            if punct == '.':
                pass
            else:
                cnt_punct += text.count(punct)
            text = text.replace(punct, '')

        text = text + ' ' + '0' * cnt_digits + ' ' + '!' * cnt_punct
        return text

    def _ngr_add(self, ngram):
        if ngram not in self._ngram_to_number:
            self._ngram_to_number[ngram] = len(self._ngram_to_number)

    def _encode_opinions(self, op_list):
        ''' 
            Transforms opinion list like [[('ggg', positive), 
            ('fff', negative)], [('qqq', eutral)]] into
            [[1, 1, 0], [0, 0, 1]]
        '''
        converted_op_list = []
        for ops in op_list:
            for op in ops:
                if op not in self._opinion_to_number:
                    self._opinion_to_number[op] = len(self._opinion_to_number)

            converted_op_list.append(tuple(map(lambda x: \
                    self._opinion_to_number[x], ops)))

        return label_binarize(converted_op_list, \
                multilabel=True, classes=range(len(self._opinion_to_number)))

    def _decode_opinions(self, bvect):
        ''' 
            Reverse transformation to _encode_opinions.
        '''
        ret = []
        for (op, number) in self._opinion_to_number.items():
            if bvect[number] == 1:
                ret.append(op)

        return ret

    @staticmethod
    def _text_tokenize(text):
        text = Solution._normalize_text(text)
        tokens = word_tokenize(text)
        tokens = list(map(lambda x: '^' + x + '$', tokens))
        return tokens
        
    @staticmethod
    def _get_ngrams(token):
        for sz in xrange(1, Solution._ngram + 1):
            for i in xrange(len(token) - sz + 1):
                yield token[i:i + sz]

    def _get_features_from_tokens(self, tokens, useTransform=False):
        ret = np.array([0] * len(self._ngram_to_number))

        for token in tokens:
            for ngram in Solution._get_ngrams(token):
                idx = self._ngram_to_number.get(ngram, -1)
                if idx != -1:
                    ret[idx] += 1

        if useTransform:
            ret = self._feature_transformer.transform(ret)

        return ret

    def train(self, train_corp):
        texts = train_corp[0]

        target = self._encode_opinions(train_corp[1])
        features_list = []

        token_list = []
        for text in texts:
            tokens = Solution._text_tokenize(text)
            token_list.append([])

            for token in tokens:
                token_list[-1].append(token)

                for ngram in Solution._get_ngrams(token):
                    self._ngr_add(ngram)

        for tokens in token_list:
            features_list.append(self._get_features_from_tokens(tokens))

        if self._debug:
            print 'Initial number of features:', len(features_list[0])

        features_list = self._feature_transformer.fit(features_list, target)

        if self._debug:
            print 'Reduced number of features:', len(features_list[0])

        self._clf.fit(features_list, target)


    def fit(self, x, y):
        self.train((x, y))
        return self

    def predict(self, text):
        tokens = Solution._text_tokenize(text)
        features = self._get_features_from_tokens(tokens, True)

        answer = self._clf.predict(features)[0]
        
        return self._decode_opinions(answer)

    def getClasses(self, texts):
        classes = []
        
        for text in texts:
            classes.append(self.predict(text))

        return classes

