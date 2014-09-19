from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import label_binarize
from nltk import word_tokenize
from string import digits, punctuation
from nltk.stem.snowball import RussianStemmer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier


class Solution:
    _ngram = 3
    _stemmer = RussianStemmer('russian')

    def __init__(self, debug=False):
        self._opinion_to_number = dict()
        self._ngram_to_number = dict()
        self._n_features = 0
        self._debug = debug
        xxx = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),\
                n_estimators=20, learning_rate=1)
        xxx = RandomForestClassifier(n_estimators=20, min_samples_split=1)
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
        n_ops = 0
        converted_op_list = []
        for ops in op_list:
            for op in ops:
                if op not in self._opinion_to_number:
                    self._opinion_to_number[op] = n_ops
                    n_ops += 1

            converted_op_list.append(tuple(map(lambda x: \
                    self._opinion_to_number[x], ops)))

        return label_binarize(converted_op_list, \
                multilabel=True, classes=range(n_ops))

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
        tokens = list(map(lambda x: '^' + Solution._stemmer.stem(x) + '$', tokens))
        return tokens
        
    @staticmethod
    def _get_ngrams(token):
        for sz in xrange(1, Solution._ngram + 1):
            for i in xrange(len(token) - sz + 1):
                yield token[i:i + sz]

    def _get_features_from_tokens(self, tokens):
        ret = [0 for i in xrange(len(self._ngram_to_number))]

        for token in tokens:
            for ngram in Solution._get_ngrams(token):
                idx = self._ngram_to_number.get(ngram, -1)
                if idx != -1:
                    ret[idx] += 1

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

        self._clf.fit(features_list, target)


    def fit(self, x, y):
        self.train((x, y))
        return self

    def predict(self, text):
        tokens = Solution._text_tokenize(text)
        features = self._get_features_from_tokens(tokens)

        answer = self._clf.predict(features)[0]
        
        return self._decode_opinions(answer)

    def predict_proba(self, X):
        ret = []
        for e in self._clf.estimators_:
            try:
                ret.append(e.predict_proba(X)[:,1])
            except:
                ret.append(e.predict(X))
        return np.column_stack(ret)

    def getClasses(self, texts):
        classes = []
        
        for text in texts:
            classes.append(self.predict(text))

        return classes

