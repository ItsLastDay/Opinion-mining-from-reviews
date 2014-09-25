from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import Bootstrap
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import label_binarize
from nltk import word_tokenize
from string import digits, punctuation
from subprocess import check_output

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

class MyStemWrapper:
    def __init__(self):
        self._mystem = './mystem'
        self._lemmatized = dict()
        self._has_punct = lambda x:\
                any([c in x for c in punctuation + digits])

    def _remove_questions(self, lst):
        ret = []

        for word in lst:
            if word and word[-1] == '?':
                ret.append(word[:-1])
            else:
                ret.append(word)

        return ret

    def _bulk_lemmatize(self, tokens):
        '''
            Returns list of pairs (token, lemmas). Just one call to mystem is used.
        '''
        tokens = list(filter(lambda x: x != '' and not self._has_punct(x), tokens))
        output = check_output("echo " + ' '.join(tokens) + " | " +\
                self._mystem + " -nl", universal_newlines=True,
                shell=True)

        ret = []
        for (line, word) in zip(output.split('\n'), tokens):
            line = line.strip().split('|')
            line = self._remove_questions(line)
            ret.append((word, line))

        return ret

    def _lemmatize(self, token):
        '''
            Returns list of possible lemmas of token using MyStem library.
        '''
        if self._has_punct(token):
            # mystem thinks there is >1 words if '-' or ' ' are presented
            return [token]
        output = check_output("echo " + token + " | " + self._mystem + "-nl",\
                universal_newlines=True)

        output = output.strip().split('|')
        output = self._remove_questions(output)

        return output
        
    def lemmatize(self, token):
        if token not in self._lemmatized:
            self._lemmatized[token] = self._lemmatize(token)
        return self._lemmatized[token]

    def bulk_lemmatize(self, tokens):
        result = self._bulk_lemmatize(tokens)
        for (token, lemmas) in result:
            self._lemmatized[token] = lemmas



class Bagger:
    def __init__(self):
        #xxx = AdaBoostClassifier(MultinomialNB(),\
        #        n_estimators=500, learning_rate=1)
        #xxx = RandomForestClassifier(n_estimators=20, min_samples_split=1)
        #xxx = svm.LinearSVC(dual=False)
        xxx = MultinomialNB()
        self._n_estimators = 50
        self._estimators = [None for i in range(self._n_estimators)]
        self._target_len = None
        for i in range(self._n_estimators):
            self._estimators[i] = OneVsRestClassifier(xxx, n_jobs=-1) 

    def fit(self, features, target):
        bs = Bootstrap(len(features), n_iter=self._n_estimators, \
                train_size=len(features) - 1, random_state=0)
        self._target_len = len(target[0])

        i = 0
        for (train, test) in bs:
            t_set_features = np.array(features)[train]
            t_set_target = np.array(target)[train]
            t_set_target[0] = [1 for k in xrange(len(t_set_target[1]))]
            self._estimators[i].fit(t_set_features, t_set_target)
            i += 1

    def predict(self, feature):
        answers = np.array([self._estimators[i].predict(feature)[0] for i in range(\
                self._n_estimators)]).mean(0)
        ret = []
        for i in range(self._target_len):
            if answers[i] * 2 >= 1.0:
                ret.append(1)
            else:
                ret.append(0)
        return ret
         


class Solution:
    _ngram = 5
    _lemmatizer = MyStemWrapper()

    def __init__(self, debug=False):
        self._opinion_to_number = dict()
        self._ngram_to_number = dict()
        self._feature_transformer = Transformer()
        self._debug = debug
        self._clf = Bagger() 
    
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

        if self._debug:
            print self._opinion_to_number
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
        tokens = list(map(lambda x: '^' + Solution._lemmatizer.lemmatize(x)[0]\
                + '$', tokens))
        return tokens
        
    @staticmethod
    def _get_ngrams(token):
        for sz in xrange(1, Solution._ngram + 1):
            for i in xrange(len(token) - sz + 1):
                yield token[i:i + sz]

    @staticmethod
    def _remove_differencies(train_corp):
        '''
            When the same text appears multiple times in corpus,
            we only remain features, that were marked by all people.
        '''
        cnt = dict()
        op_cnt = dict()
        for i in xrange(len(train_corp[0])):
            text = train_corp[0][i]
            features = set(train_corp[1][i])

            op_cnt[text] = op_cnt.get(text, 0) + 1

            if text not in cnt:
                cnt[text] = dict()
            for f in features:
                cnt[text][f] = cnt[text].get(f, 0) + 1
            

        texts = [text for text in cnt.keys()]
        target = []
        for text in texts:
            features = cnt[text]
            men_voted = op_cnt[text]

            features = filter(lambda x: cnt[text][x] * 2 > men_voted, features)

            target.append(features)

        return (texts, target)

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
        train_corp = Solution._remove_differencies(train_corp)
        texts = train_corp[0]

        target = self._encode_opinions(train_corp[1])
        features_list = []

        all_texts = set()
        for text in texts:
            all_texts.update(word_tokenize(text))
        self._lemmatizer.bulk_lemmatize(list(all_texts))

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

        answer = self._clf.predict(features)
        
        return self._decode_opinions(answer)

    def getClasses(self, texts):
        classes = []
        
        for text in texts:
            classes.append(self.predict(text))

        return classes

