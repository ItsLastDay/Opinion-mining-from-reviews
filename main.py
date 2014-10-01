from data_extractor import get_nice_data, clean_answer, get_data
from solution import Solution
from sklearn.metrics import f1_score
from sklearn.cross_validation import KFold
import numpy as np
import sys

def encode_ops(ops):
    all_ops = set()
    for op in ops:
        all_ops.update(set(op))

    d = dict()
    for op in all_ops:
        d[op] = len(d)

    return d

def transform(ops, tr):
    ret = []
    for op in ops:
        ret.append(list(map(lambda x: tr[x], op)))
    return ret

if True:
    train_data = get_nice_data(get_data('reviews.json'))
    train_data = list(map(lambda x: np.array(x[:100]), train_data))

    scores = []
    for train_idx, test_idx in KFold(len(train_data[0]), n_folds=3, \
            shuffle=True):
        X_train = train_data[0][train_idx]
        Y_train = train_data[1][train_idx]

        X_test = train_data[0][test_idx]
        Y_test = train_data[1][test_idx]

        sol = Solution(True)
        sol.train((X_train, Y_train))

        # sometimes it says "AttributeError: '_ConstantPredictor'
        # object has no attribute 'predict_proba'". It happens when some 
        # opinion is presented in all training data. I think it's data problem,
        # not classificator's.
        answer = sol.getClasses(X_test)

        transformer = encode_ops(answer + Y_test)
        answer = transform(answer, transformer)
        Y_test = transform(Y_test, transformer)
        
        f_m = f1_score(Y_test, answer, labels=range(len(transformer)) ,average='micro')

        print 'F-measure for this fold is', f_m
        scores.append(f_m)

    print 'Total score is:', np.array(scores).mean()
else:
    train_data = get_data('reviews.json')
    x = Solution()
    x.train(train_data)
