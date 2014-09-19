from data_extractor import get_nice_data, clean_answer
from solution import Solution
from sklearn.metrics import f1_score
from sklearn.cross_validation import KFold
import numpy as np


train_data = get_nice_data('reviews.json')
train_data = list(map(lambda x: np.array(x), train_data))

scores = dict()
for train_idx, test_idx in KFold(len(train_data[0]), n_folds=10, \
        shuffle=True):
    X_train = train_data[0][train_idx]
    Y_train = train_data[1][train_idx]

    X_test = train_data[0][test_idx]
    Y_test = train_data[1][test_idx]

    sol = Solution()
    sol.train((X_train, Y_train))

    # sometimes it says "AttributeError: '_ConstantPredictor'
    # object has no attribute 'predict_proba'". It happens when some 
    # opinion is presented in all training data. I think it's data problem,
    # not classificator's.
    answer = sol.getClasses(X_test)

    print 'fold calculated'
    for strat in ['samples']:
        score = f1_score(Y_test, answer, average=strat)
        print score
        if strat not in scores:
            scores[strat] = np.array([])
        scores[strat] = np.append(scores[strat], score)

for strat in ['samples']:
    print 'Total f-score by', strat, 'averaging is', scores[strat].mean()
