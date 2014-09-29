from data_extractor import get_nice_data, clean_answer
from solution import Solution
from sklearn.metrics import f1_score
from sklearn.cross_validation import KFold
import numpy as np
import sys


train_data = get_nice_data('reviews.json')
train_data = list(map(lambda x: np.array(x[:1000]), train_data))

scores = []
for train_idx, test_idx in KFold(len(train_data[0]), n_folds=10, \
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
    
    current_score = []
    presented_classes = set()
    for x in Y_test:
        presented_classes.update(set(x))

    for cl in sorted(presented_classes):
        bin_test = [1 if cl in x else 0 for x in Y_test]
        bin_ans = [1 if cl in x else 0 for x in answer]
        current_score.append(f1_score(bin_test, bin_ans))

    for (i, cl) in enumerate(sorted(presented_classes)):
        bin_test = [1 if cl in x else 0 for x in Y_test]
        bin_ans = [1 if cl in x else 0 for x in answer]
        print cl[0], cl[1], 'occured in answer', sum(bin_test), 'times, and in my', \
                sum(bin_ans), 'times; f-score is', current_score[i]
    score = np.array(current_score).mean()
    print score

    scores.append(score)
    '''

    total_answers_test = sum([len(x) for x in Y_test])
    total_answers_system = sum([len(x) for x in answer])

    correct_answers = 0
    for i in xrange(len(Y_test)):
        my = set(answer[i])
        ans = set(Y_test[i])
        
        correct_answers += len(my.intersection(ans))

    print 'Fold calculated:'
    print correct_answers, total_answers_test, total_answers_system
    precision = float(correct_answers) / total_answers_system
    recall = float(correct_answers) / total_answers_test
    f_m = 2 * precision * recall / (precision + recall)
    print precision, recall, f_m

    scores.append(f_m)
    '''

print 'Total score is:', np.array(scores).mean()
