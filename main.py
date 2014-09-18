from data_extractor import get_nice_data, clean_answer
from solution import Solution
from sklearn.metrics import f1_score
from sklearn.cross_validation import KFold
import numpy as np


train_data = get_nice_data('reviews.json')
train_data = list(map(lambda x: np.array(x), train_data))

scores = np.array([])
for train_idx, test_idx in KFold(len(train_data[0]), n_folds=10):
    X_train = train_data[0][train_idx]
    Y_train = train_data[1][train_idx]

    X_test = train_data[0][test_idx]
    Y_test = train_data[1][test_idx]

    sol = Solution()
    sol.train((X_train, Y_train))

    answer = sol.getClasses(X_test)

    score = f1_score(Y_test, answer, average='samples')
    print score
    scores = np.append(scores, score)

print 'Total f-score:', scores.mean()
