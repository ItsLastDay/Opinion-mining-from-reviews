from data_extractor import get_nice_data, clean_answer
from solution import Solution
from sklearn import metrics, cross_validation


train_data = get_nice_data('reviews.json')
train_data = list(map(lambda x: x[:100], train_data))
sol = Solution()
sol.train(train_data)

#clean_answer(sol.predict(train_data[0][0]), 'predicted:', 'w')
#clean_answer(train_data[1][0], 'answer:', 'a')
scores = cross_validation.cross_val_score(sol, train_data[0], train_data[1], \
        cv=3)
