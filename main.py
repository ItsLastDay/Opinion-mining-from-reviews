from data_extractor import get_nice_data
#from solution import Solution
from sklearn import metrics, cross_validation


train_data = get_nice_data('reviews.json')
print train_data[1]
#sol = Solution()

#scores = cross_validation.cross_val_score(sol, train_data[0], train_data[1], \
#        cv=3, scoring='f1')
