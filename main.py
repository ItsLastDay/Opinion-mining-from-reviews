from data_extractor import get_nice_data
from solution import Solution


train_data = get_nice_data('reviews.json')
sol = Solution()
sol.train(train_data)
