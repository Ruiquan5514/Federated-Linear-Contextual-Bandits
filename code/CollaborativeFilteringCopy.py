import numpy as np
import pandas as pd
import pdb


def load_ratings(rating_file):
    """
    return `ratings`: shape [n_ratings, 3], each row is of the form [uid, iid, r] where uid and iid start from 0
    """
    df = pd.read_csv(rating_file, sep='\t', header=None, names=['user id', 'movie id', 'rating', 'timestamp'])
    ratings = np.array(df)
    ratings[:,:2] -= 1 # user_id and item_id start from 1 so subtract 1
    ratings = ratings[:,:3] # drop last column (timestamp)

    return ratings


class CollaborativeFiltering():
    """
    A Recommender System that uses item-based or user-based collaborative filtering (cf) as specified
    """
    def __init__(self, rating_matrix, n, item_based=True):
        """
        rating_matrix: [n_users x n_items] matrix
        n: predict ratings based on `n` most similar items/users
        item_based: if True, then item-item cf is used otherwise user-user cf is used (default: True)
        """
        self.rating_matrix = rating_matrix
        self.n_users = self.rating_matrix.shape[0]
        self.n_items = self.rating_matrix.shape[1]
        self.n = n
        self.item_based = item_based

    def pearson_correlation_sim(self, id1, id2):
        """
        calculative pearson correlation similarity score for item/user id1, id2 (index starts with 0)
        """
        if self.item_based:
            ratings_x = self.rating_matrix[:,id1]
            ratings_y = self.rating_matrix[:,id2]
        else:
            ratings_x = self.rating_matrix[id1,:]
            ratings_y = self.rating_matrix[id2,:]

        common_ratings = np.nonzero(np.multiply(ratings_x, ratings_y))

        if len(common_ratings) == 0:
            return 0

        rx_minus_mean = ratings_x[common_ratings] - ratings_x.mean()
        ry_minus_mean = ratings_y[common_ratings] - ratings_y.mean()

        denominator_val = (np.linalg.norm(rx_minus_mean) * np.linalg.norm(ry_minus_mean))
        if denominator_val == 0:
            return 0

        return np.sum(np.multiply(rx_minus_mean, ry_minus_mean)) / denominator_val

    def n_most_similar_items(self, item_id, rated_by_user_id):
        """
        find `n` most similar items to item `item_id` (index starts from 0) based on pearson correlation similarity which are rated by user `rated_by_user_id`
        return: ids of `n` most similar items to item `item_id`, their respective similarity scores
        """
        # similarities is list of [id,sim] for those items rated by `user_id`. sim is similarity of `item_id` and id
        similarities = [ [id2,self.pearson_correlation_sim(item_id,id2)] for id2 in range(self.n_items) if not self.rating_matrix[rated_by_user_id, id2] == 0 and not id2 == item_id ]
        similarities = np.array(sorted(similarities, key = lambda x: x[1], reverse=True))[:self.n]

        if similarities.shape[0] == 0: # no similar found
            return [],[]

        n_most_similar = similarities[:,0].astype(int)
        similarity_scores = similarities[:,1]

        return n_most_similar, similarity_scores

    def n_most_similar_users(self, user_id, rated_item_id):
        """
        find `n` most similar users to item `user_id` (index starts from 0) based on pearson correlation similarity who have rated item `rated_item_id`
        return: ids of `n` most similar users to user `user_id`, their respective similarity scores
        """
        # similarities is list of [id,sim] for those items rated by `user_id`. sim is similarity of `item_id` and id
        similarities = [ [id2,self.pearson_correlation_sim(user_id,id2)] for id2 in range(self.n_users) if not self.rating_matrix[id2, rated_item_id] == 0 and not id2 == user_id ]
        similarities = np.array(sorted(similarities, key = lambda x: x[1], reverse=True))[:self.n]
        
        if similarities.shape[0] == 0: # no similar found
            return [],[]

        n_most_similar = similarities[:,0].astype(int)
        similarity_scores = similarities[:,1]

        return n_most_similar, similarity_scores

    def predict_rating(self, user_id, item_id):
        """
        predict rating of `user_id` to `item_id` based on weighted avg of `n` most similar items/users
        """
        if self.item_based:
            n_most_similar, similarity_scores = self.n_most_similar_items(item_id, user_id)
            if sum(similarity_scores) == 0:
                return 0
            ratings_of_n_most_similar = self.rating_matrix[user_id,n_most_similar] # ratings by `user_id` for n most similar items to `item_id`
        else:
            n_most_similar, similarity_scores = self.n_most_similar_users(user_id, item_id)
            if sum(similarity_scores) == 0:
                return 0
            ratings_of_n_most_similar = self.rating_matrix[n_most_similar, item_id] # ratings by n most similar users for `item_id`

        predicted_rating = np.average(ratings_of_n_most_similar, weights=similarity_scores)
        return predicted_rating

    def calc_rmse(self, test_ratings):
        """
        test_ratings: [n_test_ratings x 3] array where each row of the form [user_id, item_id, r]
        """
        actual = test_ratings[:,2]
        predicted = np.array([self.predict_rating(uid,iid) for uid,iid,_ in test_ratings])
        print(predicted)
        rmse = np.sqrt(np.mean((actual - predicted)**2))
        return rmse


if __name__ == "__main__":
    train_ratings = load_ratings('movielens-100k/ua.base')
    test_ratings = load_ratings('movielens-100k/ua.test')

    n_users, n_items = 943, 1682

    print('No. of users:', n_users)
    print('No. of items:', n_items)
    print('No. of train ratings:', train_ratings.shape[0])
    print('No. of test ratings:', test_ratings.shape[0])

    rating_matrix = np.zeros((n_users, n_items), np.int8)
    for uid, iid, r in train_ratings:
        rating_matrix[uid, iid] = r
    
    neighbor_size = 10
    #recommender = CollaborativeFiltering(rating_matrix, n=neighbor_size, item_based=True)
    recommender = CollaborativeFiltering(rating_matrix, n=neighbor_size, item_based=False)
    
    prediction = np.zeros([n_users,n_items])
    for i in range(n_users):
        print(i)
        for a in range(n_items):
            prediction[i,a] = recommender.predict_rating(i,a)
    df = pd.DataFrame(prediction)
    df.to_csv(complete_ratings)
    #pdb.set_trace()
    
    #print('\nCalculating Root Mean Squared Error (rmse) on Test Ratings using Collaborative Filtering...')
    #print('rmse:', recommender.calc_rmse(test_ratings))
