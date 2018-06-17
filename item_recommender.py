# load libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.sparse as sparse
import implicit
import numpy as np
from operator import itemgetter
import time

class RecSys():
    def __init__(self):
        self.user_id = None
        self.user_dict = None
        self.item_dict = None
        self.interaction_sparse_matrix = None
        self.predicted_preference = None
        self.recommended_items = None
        self.users = None
        self.items = None
        self.agg_df = None
        self.interaction_index_to_replace = None
        self.predicted_df = None
        self.compare_df = None

        print('Preparing recommender engine...')

    def build_recsys(self, alpha = 40, factors = 100, regularization = 0.01, iterations = 10):
        self.alpha = alpha
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations

        # output parameters
        print('This Implicit Collaborative Filtering model is built using the following parameter values: ')
        print('alpha = ' + str(alpha))
        print('factors = ' + str(factors))
        print('regularization = ' + str(regularization))
        print('no of iterations = ' + str(iterations))

        # create a sparse coordinate matrix
        interaction_sparse_matrix = self.get_item_user_sparse_matrix()

        # create the training data
        train_matrix = self.create_training_set(interaction_sparse_matrix)

        # build item-user confidence matrix weighted by alpha
        item_user_confidence = alpha * train_matrix

        # create the ALS model using the Cython package 'implicit' which allows parallelisation
        als_model = implicit.als.AlternatingLeastSquares(factors=factors,
                                                         regularization=regularization,
                                                         iterations=iterations)

        # train the ALS model on the training data - confidence matrix
        # this step will initialise the user-factors and item-factors vectors
        als_model.fit(item_users=item_user_confidence)

        # build our predicted preference matrix
        user_factors = als_model.user_factors
        item_factors = als_model.item_factors
        predicted_preference = user_factors.dot(item_factors.T)

        # return the predicted preference matrix
        self.predicted_preference = predicted_preference
        print('The recommender engine is ready!')
        print('================================')

    def get_item_user_sparse_matrix(self, data='~/recommender-system/app_recommender-system/data/data.csv'):
        '''
        :param data: path to csv file containing the data
        :return: item-user sparse matrix
        '''

        self.data = data

        # load the csv file
        df = pd.read_csv(filepath_or_buffer=data)

        # assign column names
        df.columns = ['user_id', 'item_id', 'interaction_time']

        # aggregate data so each row is user-item pair interaction
        agg_df = \
            df \
            .groupby(['user_id', 'item_id']) \
            .size() \
            .reset_index(name='user_item_interaction_count') \
            .sort_values('user_item_interaction_count', ascending=False)

        self.agg_df = agg_df

        ## map each item and user to a unique numeric value
        items = agg_df.item_id.astype('category')
        users = agg_df.user_id.astype('category')

        self.items = items
        self.users = users

        ### create mapping dictionary
        item_dict = dict(enumerate(items.cat.categories))
        user_dict = dict(enumerate(users.cat.categories))

        self.item_dict = item_dict
        self.user_dict = user_dict

        ## map no of interaction to integer
        interactions = agg_df.user_item_interaction_count.astype('int')

        # build the item-user matrix
        ## since we are using the implicit library, the rows of the matrix will be items and columns will be users
        sparse_rows = items.cat.codes
        sparse_columns = users.cat.codes

        # create a sparse coordinate matrix
        interaction_sparse_matrix = sparse.csr_matrix((interactions, (sparse_rows, sparse_columns)))

        return interaction_sparse_matrix

    def create_training_set(self, interaction_sparse_matrix, seed_no=9257, percent_masked = 0.2):
        '''
        :param interaction_sparse_matrix: item-user CSR sparse matrix
        :param seed_no: this is for reproducibility
        :param percent_masked: proportion of user-item interactions to be masked
        :return: training set matrix on which the model is trained
        '''

        self.interaction_sparse_matrix = interaction_sparse_matrix
        self.seed_no = seed_no
        self.percent_masked = percent_masked

        # set seed for reproducibility
        np.random.seed(seed_no)

        # copy the original matrix to training matrix
        train_matrix = interaction_sparse_matrix.copy()

        # identify indices of the purchase sparse matrix where the element is not zero
        existing_interaction_index = np.transpose(np.nonzero(train_matrix))

        interaction_index_to_replace = train_test_split(existing_interaction_index,
                                                        test_size=percent_masked)[1]  # mask % of user-item interactions
        interaction_index_to_replace = np.transpose(interaction_index_to_replace) # so we can subset our matrix

        self.interaction_index_to_replace = interaction_index_to_replace

        # replace with 0's
        train_matrix[interaction_index_to_replace[0], interaction_index_to_replace[1]] = 0

        return train_matrix

    def recommend_items(self, user_id, n_items = 10):
        '''
        :param user_id: user id to give item recommendations to
        :param n_items: number of recommended items to give to user id
        :return: n recommended items
        '''
        self.user_id = user_id
        self.n_items = n_items

        user_dict = self.user_dict
        item_dict = self.item_dict
        predicted_preference = self.predicted_preference

        user_index = list(user_dict.keys())[list(user_dict.values()).index(user_id)]
        user_item_preference = predicted_preference[user_index]

        # sorted_user_item_preference = -np.sort(-user_item_preference) # this would get the score, irrelevant for now

        sorted_preference_index = np.argsort(-user_item_preference)[0:n_items]
        recommended_items = list(itemgetter(*sorted_preference_index)(item_dict))

        # recommended_items = list(zip(recommended_items, sorted_user_item_preference)) # if score is included, can use this

        print('Here\'s the top 10 recommended items for user %s' % (user_id))
        for i in range(n_items):
            print(str(i + 1) + '. ' + str(recommended_items[i]))

        return recommended_items

    def get_predicted_df(self):
        '''
        transform the predicted matrix to data frame
        to contain user id and 10 recommended items for each user id
        '''

        # set empty list to contain user and recommended items
        user_test = []
        item_test = []
        n = 10 # to repeat user id for 10 times

        # get list of all users
        users = self.users
        unique_users = users.unique()

        # for each user, get the top 10 recommended items
        for user in unique_users:
            rec = self.recommend_items(user)
            item_test.extend(rec)

            u = [user for _ in range(n)]
            user_test.extend(u)

        # store the result in data frame
        predicted_df = pd.DataFrame({'user_id': user_test, 'item_id': item_test})
        predicted_df.sort_values('user_id', inplace=True)

        return predicted_df

    def recall_measure(self):
        '''
        the ratio between the number of items that were correctly predicted by
        the recommender system and the actual items purchased by users
        '''

        predicted_df = self.get_predicted_df()
        agg_df = self.agg_df

        agg_df_to_compare = agg_df.loc[:, ['user_id', 'item_id']]
        agg_df_to_compare.sort_values('user_id', inplace=True)

        compare_df = pd.merge(agg_df_to_compare, predicted_df, on=['user_id', 'item_id'],
                              how='left', indicator='predicted')

        self.compare_df = compare_df

        # recall % of the relevant items were recommended in the top 10 items
        recall_accuracy = round(sum(compare_df.predicted == 'both') / agg_df_to_compare.shape[0] * 100, 1)

        return recall_accuracy

    def precision_measure(self):
        '''
        the ratio between the number of items that were correctly predicted by
        the recommender system and the number of items recommended to the user
        '''

        predicted_df = self.get_predicted_df()
        precision_df = self.compare_df

        # precision % of the recommended items
        precision = round(sum(precision_df.predicted == 'both') / predicted_df.shape[0] * 100, 1)

        return precision

    def masked_predicted_measure(self):
        '''
        measures if the user ended up buying any of the items our system recommended,
        ie. focusing on the users in the training set
        where some of their purchased items were masked to build the model
        '''

        predicted_df = r.get_predicted_df()
        interaction_index_to_replace = self.interaction_index_to_replace # indexes of item-user sparse matrix
        # that were replaced with 0's for the training set

        user_dict = self.user_dict
        item_dict = self.item_dict

        # get list of users of which some of their items were masked
        user_replaced_index = interaction_index_to_replace[1]
        user_replaced = list(itemgetter(*user_replaced_index)(user_dict))

        # get list of items which were masked
        item_replaced_index = interaction_index_to_replace[0]
        item_replaced = list(itemgetter(*item_replaced_index)(item_dict))

        # create a data frame of these masked user-item pairs
        user_item_replaced = pd.DataFrame({'user_id': user_replaced,
                                           'item_id': item_replaced})
        user_item_replaced.sort_values('user_id', inplace=True)

        # merge the data frame to compare with the predicted df
        compare_df_for_auc = pd.merge(user_item_replaced, predicted_df, on=['user_id', 'item_id'],
                                      how='left', indicator='predicted')

        # calculate simplified auc
        masked_predicted = round(sum(compare_df_for_auc.predicted == 'both') / user_item_replaced.shape[0] * 100, 1)

        return masked_predicted
###### end of Class definition ######



###### set up the command line interface ######
if __name__ == "__main__":
    r = RecSys() # initiate the recsys class
    r.build_recsys() # build the model

    # take input from user and return recommended items
    try:
        while True:
            user_id_input = int(input("Who do you want to recommend items to: "))
            print('-----')
            try:
                r.recommend_items(user_id_input)
                print('================================')
                time.sleep(1)
            except ValueError:
                print('Oops that user_id is not in our database! Try another one...')
                print('================================')
    except KeyboardInterrupt:
        print('Hope that was useful and relevant! See you next time!')
###### end of application ######