# import Class RecSys
from item_recommender import RecSys

# initial test before altering any model parameters
parameters = [40, 100, 0.01, 10]
alpha = parameters[0]
factors = parameters[1]
regularization = parameters[2]
iterations = parameters[3]

r = RecSys()
r.build_recsys(alpha=alpha,
               factors=factors,
               regularization=regularization,
               iterations=iterations)
recall = r.recall_measure() # 59.3%
precision = r.precision_measure() # 34.7%
masked_predicted = r.masked_predicted_measure() # 39.9%

# test after altering any model parameters - see below series of testing
parameters = [590, 200, 0.01, 10]
alpha = parameters[0]
factors = parameters[1]
regularization = parameters[2]
iterations = parameters[3]

r = RecSys()
r.build_recsys(alpha=alpha,
               factors=factors,
               regularization=regularization,
               iterations=iterations)
recall = r.recall_measure() # 63.0%
precision = r.precision_measure() # 36.9%
masked_predicted = r.masked_predicted_measure() # 21.4% - most important metric
# it seems like the initial parameter values will be kept to keep the balance between the performance metrics,
# as the masked predicted metric will be significantly lower if using the altered parameter values

###### Recall Measure Performance Optimisation Test - START ######
# testing for alpha
r = RecSys()
alpha_list = []
alpha_recall_list = []

for i in range(10,1100,10):
    alpha = i
    print('testing alpha = ' + str(i))
    alpha_list.append(i)

    r.build_recsys(alpha=alpha,
                   factors=factors,
                   regularization=regularization,
                   iterations=iterations)
    recall = r.recall_measure()
    print('recall measure is ' + str(recall))
    alpha_recall_list.append(recall)

    print('NEXT')

alpha_recall_df = pd.DataFrame({'alpha': alpha_list,
                                'recall_accuracy': alpha_recall_list})

# testing number of factors
r = RecSys()

alpha = 40 # 40 or 590
factors_list = []
factors_recall_list = []

for i in range(50,300,50):
    factors = i
    print('testing factors = ' + str(i))
    factors_list.append(i)

    r.build_recsys(alpha=alpha,
                   factors=factors,
                   regularization=regularization,
                   iterations=iterations)
    recall = r.recall_measure()
    print('recall measure is ' + str(recall))
    factors_recall_list.append(recall)

    print('NEXT')

factors_recall_df = pd.DataFrame({'factors': factors_list,
                                  'recall_accuracy': factors_recall_list})

# testing the regularization term
r = RecSys()

alpha = 40 # 40 or 590
factors = 100 # 100 or 200
regularization_list = []
regularization_recall_list = []

for i in np.arange(0.01, 0.1, 0.01):
    regularization = i
    print('testing regularization = ' + str(i))
    regularization_list.append(i)

    r.build_recsys(alpha=alpha,
                   factors=factors,
                   regularization=regularization,
                   iterations=iterations)
    recall = r.recall_measure()
    print('recall measure is ' + str(recall))
    regularization_recall_list.append(recall)

    print('NEXT')

regularization_recall_df = pd.DataFrame({'regularization': regularization_list,
                                         'recall_accuracy': regularization_recall_list})

# testing the number of iterations
r = RecSys()

alpha = 40 # 40 or 590
factors = 100 # 100 or 200
regularization = 0.01
iterations_list = []
iterations_recall_list = []

for i in range(5,20,1):
    iterations = i
    print('testing iterations = ' + str(i))
    iterations_list.append(i)

    r.build_recsys(alpha=alpha,
                   factors=factors,
                   regularization=regularization,
                   iterations=iterations)
    recall = r.recall_measure()
    print('recall measure is ' + str(recall))
    iterations_recall_list.append(recall)

    print('NEXT')

iterations_recall_df = pd.DataFrame({'iterations': iterations_list,
                                     'recall_accuracy': iterations_recall_list})
###### Recall Measure Performance Optimisation Test - END ######