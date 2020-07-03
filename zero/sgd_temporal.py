import numpy as np
import random
import math
from sklearn.metrics import mean_squared_error

class MangakiSGDTemporal:
    def __init__(self, nb_users, nb_works, nb_components=20, nb_iterations=10,
                 gamma=0.01, lambda_=0.1, temporal_hyperparameter = 0.1):
        self.nb_components = nb_components
        self.nb_iterations = nb_iterations
        self.nb_users = nb_users
        self.nb_works = nb_works
        self.gamma = gamma
        self.lambda_ = lambda_
        self.temporal_hyperparameter = temporal_hyperparameter
        # self.bias = np.random.random()
        # self.bias_u = np.random.random(self.nb_users)
        # self.bias_v = np.random.random(self.nb_works)
        #self.nabla_loss = np.zeros((self.nb_users, self.nb_components))
        self.U = np.random.random((self.nb_users, self.nb_components))
        self.V = np.random.random((self.nb_works, self.nb_components))

    def fit_user(self, i, items, ratings, timestamps):
        for index in range(len(items)):
            previous_items = items[:index+1]
            previous_ratings = ratings[:index+1]
            previous_timestamps = timestamps[:index+1]
            item = items[index]
            previousU = self.U[i]
            previousV = self.V[item]
            for j, rating, timestamp in zip(previous_items, previous_ratings, previous_timestamps):
                predicted_rating = self.predict_one(i, j)
                error = predicted_rating - rating
                #self.nabla_loss[i] += error * self.V[j]
                #self.nabla_loss[i] *= self.gamma
                self.U[i] -= self.gamma*error * self.V[j]
            self.U[i] -=  self.lambda_*previousU
            self.V[item] -=  (self.gamma*error * self.U[i] + self.lambda_*previousV)


    def test_user(self, i, items, ratings, timestamps):
        predictions = []
        for index in range(len(items)):
            previous_items = items[:index+1]
            previous_ratings = ratings[:index+1]
            previous_timestamps = timestamps[:index+1]
            previousU = self.U[i]
            for j, rating, timestamp in zip(previous_items, previous_ratings, previous_timestamps):
                predicted_rating = self.predict_one(i, j)
                error = predicted_rating - rating
                #self.nabla_loss[i] += error * self.V[j]
                #self.nabla_loss[i] *= self.gamma
                self.U[i] -= self.gamma*error * self.V[j]
            self.U[i] -=  self.lambda_*previousU
            predictions.append(self.predict_one(i,j))
        return (predictions, ratings) #contains tuples of (estimation, rating) to assemble with ones from other users to compute RMSE

    def test_set(self, users_dict):
        predictions = []
        ratings_predicted = []
        for i in users_dict.keys() :
            user_data = users_dict[i]
            items = [a[0] for a in user_data]
            ratings = [a[1] for a in user_data]
            timestamps = [a[2] for a in user_data]
            (user_predictions, user_ratings) = self.test_user(i, items, ratings, timestamps)
            predictions.append(user_predictions)
            ratings_predicted.append(user_ratings)
        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_ratings = [item for sublist in ratings_predicted for item in sublist]
        return mean_squared_error(flat_predictions, flat_ratings) ** 0.5

    def fit_set(self, users_dict):
        for i in users_dict.keys() :
            user_data = users_dict[i]
            items = [a[0] for a in user_data]
            ratings = [a[1] for a in user_data]
            timestamps = [a[2] for a in user_data]
            self.fit_user(i, items, ratings, timestamps)

    def global_test(self, users, items, ratings, timestamps):
        dicts = self.split_into_training_and_testing_dictionnaries(users, items, ratings, timestamps)
        training_dict = dicts[0]
        testing_dict = dicts[1]
        self.fit_set(training_dict)
        rmse = self.test_set(testing_dict)
        return rmse

    def fit(self, X, y):
        for epoch in range(self.nb_iterations):
            step = 0
            for (i, j), rating in zip(X, y):
                if step % 100000 == 0:  # Pour afficher l'erreur de train
                    y_pred = self.predict(X)
                    print('Train RMSE (epoch={}, step={}): %f'.format(
                        epoch, step, mean_squared_error(y, y_pred) ** 0.5))
                predicted_rating = self.predict_one(i, j)
                error = predicted_rating - rating
                # self.bias += self.gamma * error
                # self.bias_u[i] -= (self.gamma *
                #                    (error + self.lambda_ * self.bias_u[i]))
                # self.bias_v[j] -= (self.gamma *
                #                    (error + self.lambda_ * self.bias_v[j]))
                self.U[i] -= self.gamma * (error * self.V[j] +
                                           self.lambda_ * self.U[i])
                self.V[j] -= self.gamma * (error * self.U[i] +
                                           self.lambda_ * self.V[j])
                step += 1

    def predict_one(self, i, j):
        return (  # self.bias + self.bias_u[i] + self.bias_v[j] +
                self.U[i].dot(self.V[int(j)]))

    def predict(self, X):
        y = []
        for i, j in X:
            y.append(self.predict_one(i, j))
        return np.array(y)

#utils

    def split_list_into_users_dict(self, users, items, ratings, timestamps):
        dict = {}
        for (i,j,r,t) in zip(users, items, ratings, timestamps) :
            if i in dict :
                dict[i].append((j,r,t))
            else :
                dict[i] = [(j,r,t)]
        return dict

    def remove_duplicates(self,l):
        newl = []
        for x in l :
            if x not in newl:
                newl.append(x)
        return newl

    def split_into_training_and_testing_dictionnaries(self, users, items, ratings, timestamps):
        training_dict = {}
        testing_dict = {}
        users_individual = self.remove_duplicates(users)
        random.shuffle(users_individual)
        nb_users = len(users_individual)
        training_users = users_individual[:math.floor(0.8*nb_users)]
        testing_users = users_individual[math.floor(0.8*nb_users):]
        for (i,j,r,t) in zip(users, items, ratings, timestamps) :
            if i in training_users :
                if i in training_dict :
                    training_dict[i].append([j,r,t])
                else :
                    training_dict[i] = [[j,r,t]]
            else :
                if i in testing_dict :
                    testing_dict[i].append([j,r,t])
                else :
                    testing_dict[i] = [[j,r,t]]
        return (training_dict, testing_dict)
        return 2
