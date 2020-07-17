import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class MangakiSGDTemporal:
    def __init__(self, nb_users, nb_works, nb_components=20, nb_iterations=20,
                 gamma=0.01, lambda_=0.1, temporal_hyperparameter = 0.1, dynamic_strategy = "no temporal factor"):
        self.nb_components = nb_components
        self.nb_iterations = nb_iterations
        self.nb_users = nb_users
        self.nb_works = nb_works
        self.gamma = gamma
        self.lambda_ = lambda_
        self.temporal_hyperparameter = temporal_hyperparameter
        self.temporal_hyperparameter_beta = temporal_hyperparameter
        self.temporal_hyperparameter_c = temporal_hyperparameter
        self.dynamic_strategy = dynamic_strategy
        # self.bias = np.random.random()
        # self.bias_u = np.random.random(self.nb_users)
        # self.bias_v = np.random.random(self.nb_works)
        self.U = np.random.random((self.nb_users, self.nb_components))
        self.V = np.random.random((self.nb_works, self.nb_components))
        self.loss = 0

    def get_temp_factor(self, time_spent):
        if self.dynamic_strategy == "exponential":
            return math.exp(abs(self.temporal_hyperparameter) * (-time_spent))
        elif self.dynamic_strategy == "exponential factor":
            return  math.exp(abs(self.temporal_hyperparameter_beta) * (-time_spent)) * abs(self.temporal_hyperparameter_c)
        elif self.dynamic_strategy == "no temporal factor":
            return 1

    def get_temp_gradient(self, error, time_spent, temp_factor):
        if self.dynamic_strategy == "exponential":
            return (self.gamma)*(error*error*(-time_spent)*temp_factor
            + self.lambda_ *self.temporal_hyperparameter)
        elif self.dynamic_strategy == "exponential factor":
            return ((self.gamma)*(error*error*(-time_spent)*temp_factor + self.lambda_ *self.temporal_hyperparameter_beta), self.gamma/100 * (error * error * math.exp(abs(self.temporal_hyperparameter_beta) * (-time_spent)) + self.lambda_*self.temporal_hyperparameter_c))
        elif self.dynamic_strategy == "no temporal factor":
            return 1

    '''def fast_fit_user(self, i, items, ratings, timestamps):
        first_element = True
        u_gradient = np.zeros(self.nb_components)
        for index in range(len(items)):
            if first_element :
                time_spent = 0
                first_element = False
            else:
                time_spent = timestamps[index] - timestamps[index-1]
            temp_factor = self.get_temp_factor(time_spent)
            j = items[index]
            predicted_rating = self.predict_one(i, j)
            error = predicted_rating - ratings[index]
            u_gradient = u_gradient * temp_factor
            u_gradient += self.gamma*error * self.V[j]
            self.U[i] -= (u_gradient + self.lambda_ * self.U[i])
            self.V[j] -= self.gamma*(error * self.U[i]*temp_factor + self.lambda_ *self.V[j])
            if self.dynamic_strategy == "exponential factor" :
                (delta_beta,delta_c)=self.get_temp_gradient(error, time_spent, temp_factor)
                self.temporal_hyperparameter_c -= delta_c
                self.temporal_hyperparameter_beta -= delta_beta
            elif self.dynamic_strategy == "exponential":
                self.temporal_hyperparameter -= self.get_temp_gradient(error, time_spent, temp_factor)
            predictions.append(self.predict_one(i,j))


    def fit_user(self, i, items, ratings, timestamps):
        for index in range(len(items)):
            previous_items = items[:index+1]
            previous_ratings = ratings[:index+1]
            previous_timestamps = timestamps[:index+1]
            item = items[index]
            #for subindex in range(index+1):
            #    time_spent = timestamps[index] - timestamps[subindex]
            #    temp_factor = self.get_temp_factor(time_spent)
            #    j = items[subindex]
            #    predicted_rating = self.predict_one(i, j)
            #    error = predicted_rating - ratings[subindex]
            #    self.U[i] -= self.gamma*error * self.V[j]*temp_factor
            #    self.V[j] -= self.gamma*error * self.U[i]*temp_factor
            #    self.temporal_hyperparameter -= self.get_temp_gradient(error, time_spent, temp_factor)
            for j, rating, timestamp in zip(previous_items, previous_ratings, previous_timestamps):
                time_spent = timestamps[index] - timestamp
                temp_factor = self.get_temp_factor(time_spent)
                predicted_rating = self.predict_one(i, j)
                error = predicted_rating - rating
                self.U[i] -= self.gamma*(error * self.V[j]*temp_factor + self.lambda_ *self.U[i])
                self.V[j] -= self.gamma*(error * self.U[i]*temp_factor + self.lambda_ *self.V[j])
                self.temporal_hyperparameter -= self.get_temp_gradient(error, time_spent, temp_factor)

    def fast_test_user(self, i, items, ratings, timestamps):
        first_element = True
        predictions = []
        u_gradient = np.zeros(self.nb_components)
        for index in range(len(items)):
            if first_element :
                time_spent = 0
                first_element = False
            else:
                time_spent = timestamps[index] - timestamps[index-1]
            temp_factor = self.get_temp_factor(time_spent)
            j = items[index]
            predicted_rating = self.predict_one(i, j)
            error = predicted_rating - ratings[index]
            u_gradient = u_gradient * temp_factor
            u_gradient += self.gamma*error * self.V[j]
            self.U[i] -= (u_gradient + self.lambda_ * self.U[i])
            predictions.append(self.predict_one(i,j))
        return (predictions, ratings)


    def test_user(self, i, items, ratings, timestamps):
        predictions = []
        for index in range(len(items)):
            previous_items = items[:index+1]
            previous_ratings = ratings[:index+1]
            previous_timestamps = timestamps[:index+1]
            previousU = self.U[i]
            for j, rating, timestamp in zip(previous_items, previous_ratings, previous_timestamps):
                time_spent = timestamps[index] - timestamp
                temp_factor = self.get_temp_factor(time_spent)
                predicted_rating = self.predict_one(i, j)
                error = predicted_rating - rating
                self.U[i] -= self.gamma*(error * self.V[j]*temp_factor + self.lambda_ *self.U[i])
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
        print("initialization done")
        self.fit_set(training_dict)
        print("training done")
        print(self.temporal_hyperparameter)
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
                step += 1'''

    def predict_one(self, i, j):
        return (  # self.bias + self.bias_u[i] + self.bias_v[j] +
                self.U[i].dot(self.V[int(j)]))

    def predict(self, X):
        y = []
        for i, j in X:
            y.append(self.predict_one(i, j))
        return np.array(y)

#utils

    def add_in_ordered_list_of_6_uplets(self, list, item):
        (a,b,c,d,e,f) = item
        i=0
        while i<len(list) and c > list[i][2] :
            i+=1
        list.insert(i, item)

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

    def split_into_training_and_testing_dictionnaries(self, users, items, ratings, timestamps, mean_of_user, std_of_user, scaled_ratings):
        training_dict = {}
        testing_dict = {}
        users_individual = self.remove_duplicates(users)
        random.shuffle(users_individual)
        nb_users = len(users_individual)
        training_users = users_individual[:math.floor(0.8*nb_users)]
        testing_users = users_individual[math.floor(0.8*nb_users):]
        for (i,j,r,t,m,sd,sr) in zip(users, items, ratings, timestamps, mean_of_user, std_of_user, scaled_ratings) :
            if i in training_users :
                if i in training_dict :
                    self.add_in_ordered_list_of_6_uplets(training_dict[i],(j,r,t,m,sd,sr))
                else :
                    training_dict[i] = [(j,r,t,m,sd,sr)]
            else :
                if i in testing_dict :
                    self.add_in_ordered_list_of_6_uplets(testing_dict[i],(j,r,t,m,sd,sr))
                else :
                    testing_dict[i] = [(j,r,t,m,sd,sr)]
        return (training_dict, testing_dict)

    def square_norm(self,U):
        n = len(U)
        sn = 0
        for i in range(n):
            sn += U[i]**2
        return sn

    def normalize_timestamps(self, timestamps):
        max = 0
        for x in timestamps:
            if x> max:
                max = x
        for i in range(len(timestamps)):
            timestamps[i] = timestamps[i]/max
        print(max)

#####################################################################################################

    def new_test_user(self, i, items, ratings, timestamps, mean_of_user, std_of_user, scaled_ratings):
        predictions = []
        for j, rating, timestamp, mean, sd, sr in zip(items, ratings, timestamps, mean_of_user, std_of_user, scaled_ratings):
            time_spent = timestamps[-1] - timestamp
            temp_factor = self.get_temp_factor(time_spent)
            predicted_rating = self.predict_one(i, j)
            error = predicted_rating - sr
            predictions.append(self.predict_one(i,j) * sd + mean)
            self.U[i] -= self.gamma*(error * self.V[j]*temp_factor + self.lambda_ *self.U[i])
            self.loss += temp_factor*error*error
        return (predictions, ratings) #contains tuples of (estimation, rating) to assemble with ones from other users to compute RMSE

    def new_fit_user(self, i, items, ratings, timestamps, mean_of_user, std_of_user, scaled_ratings):
        predictions = []
        for j, rating, timestamp, mean, sd, sr in zip(items, ratings, timestamps, mean_of_user, std_of_user, scaled_ratings):
            time_spent = timestamps[-1] - timestamp
            temp_factor = self.get_temp_factor(time_spent)
            predicted_rating = self.predict_one(i, j)
            error = (predicted_rating - sr)#*temp_factor
            self.U[i] -= self.gamma*(error * self.V[j]*temp_factor
            + self.lambda_ *self.U[i])
            self.V[j] -= self.gamma*(error * self.U[i]*temp_factor
            + self.lambda_ *self.V[j])
            if self.dynamic_strategy == "exponential factor" :
                (delta_beta,delta_c)=self.get_temp_gradient(error, time_spent, temp_factor)
                self.temporal_hyperparameter_c -= delta_c
                self.temporal_hyperparameter_beta -= delta_beta
            elif self.dynamic_strategy == "exponential":
                self.temporal_hyperparameter -= self.get_temp_gradient(error, time_spent, temp_factor)
            predictions.append(self.predict_one(i,j) + mean)
            self.loss += temp_factor*error*error
        return (predictions, ratings)

    def new_test_set(self, users_dict):
        predictions = []
        ratings_predicted = []
        for i in users_dict.keys() :
            user_data = users_dict[i]
            items = [a[0] for a in user_data]
            ratings = [a[1] for a in user_data]
            timestamps = [a[2] for a in user_data]
            mean_of_user = [a[3] for a in user_data]
            std_of_user = [a[4] for a in user_data]
            scaled_ratings = [a[5] for a in user_data]
            (user_predictions, user_ratings) = self.new_test_user(i, items, ratings, timestamps, mean_of_user, std_of_user, scaled_ratings)
            predictions.append(user_predictions)
            ratings_predicted.append(user_ratings)
        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_ratings = [item for sublist in ratings_predicted for item in sublist]
        return mean_squared_error(flat_predictions, flat_ratings) ** 0.5

    def new_fit_set(self, users_dict):
        predictions = []
        ratings_predicted = []
        for i in users_dict.keys() :
            user_data = users_dict[i]
            items = [a[0] for a in user_data]
            ratings = [a[1] for a in user_data]
            timestamps = [a[2] for a in user_data]
            mean_of_user = [a[3] for a in user_data]
            std_of_user = [a[4] for a in user_data]
            scaled_ratings = [a[5] for a in user_data]
            (user_predictions, user_ratings) = self.new_fit_user(i, items, ratings, timestamps, mean_of_user, std_of_user, scaled_ratings)
            predictions.append(user_predictions)
            ratings_predicted.append(user_ratings)
        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_ratings = [item for sublist in ratings_predicted for item in sublist]
        return mean_squared_error(flat_predictions, flat_ratings) ** 0.5

    def new_global_test(self, users, items, ratings, timestamps, mean_of_user, std_of_user, scaled_ratings, dynamic_strategy = "no temporal factor"):
        self.dynamic_strategy = dynamic_strategy
        #self.normalize_timestamps(timestamps)
        dicts = self.split_into_training_and_testing_dictionnaries(users, items, ratings, timestamps, mean_of_user, std_of_user, scaled_ratings)
        training_dict = dicts[0]
        testing_dict = dicts[1]
        print("initialization done")
        train_rmse_list =[]
        test_rmse_list = []
        loss_list = []
        for i in range(self.nb_iterations):
            train_rmse = self.new_fit_set(training_dict)
            if self.dynamic_strategy == "exponential factor" :
                print(self.temporal_hyperparameter_c)
                print(self.temporal_hyperparameter_beta)
            elif self.dynamic_strategy == "exponential" :
                print(self.temporal_hyperparameter)
            print("training rmse = " + str(train_rmse))
            test_rmse = self.new_test_set(testing_dict)
            print("testing rmse = " + str(test_rmse))
            train_rmse_list.append(train_rmse)
            test_rmse_list.append(test_rmse)
            for i in range(self.nb_users):
                self.loss += self.square_norm(self.U[i])
            for i in range(self.nb_works):
                self.loss += self.square_norm(self.V[i])
            if self.dynamic_strategy == "exponential factor" :
                self.loss += self.lambda_ * self.temporal_hyperparameter_c * self.temporal_hyperparameter_c
                self.loss += self.lambda_ * self.temporal_hyperparameter_beta * self.temporal_hyperparameter_beta
            elif self.dynamic_strategy == "exponential" :
                self.loss += self.lambda_ * self.temporal_hyperparameter * self.temporal_hyperparameter
            loss_list.append(self.loss)
            self.loss = 0
        iterations_array = np.arange(self.nb_iterations)
        f = plt.figure(figsize=(6, 8))
        plt.plot(iterations_array, train_rmse_list, label = "train rmse")
        plt.plot(iterations_array, test_rmse_list, label = "test rmse")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        axes = plt.gca()
        axes.set_xlim([0, self.nb_iterations])
        axes.set_ylim([0,2])
        g = plt.figure(figsize=(6, 8))
        plt.plot(iterations_array, loss_list, label = "loss")
        return test_rmse

##################################################################################

    def cl_test_user(self, i, items, ratings, timestamps, mean_of_user, std_of_user, scaled_ratings):
        predictions = []
        previous_ratings = []
        for j, rating, timestamp, mean, sd, sr in zip(items, ratings, timestamps, mean_of_user, std_of_user, scaled_ratings):
            U = self.cl_compute_u(i, j, previous_ratings)
            predicted_rating = U.dot(self.V[int(j)])
            error = predicted_rating - sr
            predictions.append(predicted_rating+mean)
            self.loss += error*error
            previous_ratings.append((j, rating,timestamp, mean, sd, sr))
        return (predictions, ratings) #contains tuples of (estimation, rating) to assemble with ones from other users to compute RMSE

    def cl_fit_user(self, i, items, ratings, timestamps, mean_of_user, std_of_user, scaled_ratings):
        predictions = []
        previous_ratings = []
        for j, rating, timestamp, mean, sd, sr in zip(items, ratings, timestamps, mean_of_user, std_of_user, scaled_ratings):
            U = self.cl_compute_u(i, j, previous_ratings)
            predicted_rating = U.dot(self.V[int(j)])
            error = predicted_rating - sr
            self.V[j] -= self.gamma*(error * self.U[i]
            + self.lambda_ *self.V[j])
            #print(U)
            predictions.append(predicted_rating+ mean)
            self.loss += error*error
            previous_ratings.append((j, rating,timestamp, mean, sd, sr))
        return (predictions, ratings)

    def cl_test_set(self, users_dict):
        predictions = []
        ratings_predicted = []
        for i in users_dict.keys() :
            user_data = users_dict[i]
            items = [a[0] for a in user_data]
            ratings = [a[1] for a in user_data]
            timestamps = [a[2] for a in user_data]
            mean_of_user = [a[3] for a in user_data]
            std_of_user = [a[4] for a in user_data]
            scaled_ratings = [a[5] for a in user_data]
            (user_predictions, user_ratings) = self.cl_test_user(i, items, ratings, timestamps, mean_of_user, std_of_user, scaled_ratings)
            predictions.append(user_predictions)
            ratings_predicted.append(user_ratings)
        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_ratings = [item for sublist in ratings_predicted for item in sublist]
        return mean_squared_error(flat_predictions, flat_ratings) ** 0.5

    def cl_fit_set(self, users_dict):
        predictions = []
        ratings_predicted = []
        for i in users_dict.keys() :
            user_data = users_dict[i]
            items = [a[0] for a in user_data]
            ratings = [a[1] for a in user_data]
            timestamps = [a[2] for a in user_data]
            mean_of_user = [a[3] for a in user_data]
            std_of_user = [a[4] for a in user_data]
            scaled_ratings = [a[5] for a in user_data]
            (user_predictions, user_ratings) = self.cl_fit_user(i, items, ratings, timestamps, mean_of_user, std_of_user, scaled_ratings)
            predictions.append(user_predictions)
            ratings_predicted.append(user_ratings)
        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_ratings = [item for sublist in ratings_predicted for item in sublist]
        return mean_squared_error(flat_predictions, flat_ratings) ** 0.5

    def cl_global_test(self, users, items, ratings, timestamps, mean_of_user, std_of_user, scaled_ratings, dynamic_strategy = "no temporal factor"):
        self.dynamic_strategy = dynamic_strategy
        #self.normalize_timestamps(timestamps)
        dicts = self.split_into_training_and_testing_dictionnaries(users, items, ratings, timestamps, mean_of_user, std_of_user, scaled_ratings)
        training_dict = dicts[0]
        testing_dict = dicts[1]
        print("initialization done")
        train_rmse_list =[]
        test_rmse_list = []
        loss_list = []
        for i in range(self.nb_iterations):
            train_rmse = self.cl_fit_set(training_dict)
            print("training rmse = " + str(train_rmse))
            test_rmse = self.cl_test_set(testing_dict)
            print("testing rmse = " + str(test_rmse))
            train_rmse_list.append(train_rmse)
            test_rmse_list.append(test_rmse)
            for i in range(self.nb_works):
                self.loss += self.square_norm(self.V[i])
            loss_list.append(self.loss)
            self.loss = 0
        iterations_array = np.arange(self.nb_iterations)
        f = plt.figure(figsize=(6, 8))
        plt.plot(iterations_array, train_rmse_list, label = "train rmse")
        plt.plot(iterations_array, test_rmse_list, label = "test rmse")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        axes = plt.gca()
        axes.set_xlim([0, self.nb_iterations])
        axes.set_ylim([0,2])
        g = plt.figure(figsize=(6, 8))
        plt.plot(iterations_array, loss_list, label = "loss")
        return test_rmse

    def cl_compute_u(self, i, j, previous_ratings):
        u = np.zeros(self.nb_components)
        if previous_ratings != [] :
            (_,_,t,_,_,_) = previous_ratings[-1]
        for (item, rating, timestamp, mean, sd, sr) in previous_ratings:
            u += self.get_temp_factor(t-timestamp)*sr*self.V[item]
        norm = math.sqrt(self.square_norm(u))
        if norm ==0 :
            return u
        else :
            return u/norm
