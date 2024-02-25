import os
import pandas as pd
import numpy as np
import copy
import random
from datetime import datetime

random.seed(1)

class Ridge_Regression:
    def __init__(self) -> None:
        pass

    def normalize_and_add_one(self, data):
        res = []
        X_min = np.min(data)
        X_max = np.max(data)
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = (data[i][j] - X_min) / (X_max - X_min) #normalize
            res.append(np.insert(data[i], 0, 1)) #add one
        res = np.array(res)
        return res

    def fit(self, X_train, Y_train, LAMBDA):
        assert len(X_train.shape) == 2 and X_train.shape[0] == Y_train.shape[0]
        w = np.linalg.inv(X_train.transpose().dot(X_train) + LAMBDA*np.identity(X_train.shape[1])).dot(X_train.transpose()).dot(Y_train)
        return w
    
    def fit_Gradient(self, X_train, Y_train, LAMBDA, learning_rate, max_num_epoch, batch_size):
        W = np.random.rand(X_train.shape[1])
        lastLoss = 1e10
        num_mini_batch= int(np.ceil(X_train.shape[0] / batch_size))

        for epoch in range(max_num_epoch):
            orders = np.array(range(X_train.shape[0]))
            np.random.shuffle(orders)
            X_train, Y_train = X_train[orders], Y_train[orders]
            for i in range(num_mini_batch):
                index = i*batch_size
                X_train_sub = X_train[index:index+batch_size]
                Y_train_sub = Y_train[index:index+batch_size]
                gradient = X_train_sub.transpose().dot(X_train_sub.dot(W) - Y_train_sub) + LAMBDA*W
                W = W - learning_rate*gradient
            newLoss = self.computeRss(Y_train, self.predict(X_train, W))
            if np.abs(newLoss - lastLoss) < 1e-10:
                break
            lastLoss = newLoss
        return W

    def get_the_best_LAMBDA(self, X_train, Y_train, num_fold):
        def cross_validation(LAMBDA):
            row_ids = np.array(range(X_train.shape[0]))
            size = int(np.ceil(X_train.shape[0] / num_fold))
            valid_ids = []
            for i in range(num_fold - 1):
                valid_ids.append(row_ids[i*size:(i+1)*size])
            valid_ids.append(row_ids[(num_fold - 1)*size:])
            avg_RSS = 0
            for i in range(num_fold):
                valid_part = valid_ids[i]
                train_part = np.array([])
                for j in range(num_fold):
                    if j == i: continue
                    else:
                        train_part = np.concatenate((train_part, valid_ids[j]))
                train_part = train_part.astype(int)
                W = self.fit(X_train[train_part], Y_train[train_part], LAMBDA)
                Y_Predict = self.predict(X_train[valid_part], W)
                avg_RSS += self.computeRss(Y_train[valid_part], Y_Predict)    
            return avg_RSS / num_fold

        def range_scan(best_LAMBDA, minimum_RSS, LAMBDA_Values):
            for current_LAMBDA in LAMBDA_Values:
                avg_RSS = cross_validation(current_LAMBDA)
                if avg_RSS < minimum_RSS:
                    minimum_RSS = avg_RSS
                    best_LAMBDA = current_LAMBDA
            return best_LAMBDA, minimum_RSS
        
        #0, 1, 2, ..., 49
        best_LAMBDA, minimum_RSS = range_scan(0, 1e10, range(X_train.shape[0]))

        LAMBDA_values = [k/1000 for k in range(max(0, (best_LAMBDA - 1)*1000), (best_LAMBDA + 1)*1000, 1)]

        best_LAMBDA, minimum_RSS = range_scan(best_LAMBDA, minimum_RSS, LAMBDA_values)

        return best_LAMBDA
    

    def predict(self, X_New, W):
        return X_New.dot(W)
    
    def computeRss(self, Y_New, Y_Predict):
        loss = (1 / Y_New.shape[0]) * np.sum((Y_New - Y_Predict) ** 2)
        return loss


def read_data(data_path):
    #readlines
    with open(data_path, 'r') as f:
        lines = f.readlines()[72:]
    #convert to float number
    data = []
    for line in lines:
        temp = [float(item) for item in line.split()]
        data.append(temp[1:])
    data = np.array(data)
    return data

if __name__=="__main__":
    current_directory = os.getcwd()
    print(f'Current Directory: {current_directory}')
    data = read_data("Session_1/Regression/Data.txt")

    RR = Ridge_Regression()
    data = RR.normalize_and_add_one(data)
    size_train = 50
    X_train, X_test = data[:size_train, :-1], data[size_train:, :-1]
    Y_train, Y_test = data[:size_train, -1], data[size_train:, -1]
    
    LAMBDA = RR.get_the_best_LAMBDA(X_train, Y_train, num_fold=5)

    #W = RR.fit(X_train, Y_train, LAMBDA=0.8)
    W = RR.fit_Gradient(X_train, Y_train, LAMBDA=LAMBDA, learning_rate=0.001, max_num_epoch=500, batch_size=128)
    Y_Predict = RR.predict(X_test, W)
    Loss = RR.computeRss(Y_test, Y_Predict)
    print(f'LAMBDA: {LAMBDA}')
    print(f'RSS: {Loss}')

    with open("Session_1/Regression/Result.txt", "w") as f:
        f.write(f'{datetime.now()}\n')
        f.write(f'Best LAMBDA: {LAMBDA}\n')
        f.write(f'RSS: {Loss}')
    print("Done !")

