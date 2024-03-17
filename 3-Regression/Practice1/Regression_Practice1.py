import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

if __name__ == "__main__":
    diabetes = datasets.load_diabetes()
    print(f"so chieu du lieu input: {diabetes.data.shape}")
    print(f"kieu du lieu input: {type(diabetes.data)}")
    print(f"so chieu du lieu output: {diabetes.target.shape}")
    print(f"kieu du lieu output: {type(diabetes.target)}")

    #Chia data thành 2 phần train và test
    diabetes_X = diabetes.data
    diabetes_Y = diabetes.target

    diabetes_X_train = diabetes_X[:361]
    diabetes_Y_train = diabetes_Y[:361]

    diabetes_X_test = diabetes_X[362:]
    diabetes_Y_test = diabetes_Y[362:]


    #xây dựng model sử dụng sklearn
    regr = linear_model.LinearRegression()

    regr_ridge = linear_model.Ridge(alpha=0.1)

    regr.fit(diabetes_X_train, diabetes_Y_train)
    print(f"[w1, ..., wn] =  {regr.coef_}")
    print(f"w0 = {regr.intercept_}")

    regr_ridge.fit(diabetes_X_train, diabetes_Y_train)
    print(f"[w1, ..., wn] =  {regr_ridge.coef_}")
    print(f"w0 = {regr_ridge.intercept_}")

    #Giá trị đúng
    print("Gia tri true: ", diabetes_Y_test[0])

    #Dự đoán cho mô hình Linear Regression sử dụng hàm dự đoán của thư viện
    y_pred_linear = regr.predict(diabetes_X_test[0:1])
    print(f"Gia tri du doan cho mô hình linear regression: {y_pred_linear}")

    #Viết code tính và in kết quả dự đoán cho mô hình Linear Regression sử dụng công thức tại đây
    y_pred_linear_0 = sum(regr.coef_*diabetes_X_test[0])+regr.intercept_
    print(f"Gia tri du doan cho mô hình linear regression theo công thức: {y_pred_linear_0}")

    #Dự đoán cho mô hình Ridge Regression sử dụng hàm dự đoán của thư viện
    y_pred_ridge = regr_ridge.predict(diabetes_X_test[0:1])
    print(f"Gia tri du doan cho mô hình ridge regression: {y_pred_ridge}")

    #Viết code tính và in kết quả dự đoán cho mô hình Ridge Regression sử dụng công thức tại đây
    y_pred_ridge_0 = sum(regr_ridge.coef_*diabetes_X_test[0])+regr_ridge.intercept_
    print(f"Gia tri du doan cho mô hình ridge regression theo công thức: {y_pred_ridge_0}")

    diabetes_Y_predict = regr.predict(diabetes_X_test)
    result = pd.DataFrame(data=np.array([diabetes_Y_test, diabetes_Y_predict,
                                abs(diabetes_Y_test - diabetes_Y_predict)]).T,
                                columns=["Thuc te", "Du doan", "Lech"])
    print(result)

    print(math.sqrt(mean_squared_error(diabetes_Y_test, diabetes_Y_predict)))

    _lambda = [0, 0.0001, 0.01, 0.04, 0.05, 0.06, 0.1, 0.5, 1, 5, 10, 20]

    for a_lambda in _lambda:
        ridge_regression = linear_model.Ridge(alpha = a_lambda, max_iter=1000, tol=1e-4)
        ridge_regression.fit(diabetes_X_train, diabetes_Y_train)
        diabetes_Y_predict_Ridge = ridge_regression.predict(diabetes_X_test)
        print(f"Lambda = {a_lambda}; RMSE = {math.sqrt(mean_squared_error(diabetes_Y_test, diabetes_Y_predict_Ridge))}")

    #Vẽ biểu đồ bằng seaborn
    sns.distplot(diabetes_Y_predict)
    plt.show()