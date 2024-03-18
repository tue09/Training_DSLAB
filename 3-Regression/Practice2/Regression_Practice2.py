import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def relative_error(y_true, y_predict):
    errors = np.abs(y_true - y_predict).astype(float) / y_true
    return np.mean(errors)*100

if __name__ == "__main__":
    df = pd.read_csv("elantra.csv")
    df = df.sort_values(by=['Year', 'Month'])
    df = df.reset_index(drop=True) #tham số drop=True là để loại bỏ cột index cũ
    
    '''plt.plot(df.ElantraSales.values)
    plt.xlabel('Time index')
    plt.ylabel('Sales')'''

    numeric_feats = df.columns.drop(["ElantraSales", "Month", "Year"])

    df_train = df[df.Year < 2013]
    df_test = df[df.Year >= 2013]

    y_train = df_train.ElantraSales.values
    y_test = df_test.ElantraSales.values

    #Chuẩn hóa data bằng StandardScaler, dữ liệu được chuẩn hóa theo dạng x -> (x-mean)/std
    #Nếu x có phân phối Gauss, dữ liệu chuẩn hóa sẽ thuộc phân phối N(0, 1)
    scaler = StandardScaler().fit(df_train[numeric_feats])

    X_train = scaler.transform(df_train[numeric_feats])
    X_test = scaler.transform(df_test[numeric_feats])

    Linear_Reg = linear_model.LinearRegression()
    Linear_Reg.fit(X_train, y_train)

    y_predict_test = Linear_Reg.predict(X_test)
    print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_predict_test))}')
    print(f'Mean relative errors:  {relative_error(y_test, y_predict_test)}')

    '''plt.plot(y_test)
    plt.plot(y_predict_test)

    plt.xlabel('Time index')
    plt.ylabel('Sales')'''

    month_onehot_train = pd.get_dummies(df_train.Month)

    X_train = np.hstack((X_train, month_onehot_train))
    X_test = np.hstack((X_test, pd.get_dummies(df_test.Month)))

    Linear_Reg.fit(X_train, y_train)
    y_predict_test = Linear_Reg.predict(X_test)
    print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_predict_test))}')
    print(f'Mean relative errors:  {relative_error(y_test, y_predict_test)}')

    plt.plot(y_test)
    plt.plot(y_predict_test)

    plt.xlabel('Time index')
    plt.ylabel('Sales')
    
    plt.show()
    