import os
import re
import pandas as pd
import seaborn as sns
import numpy as np
import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV

from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV

from sklearn.svm import SVR

from sklearn.linear_model import SGDRegressor

import random

#データの行数の取得
def data_col_check(df):
    return len(df)
   

#データの列名をリストとして返す
def data_row_list(df):
    row_list = df.columns
    return row_list

#目的変数が含まれているか確認
def target_check(target,func):
    return target in func



#欠損値を埋める
#def nan_processing(nan,df):
    #return df.filina(0)


#データの行数によって分析手法を選択
def choice_Analysis_method(df,target,X,Y,col_data,row_list):

    if col_data  < 40 :
        return "データ数が少ないです"
    
    if len(set(row_list)) < 3:
        return "目的変数が二値のデータはまだ対応していません"
    
    #分析手法を格納するリストの作成
    Analysis_method = list()

    #データ数によって分析手法を選択

    if col_data < 500:
        Analysis_method.append(lasso_reg(df,target,X,Y))
        Analysis_method.append(ElasticNet(df,target,X,Y))
        Analysis_method.append(ridge_reg(df,target,X,Y))
        return Analysis_method

    else:
        Analysis_method.append(SGD_reg(df,target,X,Y))
        return Analysis_method

#lasso回帰
def lasso_reg(df,target,X,Y):
    scaler = StandardScaler()

    #クロスバリデーション
    clf = LassoCV(alphas=10 ** np.arange(-6, -1, 0.1), cv=5)
    scaler.fit(X)
    clf.fit(scaler.transform(X), Y)
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    scaler.fit(x_train)
    clf.fit(scaler.transform(x_train), y_train)
    y_pred = clf.predict(scaler.transform(x_test))
    mse = mean_squared_error(y_test, y_pred)
    
    return {"mse":mse, "coef":clf.coef_,"intersept":clf.intercept_}


#リッジ回帰
def ridge_reg(df,target,X,Y):
    scaler = StandardScaler()
    #クロスバリデーション
    clf = RidgeCV(alphas=10 ** np.arange(-6, -1, 0.1), cv=5)
    scaler.fit(X)
    clf.fit(scaler.transform(X), Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    y_pred = clf.predict(scaler.transform(x_test))
    mse = mean_squared_error(y_test, y_pred)

    return {"mse":mse, "coef":clf.coef_,"intersept":clf.intercept_}



#ElasticNet
def ElasticNet(df,target,X,Y):
    scaler = StandardScaler()
    #クロスバリデーション
    clf =  ElasticNetCV(alphas=10 ** np.arange(-6, -1, 0.1), cv=5)
    scaler.fit(X)
    clf.fit(scaler.transform(X), Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    scaler.fit(x_train)
    clf.fit(scaler.transform(x_train), y_train)
    y_pred = clf.predict(scaler.transform(x_test))  
    mse = mean_squared_error(y_test, y_pred)
     
    return {"mse":mse, "coef":clf.coef_,"intersept":clf.intercept_}


#SGDregression
def SGD_reg(df,target,X,Y):
    scaler = StandardScaler()

    clf = SGDRegressor(max_iter=1000, tol=1e-3)
    clf.fit(X, Y)
    SGDRegressor()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    scaler.fit(x_train)
    clf.fit(scaler.transform(x_train), y_train)
    y_pred = clf.predict(scaler.transform(x_test))
    mse = mean_squared_error(y_test, y_pred)
    return {"mse":mse, "coef":clf.coef_,"intersept":clf.intercept_}

#メイン関数
def search(file_path,target,nan = "0"):
    #データファイルの取得
    try:
         df = pd.read_csv(file_path,index_col = 0)
    except:
        return "ファイルのpathが間違っています"
    
    #目的変数が含まれているか確認
    if not target_check:
        return  "指定された目的変数はありません"

    
    #欠損値を埋める
    #df = nan_processing(nan)

    #データの説明変数の表示
    row_list  = data_row_list(df)
    row_list = row_list[1:]


    #データの行数の取得   
    col_data =  data_col_check(df)

    #目的変数と説明変数の分割
    Y = df[target]
    X = df.drop(target, axis = 1)
    
    result = choice_Analysis_method(df,target,X,Y,col_data,row_list)
    print(result)


if __name__ == "__main__":
    while True:
        file_path = input("file path : ")
        target = input("target : ")
        msg = search(file_path, target)
        if msg:
            print(msg)
        else:
            break

    
    








 
    



