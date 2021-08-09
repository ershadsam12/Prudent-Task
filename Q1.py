#dependencies

import pandas as pd
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import accuracy_score
import  time

#EDA
df = pd.read_excel("E:\\my stuff\\Prudent Task\\default of credit card clients.xls", header = None)
df.columns
cols = pd.Series( df.iloc[1])
#considering the required columns and excluding others
df = df.loc[2:,:]
df.columns = cols
df.rename(columns = {'default payment next month': 'default.payment.next.month'}, inplace = True )
df.isna().sum()

df.info()

df=df.drop(["ID"],axis =1)
#Correlations
corr = df.corr()
print(corr)

X = df.drop(["default.payment.next.month"],axis =1 )
y = df["default.payment.next.month"]

#Train test split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size =0.2)


#Decision tree regressor
def DT_reg():
    rf = ExtraTreesRegressor()
    start_time = time.time()
    rf.fit(X,y)
    end_time = time.time()
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)
    print("MSE : ", mse)
    coef = pd.Series(rf.feature_importances_, X.columns).sort_values(ascending=False)
    coef.plot(kind='bar', title="Feature Importance" , color = 'red')
    plt.show()
    return start_time,end_time

#Random forest Regressor
def RF_reg():
    rf = RandomForestRegressor()
    start_time = time.time()
    rf.fit(X,y)
    end_time = time.time()
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)
    print("MSE : ", mse)
    coef = pd.Series(rf.feature_importances_, X.columns).sort_values(ascending=False)
    coef.plot(kind='bar', title="Feature Importance" , color = 'red')
    plt.show()
    return start_time,end_time

def model_training():
    time_diff = []
    start_time,end_time = DT_reg()
    time_diff.append((end_time-start_time))
    start_time,end_time = RF_reg()
    time_diff.append((end_time-start_time))
    return time_diff

def pp_model_training(model_list):
    pp_time_diff = []
    print ("model trainig called ")
    if('DTR' in model_list):
        print("building DTR")
        start_time,end_time = DT_reg()
        pp_time_diff.append((end_time-start_time))
        print("DT_reg execution done")
    elif('RFR' in model_list):
        print("building RFR")
        start_time, end_time = RF_reg()
        pp_time_diff.append((end_time-start_time))
        print('RFR execution done')
    else:    
        print("none")
        
    return pp_time_diff

if __name__ == '__main__':
    #log_regmodel()
    warnings.filterwarnings("ignore")
    time_diff = model_training()
    print("sequential model training completed with durations are ",time_diff)
    model_list = ['DTR', 'RFR']
    pool = multiprocessing.Pool(processes = multiprocessing.cpu_count()) 
    print('started pool')
    pp_time_diff = pool.map(pp_model_training,model_list )
    print("Parallel model training completed with durations are ", pp_time_diff )
    results_df = pd.DataFrame([time_diff,pp_time_diff], index = ['sequential training time','parallel training time'])
    results_df.columns = model_list
    print(results_df)   
    print('endpool')
    pool.close()
    