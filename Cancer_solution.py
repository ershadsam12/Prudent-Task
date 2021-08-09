
#import dependencies
#data cleaning and manipulation
import numpy as np
import pandas as pd
import multiprocessing
#data visualization
import matplotlib.pyplot as mt
import seaborn as sns

import warnings


import statsmodels.formula.api as sm
import statsmodels.api as sma

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# machine learning
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

import time
    

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)


def log_regmodel():
    breast_cancer = pd.read_csv("E:\\my stuff\\Prudent Task\\archive\\data.csv")
    breast_cancer.columns
    
    breast_cancer.info()
    # Generate and visualize the correlation matrix
    corr = breast_cancer.corr().round(2)
    corr
    
    # first, dropiing all "worst" columns
    cols = ['radius_worst', 
            'texture_worst', 
            'perimeter_worst', 
            'area_worst', 
            'smoothness_worst', 
            'compactness_worst', 
            'concavity_worst',
            'concave_points_worst', 
            'symmetry_worst', 
            'fractal_dimension_worst']
    breast_cancer = breast_cancer.drop(cols, axis=1)
    
    # then, drop all columns related to the "perimeter" and "area" attributes
    cols = ['perimeter_mean',
            'perimeter_se', 
            'area_mean', 
            'area_se']
    breast_cancer = breast_cancer.drop(cols, axis=1)
    
    # lastly, drop all columns related to the "concavity" and "concave points" attributes
    cols = ['concavity_mean',
            'concavity_se', 
            'concave_points_mean', 
            'concave_points_se']
    breast_cancer = breast_cancer.drop(cols, axis=1)
    
    # verify remaining columns
    breast_cancer.columns
    
    # Draw the heatmap again, with the new correlation matrix
    corr = breast_cancer.corr().round(2)
    corr

    en = LabelEncoder()
    x = breast_cancer.iloc[:,1:]
    x['diagnosis'] = en.fit_transform(x['diagnosis'])
    y = breast_cancer["diagnosis"]
    y = en.fit_transform(y)
    
    #splitting the data set
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=40)
    type(x)
    
    
    formula = 'diagnosis ~ radius_mean + texture_mean + smoothness_mean + compactness_mean + symmetry_mean + fractal_dimension_mean + radius_se + texture_se + smoothness_se + compactness_se + symmetry_se + fractal_dimension_se'
    type(formula)
    
    start_time = time.time()
    #building model
    logit_model = sm.glm( formula = formula, data = X_train, family = sma.families.Binomial())
    mod_fit = logit_model.fit()
    mod_fit.summary()
    end_time = time.time()
    #prediction on test data
    pred = mod_fit.predict(X_test.iloc[ :, 1:])
    
    # from sklearn import metrics
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    optimal_threshold
    import pylab as pl
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
    roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    # Plot tpr vs 1-fpr
    fig, ax = pl.subplots()
    pl.plot(roc['tpr'], color = 'red')
    pl.plot(roc['1-fpr'], color = 'blue')
    pl.xlabel('1-False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic')
    ax.set_xticklabels([])
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)
    X_test["predicted"] = np.zeros(len(y_test))
    X_test.loc[pred > optimal_threshold , "predicted"] = 1
    X_test.loc[pred < optimal_threshold , "predicted"] = 0
    # confusion matrix 
    confusion_matrix = pd.crosstab(X_test['diagnosis'], X_test['predicted'])
    confusion_matrix
    # classification report
    classification_test = classification_report(X_test['diagnosis'], X_test['predicted'])
    classification_test
    return start_time,end_time


def buildKNN():
    
    breast_cancer = pd.read_csv("E:\\my stuff\\Prudent Task\\archive\\data.csv")
    breast_cancer.columns
    
    en = LabelEncoder()
    x = breast_cancer.iloc[:,1:]
    x['diagnosis'] = en.fit_transform(x['diagnosis'])
    y = breast_cancer["diagnosis"]
    y = en.fit_transform(y)
    
    #splitting the data set
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=40)
    
    from sklearn.neighbors import KNeighborsClassifier
    
    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors = 2)
    knn.fit(X_train, y_train)
    end_time = time.time()
    pred = knn.predict(X_test)
    pred
    
    # Evaluate the model
    
    print(accuracy_score(y_test, pred))
    pd.crosstab(y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 
    return start_time,end_time


def build_classifier(optimzer):
    classifier = Sequential()
    classifier.add(Dense(units = 30, kernel_initializer = 'glorot_uniform', activation = 'relu'))
    classifier.add(Dense(16, kernel_initializer = 'glorot_uniform', activation = 'relu' ))
    classifier.add(Dense(8, kernel_initializer = 'glorot_uniform', activation = 'relu' ))
    classifier.add(Dense(1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid' ))
    classifier.compile(optimizer = optimzer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
    


def nn_model():
    breast_cancer = pd.read_csv("E:\\my stuff\\360digiTMG\\M0duI3$\\Deep Learning Challenges\\Azzignments\\breast_cancer.csv")
    
    X = breast_cancer.iloc[:,2:32]
    Y = breast_cancer.iloc[:,1:2]
    
    X.iloc[:,:] = norm_func(X.iloc[:, :])
    en = LabelEncoder()
    # malignant as 1 and benign as 0 
    Y = pd.DataFrame(  en.fit_transform(Y) )
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    

    classifier = KerasClassifier(build_fn = build_classifier)
    parameters = {'batch_size': [10,20, 30],
                  'nb_epoch': [100,150, 200],
                  'optimzer': ['adam', 'rmsprop']}
    grid_search = GridSearchCV(estimator = classifier, param_grid = parameters,scoring = 'accuracy', cv = 5)
    grid_search = grid_search.fit(X_train, y_train, verbose =0)
    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    start_time = time.time()
    #creating sequential model
    #the input are considered as 30 as data has 30 columns, activation is relu
    #hidden layers were being added and by reducing the neurons
    #output layer has single neuron and activation function is sigmoid and loss is binary cross entropy
    cancer_model = Sequential()
    cancer_model.add(Dense(units = 30, kernel_initializer = 'glorot_uniform', activation = 'relu'))
    cancer_model.add(Dense(16, kernel_initializer = 'glorot_uniform', activation = 'relu'))
    cancer_model.add(Dense(units = 8 , kernel_initializer = 'glorot_uniform', activation = 'relu'))
    cancer_model.add(Dense(units = 1 , kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
    cancer_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    cancer_model.fit(X_train, y_train, batch_size = 10, epochs = 200, verbose = 0)
    end_time = time.time()
    
    y_pred = cancer_model.predict(X_test)
    y_pred = (y_pred>0.5)
    
    cancer_matrix = confusion_matrix(y_test,y_pred)
    print("confusion matrix \n")
    print(cancer_matrix)
    
    accuracy = ((cancer_matrix.diagonal()[0]+ cancer_matrix.diagonal()[1])/(cancer_matrix.sum()))
    print(accuracy*100)
    
    #on train data
    train_y_pred = cancer_model.predict(X_train)
    train_y_pred = (train_y_pred>0.5)
    
    train_cancer_matrix = confusion_matrix(y_train,train_y_pred)
    print(train_cancer_matrix)
    train_accuracy = ((train_cancer_matrix.diagonal()[0]+ train_cancer_matrix.diagonal()[1])/(train_cancer_matrix.sum()))
    #train accuracy
    print(train_accuracy*100)
    return start_time,end_time

def model_training():
    time_diff = []
    start_time,end_time = buildKNN()
    time_diff.append((end_time-start_time))
    start_time,end_time = nn_model()
    time_diff.append((end_time-start_time))
    start_time,end_time = log_regmodel()
    time_diff.append((end_time-start_time))
    return time_diff

def pp_model_training(model_list):
    pp_time_diff = []
    print ("model trainig called ")
    if('knn' in model_list):
        print("building KNN")
        start_time,end_time = buildKNN()
        pp_time_diff.append((end_time-start_time))
        print("knn execution done")
    elif('nn' in model_list):
        print("building NN")
        start_time, end_time = nn_model()
        pp_time_diff.append((end_time-start_time))
        print('nn execution done')
    elif('log' in model_list):
        print("building logistic regression model")
        start_time,end_time = log_regmodel()
        pp_time_diff.append((end_time-start_time))
    else:    
        print("none")
        
    return pp_time_diff

if __name__ == '__main__':
    #log_regmodel()
    warnings.filterwarnings("ignore")
    time_diff = model_training()
    print("sequential model training completed with time diff are ",time_diff)
    model_list = ['knn','nn', 'log']
    pool = multiprocessing.Pool(processes = multiprocessing.cpu_count()) 
    print('started pool')
    pp_time_diff = pool.map(pp_model_training,model_list )
    print("Parallel model training completed with time diff are ", pp_time_diff )
    results_df = pd.DataFrame([time_diff,pp_time_diff], index = ['sequential training time','parallel training time'])
    results_df.columns = model_list
    print(results_df)   
    print('endpool')
    
    