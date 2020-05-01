from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

def make_my_predictions(df_90s,best_params_rf, best_params_gb, best_params_knn, best_params_ada):
    X_baseline = df_90s.iloc[:,3:-1]
    Y_baseline = df_90s.iloc[:,-1]
    x_train_baseline, x_test_baseline, y_train_baseline, y_test_baseline = train_test_split(X_baseline, Y_baseline, test_size = 0.2, random_state=5)
    num_iterations = 1
    error_list=[0]*num_iterations
    average_baseline_error=0
    for i in range(num_iterations): ##Random Forest is inherantly random, so taking an average error across several iterations may help
        rf = RandomForestClassifier(**best_params_rf)
        rf.fit(x_train_baseline, y_train_baseline)
        y_pred_baseline = rf.predict(x_test_baseline)
        wrong = sum(abs(y_test_baseline-y_pred_baseline))
        correct_pred_baseline = 100*wrong/len(y_pred_baseline)
        error_list[i] = correct_pred_baseline
    #print(error_list)
    average_baseline_error = sum(error_list)/num_iterations

    print("The model performance for baseline Random Forrest model is:")
    print("---------------------------------------------")
    print('mean absoulte error is {}%'.format(round(average_baseline_error,2)),'\n')

    gb = GradientBoostingClassifier(**best_params_gb)
    error_list=[0]*100
    average_baseline_error=0

    gb.fit(x_train_baseline, y_train_baseline)
    y_pred_gb_baseline = gb.predict(x_test_baseline)
    wrong = sum(abs(y_test_baseline-y_pred_gb_baseline))
    correct_pred_baseline = 100*wrong/len(y_pred_gb_baseline)
    #error_list[i] = correct_pred_baseline
    print("The model performance for baseline Gradient Boosting Classifier model is:")
    print("---------------------------------------------")
    print('mean absoulte error is {}%'.format(round(correct_pred_baseline,2)),'\n')

    knn = KNeighborsClassifier(**best_params_knn)
    error_list=[0]*100
    average_baseline_error=0
    
    knn.fit(x_train_baseline, y_train_baseline)
    y_pred_knn_baseline = knn.predict(x_test_baseline)
    wrong = sum(abs(y_test_baseline-y_pred_knn_baseline))
    correct_pred_baseline = 100*wrong/len(y_pred_knn_baseline)
    #error_list[i] = correct_pred_baseline
    print("The model performance for baseline KNN model is:")
    print("---------------------------------------------")
    print('mean absoulte error is {}%'.format(round(correct_pred_baseline,2)),'\n')

    #lc = LogisticClassifier()
    #lc.fit(x_train_baseline, y_train_baseline)
    #y_pred_lc_baseline = lc.predict(x_test_baseline)
    #wrong = sum(abs(y_test_baseline-y_pred_lc_baseline))
    #correct_pred_baseline = 100*wrong/len(y_pred_lc_baseline)
    #error_list[i] = correct_pred_baseline
    #print("The model performance for baseline Logistic Regression model is:")
    #print("---------------------------------------------")
    #print('mean absoulte error is {}%'.format(round(correct_pred_baseline,2)))


    ada_b = AdaBoostClassifier(**best_params_ada)
    ada_b.fit(x_train_baseline, y_train_baseline)
    y_pred_ada_baseline = ada_b.predict(x_test_baseline)
    wrong = sum(abs(y_test_baseline-y_pred_ada_baseline))
    correct_ada_pred_baseline = 100*wrong/len(y_pred_ada_baseline)
    #error_list[i] = correct_pred_baseline
    print("The model performance for baseline AdaBoost model is:")
    print("---------------------------------------------")
    print('mean absoulte error is {}%'.format(round(correct_ada_pred_baseline,2)))
    
    
from itertools import combinations
import seaborn as sns
def engineer_my_features(df_90s):
    combi = combinations([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 2)
    df_features = df_90s.iloc[:,-1].copy().to_frame()

    for i in list(combi): 
        df_features.insert(0, "interaction of " + str(df_90s.columns[i[0]] ) + " * " + str(df_90s.columns[i[1]] ) , df_90s.iloc[:,i[0]] * df_90s.iloc[:,i[1]])
    
    return df_features

def insert_top_features(df_features, df_90s):
    X_features = df_features.iloc[:,:-1]
    Y_features = df_features.iloc[:,-1]
    x_train_features, x_test_features, y_train_features, y_test_features = train_test_split(X_features, Y_features, test_size = 0.2, random_state=5)

    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=400), threshold='6*median')
    embeded_rf_selector.fit(x_train_features, y_train_features)
    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = x_train_features.loc[:,embeded_rf_support].columns.tolist()
    #print(str(len(embeded_rf_feature)), 'selected features')
    #print("Selected Feature:",embeded_rf_feature)
    for features in embeded_rf_feature:
        df_90s.insert(df_90s.shape[1]-1, features , X_features[features])
    return df_90s

from sklearn.model_selection import RandomizedSearchCV

# Use the random grid to search for best hyperparameters

def my_hyperparam_func(x_train,y_train):
# Create the random grid
    random_grid = {'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
               'min_samples_split': [1, 2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True, False]}


    rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=3, random_state=42, n_jobs = -1)
# Fit the random search model
    rf_random.fit(x_train,y_train)
    
    return rf_random.best_params_

def my_hyperparam_func2(x_train,y_train):
    gb = GradientBoostingClassifier()
    random_grid = {'learning_rate': [.008,.01,10],
                  'subsample'    : [.009,.01,.02,.1,.5,1,10,100],
                  'n_estimators' : [1,10,100, 200],
                  'max_depth'    : [0,1,2,5,10,15,50,100]
                 }
#random_grid = {
    
                #'learning_rate': list(np.linspace(10, 100, 1, dtype = float)/1000),
                #'n_estimators': list(np.linspace(10, 5000, 10, dtype = int))
                #}
    gb_random = RandomizedSearchCV(estimator = gb, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
    gb_random.fit(x_train,y_train)

    return gb_random.best_params_

def my_hyperparam_func3(x_train,y_train):

    knn = KNeighborsClassifier()
    random_grid = {'n_neighbors': [2,3,4,5,8,10,15,30,50],
                  'weights'    : ['uniform','distance'],
                   'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'p' :[1,2,3]
                 }
    knn_random = RandomizedSearchCV(estimator = knn, param_distributions = random_grid, n_iter = 30, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
    knn_random.fit(x_train,y_train)

    return knn_random.best_params_

def my_hyperparam_func4(x_train, y_train):
    
    ada = AdaBoostClassifier()
    random_grid = {'n_estimators': [2,3,4,5,8,10,15,30,50,300,500],
                   'learning_rate' : [.01,.1,.2,.3,.4,1,10,50]
                 }
    ada_random = RandomizedSearchCV(estimator = ada, param_distributions = random_grid, n_iter = 30, cv = 10, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
    ada_random.fit(x_train,y_train)

    return ada_random.best_params_

def drop_unselected_features(df_90s,embeded_gb_feature):
    print(embeded_gb_feature)
    for column in df_90s.columns:
        if column not in embeded_gb_feature:
            #print(column)
            df_90s = df_90s.drop(column, axis=1)  
    return df_90s

def make_my_prediction(df_90s):
    X_baseline = df_90s.iloc[:,3:-1]
    Y_baseline = df_90s.iloc[:,-1]
    x_train_baseline, x_test_baseline, y_train_baseline, y_test_baseline = train_test_split(X_baseline, Y_baseline, test_size = 0.2, random_state=5)
    num_iterations = 1
    error_list=[0]*num_iterations
    average_baseline_error=0
    for i in range(num_iterations): ##Random Forest is inherantly random, so taking an average error across several iterations may help
        rf = RandomForestClassifier()
        rf.fit(x_train_baseline, y_train_baseline)
        y_pred_baseline = rf.predict(x_test_baseline)
        wrong = sum(abs(y_test_baseline-y_pred_baseline))
        correct_pred_baseline = 100*wrong/len(y_pred_baseline)
        error_list[i] = correct_pred_baseline
    #print(error_list)
    average_baseline_error = sum(error_list)/num_iterations

    print("The model performance for baseline Random Forrest model is:")
    print("---------------------------------------------")
    print('mean absoulte error is {}%'.format(round(average_baseline_error,2)),'\n')

    gb = GradientBoostingClassifier()
    error_list=[0]*100
    average_baseline_error=0

    gb.fit(x_train_baseline, y_train_baseline)
    y_pred_gb_baseline = gb.predict(x_test_baseline)
    wrong = sum(abs(y_test_baseline-y_pred_gb_baseline))
    correct_pred_baseline = 100*wrong/len(y_pred_gb_baseline)
    #error_list[i] = correct_pred_baseline
    print("The model performance for baseline Gradient Boosting Classifier model is:")
    print("---------------------------------------------")
    print('mean absoulte error is {}%'.format(round(correct_pred_baseline,2)),'\n')

    knn = KNeighborsClassifier()
    error_list=[0]*100
    average_baseline_error=0
    
    knn.fit(x_train_baseline, y_train_baseline)
    y_pred_knn_baseline = knn.predict(x_test_baseline)
    wrong = sum(abs(y_test_baseline-y_pred_knn_baseline))
    correct_pred_baseline = 100*wrong/len(y_pred_knn_baseline)
    #error_list[i] = correct_pred_baseline
    print("The model performance for baseline KNN model is:")
    print("---------------------------------------------")
    print('mean absoulte error is {}%'.format(round(correct_pred_baseline,2)),'\n')

    #lc = LogisticClassifier()
    #lc.fit(x_train_baseline, y_train_baseline)
    #y_pred_lc_baseline = lc.predict(x_test_baseline)
    #wrong = sum(abs(y_test_baseline-y_pred_lc_baseline))
    #correct_pred_baseline = 100*wrong/len(y_pred_lc_baseline)
    #error_list[i] = correct_pred_baseline
    #print("The model performance for baseline Logistic Regression model is:")
    #print("---------------------------------------------")
    #print('mean absoulte error is {}%'.format(round(correct_pred_baseline,2)))


    ada_b = AdaBoostClassifier()
    ada_b.fit(x_train_baseline, y_train_baseline)
    y_pred_ada_baseline = ada_b.predict(x_test_baseline)
    wrong = sum(abs(y_test_baseline-y_pred_ada_baseline))
    correct_ada_pred_baseline = 100*wrong/len(y_pred_ada_baseline)
    #error_list[i] = correct_pred_baseline
    print("The model performance for baseline AdaBoost model is:")
    print("---------------------------------------------")
    print('mean absoulte error is {}%'.format(round(correct_ada_pred_baseline,2)))