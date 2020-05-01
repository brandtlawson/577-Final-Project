

def do_math2(X_baseline, Y_baseline):
   
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    
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

    knn = KNeighborsClassifier(n_neighbors=8)
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
    print(y_train_baseline.shape)
    ada_b.fit(x_train_baseline, y_train_baseline)
    y_pred_ada_baseline = ada_b.predict(x_test_baseline)
    wrong = sum(abs(y_test_baseline-y_pred_ada_baseline))
    correct_ada_pred_baseline = 100*wrong/len(y_pred_ada_baseline)
    #error_list[i] = correct_pred_baseline
    print("The model performance for baseline AdaBoost model is:")
    print("---------------------------------------------")
    print('mean absoulte error is {}%'.format(round(correct_ada_pred_baseline,2)))
    print(accuracy_score(y_test_baseline, y_pred_ada_baseline))