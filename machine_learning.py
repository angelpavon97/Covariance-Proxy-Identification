import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn import tree, metrics
import matplotlib.pyplot as plt
import seaborn as sn
from preprocessing import split_data, get_res_df
import statsmodels.api as sm

def train_model(X_train, y_train):

    clf = LogisticRegressionCV(cv=10, max_iter=1000)
    clf = clf.fit(X_train,y_train)

    return clf

def test_model(clf, X_test):
    return clf.predict(X_test)

def print_results(y_test, y_pred_test, metric = 'accuracy'):

    # Get Confusion matrix
    m_confusion_test = metrics.confusion_matrix(y_test, y_pred_test)

    # Get Accuracy or F1 Score
    if metric == 'accuracy':
        acc_test = metrics.accuracy_score(y_test, y_pred_test)
        print('Test acc:', acc_test)
    elif metric == 'f1':
        acc_test = metrics.f1_score(y_test, y_pred_test)
        print('Test f1:', acc_test)

    print('Test confussion matrix:\n')
    
    sn.heatmap(pd.DataFrame(m_confusion_test, index = [1, 2], columns = [1, 2]), annot=True)
    plt.xlabel('Predicted label') 
    plt.ylabel('True label')
    plt.show()

def search_best_attributes(df, class_name = 'Class', current_attributes = [], best_acc = 0, return_data = False, metric = 'accuracy'):

    for i in range(len(df.columns)):
        for col_name in df.columns:
            if col_name not in current_attributes and col_name != class_name:
                attributes = current_attributes + [col_name]
                df2 = df.copy()

                # Split data
                X_train, X_test, y_train, y_test = split_data(df2, test_size = 0.20, y_name = class_name)
                X_train2 = X_train.copy()[attributes]
                X_test2 = X_test.copy()[attributes]
                
                # Train
                clf = tree.DecisionTreeClassifier()
                clf = clf.fit(X_train2, y_train)

                # Test
                y_pred = clf.predict(X_test2)

                if metric == 'accuracy':
                    accuracy = metrics.accuracy_score(y_test, y_pred)
                elif metric == 'f1':
                    accuracy = metrics.f1_score(y_test, y_pred)

                if accuracy > best_acc:
                    best_clf = clf
                    best_attributes = current_attributes + [col_name]
                    best_acc = accuracy
                    best_res_df = get_res_df(X_test, y_test, y_pred)
                if accuracy == 0:
                    best_clf = clf
                    best_attributes = current_attributes + [col_name]
                    best_acc = accuracy
                    best_res_df = get_res_df(X_test, y_test, y_pred)


        current_attributes = best_attributes
    
    if return_data == False:
        return best_clf, best_attributes, best_acc
    else:
        return best_clf, best_attributes, best_acc, best_res_df

def search_best_attributes_for_fairness(df, protected_attribute = 'gender', class_name = 'Class', current_attributes = [], best_acc = 0, metric = 'accuracy'):
    
    for i in range(len(df.columns)):
        for col_name in df.columns:
            if col_name not in current_attributes and col_name != class_name and col_name != protected_attribute:
                attributes = current_attributes + [col_name]
                df2 = df.copy()

                # Split data
                X_train, X_test, y_train, y_test = split_data(df2, test_size = 0.20, y_name = class_name)
                X_train2 = X_train.copy()[attributes]
                X_test2 = X_test.copy()[attributes]
                
                # Train
                clf = tree.DecisionTreeClassifier()
                clf = clf.fit(X_train2, y_train)

                # Test
                y_pred = clf.predict(X_test2)
                
                if metric == 'accuracy':
                    accuracy = metrics.accuracy_score(y_test, y_pred)
                elif metric == 'f1':
                    accuracy = metrics.f1_score(y_test, y_pred)

                if accuracy > best_acc:
                    best_clf = clf
                    best_attributes = current_attributes + [col_name]
                    best_acc = accuracy
                    best_res_df = get_res_df(X_test, y_test, y_pred)


        current_attributes = best_attributes

    return best_clf, best_attributes, best_acc, best_res_df



# Stepwise regression

def process_data_regression(df):

    df_t = df.copy()

    df_t[df_t.select_dtypes(['object']).columns] = df_t.select_dtypes(['object']).apply(lambda x: x.astype('category'))
    cat_columns = df_t.select_dtypes(['category']).columns
    df_t[cat_columns] = df_t[cat_columns].apply(lambda x: x.cat.codes)

    return df_t

def forward_regression(X, y,
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True,
                       process_data = True):

    if process_data == True:
        X = process_data_regression(X)
        y = y.astype('category')

    initial_list = []
    included = list(initial_list)

    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)

        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
            
        best_pval = new_pval.min()

        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  with p-value '.format(best_feature, best_pval))

        if not changed:
            break

    return included

def backward_regression(X, y,
                           initial_list=[], 
                           threshold_in=0.01, 
                           threshold_out = 0.05, 
                           verbose=True,
                           process_data = True):

    if process_data == True:
        X = process_data_regression(X)
        y = y.astype('category')

    included=list(X.columns)
    while True:
        changed=False
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop  with p-value '.format(worst_feature, worst_pval))
        if not changed:
            break
    return included