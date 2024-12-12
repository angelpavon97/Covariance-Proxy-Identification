import pandas as pd

# function to obtain Categorical Features
def _get_categorical_features(df):
    feats = [col for col in list(df.columns) if df[col].dtype == 'object']
    return feats

# function to factorize categorical features
def _factorize_categoricals(df, cats):
    for col in cats:
        df[col], _ = pd.factorize(df[col])
    return df 

# function to create dummy variables of categorical features
def _get_dummies(df, cats):
    for col in cats:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
    return df 

# FEATURE ENGINEERING FUNCTIONS

def fe_previous_application(data, previous_application):
    ## count the number of previous applications for a given ID
    prev_apps_count = previous_application[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    previous_application['SK_ID_PREV'] = previous_application['SK_ID_CURR'].map(prev_apps_count['SK_ID_PREV'])

    ## Average values for all other features in previous applications
    prev_apps_avg = previous_application.groupby('SK_ID_CURR').mean()
    prev_apps_avg.columns = ['p_' + col for col in prev_apps_avg.columns]
    data = data.merge(right=prev_apps_avg.reset_index(), how='left', on='SK_ID_CURR')

    return data

def fe_bureau(data, bureau):
    # Average Values for all bureau features 
    bureau_avg = bureau.groupby('SK_ID_CURR').mean()
    bureau_avg['buro_count'] = bureau[['SK_ID_BUREAU','SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']
    bureau_avg.columns = ['b_' + f_ for f_ in bureau_avg.columns]
    data = data.merge(right=bureau_avg.reset_index(), how='left', on='SK_ID_CURR')

    return data

def fe_installments_payments(data, installments_payments):
    ## count the number of previous installments
    cnt_inst = installments_payments[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    installments_payments['SK_ID_PREV'] = installments_payments['SK_ID_CURR'].map(cnt_inst['SK_ID_PREV'])

    ## Average values for all other variables in installments payments
    avg_inst = installments_payments.groupby('SK_ID_CURR').mean()
    avg_inst.columns = ['i_' + f_ for f_ in avg_inst.columns]
    data = data.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')

    return data

def fe_pcb(data, pcb):
    ### count the number of pos cash for a given ID
    pcb_count = pcb[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    pcb['SK_ID_PREV'] = pcb['SK_ID_CURR'].map(pcb_count['SK_ID_PREV'])

    ## Average Values for all other variables in pos cash
    pcb_avg = pcb.groupby('SK_ID_CURR').mean()
    data = data.merge(right=pcb_avg.reset_index(), how='left', on='SK_ID_CURR')

    return data

def fe_credit_card_balance(data, credit_card_balance):
    ### count the number of previous applications for a given ID
    nb_prevs = credit_card_balance[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    credit_card_balance['SK_ID_PREV'] = credit_card_balance['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

    ### average of all other columns 
    avg_cc_bal = credit_card_balance.groupby('SK_ID_CURR').mean()
    avg_cc_bal.columns = ['cc_bal_' + f_ for f_ in avg_cc_bal.columns]
    data = data.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')

    return data

# MAIN
def get_kaggle_processed_data(remove_id=True):
    path = '/Users/app245/Documents/Datasets/HomeCredit/'

    app_train = pd.read_csv(path + 'application_train.csv')
    bureau = pd.read_csv(path + "bureau.csv")
    bureau_balance = pd.read_csv(path + "bureau_balance.csv")
    credit_card_balance = pd.read_csv(path + "credit_card_balance.csv")
    pcb = pd.read_csv(path + "POS_CASH_balance.csv")
    previous_application = pd.read_csv(path + "previous_application.csv")
    installments_payments = pd.read_csv(path + "installments_payments.csv")

    # get categorical features
    Y = app_train['TARGET']
    train_X = app_train.drop(['TARGET'], axis = 1)

    data = train_X.copy()
    data_cats = _get_categorical_features(data)
    prev_app_cats = _get_categorical_features(previous_application)
    bureau_cats = _get_categorical_features(bureau)
    pcb_cats = _get_categorical_features(pcb)
    ccbal_cats = _get_categorical_features(credit_card_balance)

    # create additional dummy features - 
    previous_application = _get_dummies(previous_application, prev_app_cats)
    bureau = _get_dummies(bureau, bureau_cats)
    pcb = _get_dummies(pcb, pcb_cats)
    credit_card_balance = _get_dummies(credit_card_balance, ccbal_cats)

    # factorize the categorical features from train and test data
    data = _factorize_categoricals(data, data_cats)

    # FEATURE ENGINEERING
    data = fe_previous_application(data, previous_application)
    data = fe_bureau(data, bureau)
    data = fe_installments_payments(data, installments_payments)
    data = fe_pcb(data, pcb)
    data = fe_credit_card_balance(data, credit_card_balance)

    # Remove ID
    if remove_id == True:
        data = data.drop('SK_ID_CURR', axis=1)

    #Own processing for fairness
    data = data.rename(columns={'CODE_GENDER': 'gender'})
    data['Class'] = Y

    return data

