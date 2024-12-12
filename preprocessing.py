
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype

def get_data(file_name):

    columns_name = ['status_current_account', 'duration', 'credit_history', 'purpose',
               'credit_amount', 'savings', 'employed_since', 'installment_rate',
               'status_and_sex', 'other_debtors', 'present_residence_since', 'property',
               'age', 'other_installment_plans', 'housing', 'n_credits', 'job',
               'n_maintenance_people', 'telephone', 'foreign', 'Class']
    df = pd.read_csv(file_name, sep=' ', names=columns_name)

    return df

def clean_data_german(df):
    dict_attr = {
        'A11': '< 0',
        'A12': '0 <= X < 200',
        'A13': '>= 200',
        'A14': 'no checking account',
        'A30': 'no credit taken / all paid other banks',
        'A31': 'paid back',
        'A32': 'current credit paid',
        'A33': 'delay',
        'A34': 'critical / other banks credit',
        'A40': 'new car',
        'A41': 'car used',
        'A42': 'furniture',
        'A43': 'radio/tv',
        'A44': 'domestic appliances',
        'A45': 'repairs',
        'A46': 'education',
        'A47': 'vacation',
        'A48': 'retraining',
        'A49': 'business',
        'A410': 'others',
        'A61': '< 100',
        'A62': '100 <= X < 500',
        'A63': '500 <= X < 1000',
        'A64': '>= 1000',
        'A65': 'unknown',
        'A71': 'unemployed',
        'A72': '< 1',
        'A73': '1 <= X < 4',
        'A74': '4 <= X < 7',
        'A75': '>= 7',
        'A91': 'male divorced/separated',
        'A92': 'female divorced/separated/married',
        'A93': 'male single',
        'A94': 'male married/widowed ',
        'A95': 'female single',
        'A101': 'none',
        'A102': 'co-applicant',
        'A103': 'guarantor',
        'A121': 'real estate',
        'A122': 'building society/life insurance',
        'A123': 'car or other',
        'A124': 'unknown / no property',
        'A141': 'bank',
        'A142': 'stores',
        'A143': 'none',
        'A151': 'rent',
        'A152': 'own',
        'A153': 'for free',
        'A171': 'unemployed or unskulled non resident',
        'A172': 'unskilled resident',
        'A173': 'skilled/official',
        'A174': 'management/self-employed/highly qualified/officer',
        'A191': 'none',
        'A192': 'yes',
        'A201': 'yes',
        'A202': 'no'
    }

    df = df.replace(dict_attr)

    return df

def clean_data_home(df):
    df[df.select_dtypes(include=np.number).columns] = df.select_dtypes(include=np.number).fillna(-1)
    df[df.select_dtypes(exclude=np.number).columns] = df.select_dtypes(exclude=np.number).fillna('nan')
    # df.dropna(axis=1, inplace = True)
    df.rename(columns={'TARGET': 'Class'}, inplace=True)

    return df

def clean_data(df, dataset_name='german'):

    if dataset_name == 'german':
        df = clean_data_german(df)
    elif dataset_name == 'home':
        df = clean_data_home(df)

    return df
    

def process_data_german(df):

    df = df.copy()
    
    df['Class'] = df['Class'].replace(2,0)

    # Create gender column (1 female 0 male)
    if 'status_and_sex' in df.columns:
        df.insert(len(df.columns)-1, 'gender', 
                       np.where(df['status_and_sex'] == 'female divorced/separated/married', 1, 0))
        # Remove status_and_sex column
        df = df.drop('status_and_sex', axis=1)

    # Select cols to encode
    cols_enc = list(df.select_dtypes([object]).columns)

    # Encoder creation
    ce_be = ce.OneHotEncoder(cols=cols_enc)

    # transform the data
    df_binary = ce_be.fit_transform(df)

    return df_binary

def process_data_home(df):

    df = df.copy()

    df.rename(columns={'TARGET': 'Class'}, inplace=True)

    # Create gender column (1 female 0 male)
    if 'CODE_GENDER' in df.columns:
        df = df[df['CODE_GENDER'] != 'XNA']

        df.insert(len(df.columns)-1, 'gender', 
                       np.where(df['CODE_GENDER'] == 'F', 1, 0))
        # Remove CODE_GENDER column
        df = df.drop('CODE_GENDER', axis=1)

    # Select cols to encode
    cols_enc = list(df.select_dtypes([object]).columns)

    # Encoder creation
    ce_be = ce.OneHotEncoder(cols=cols_enc)

    # transform the data
    df_binary = ce_be.fit_transform(df)

    # Drop nan
    df_binary = df_binary.dropna(axis=1)

    return df_binary

def process_data(df, dataset_name='german'):

    if dataset_name == 'german':
        df = process_data_german(df)
    elif dataset_name == 'home':
        df = process_data_home(df)

    return df

def split_data(df, test_size = 0.10, y_name = 'Class', get_test = True, seed = None):

    # Get attributes
    X = df.loc[:, df.columns != y_name]

    # Get class
    y = df[y_name]

    if get_test == True:
    # Stratified division
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, stratify = y, random_state=seed)

        return X_train, X_test, y_train, y_test

    else:
        return X, y


def get_df4chi(df, dataset='german'):

    if dataset == 'german':
        df4chi = df.copy()
        df4chi.loc[df4chi['status_and_sex'] != 'female divorced/separated/married', 'status_and_sex'] = 'male'
        df4chi.loc[df4chi['status_and_sex'] == 'female divorced/separated/married', 'status_and_sex'] = 'female'

        df4chi.loc[df4chi['Class'] == 1, 'Class'] = 'Acepted'
        df4chi.loc[df4chi['Class'] == 2, 'Class'] = 'Rejected'

    elif dataset == 'home':
        df4chi = df.copy()
        df4chi.loc[df4chi['CODE_GENDER'] == 'M', 'CODE_GENDER'] = 'male'
        df4chi.loc[df4chi['CODE_GENDER'] == 'F', 'CODE_GENDER'] = 'female'
        df4chi = df4chi[df4chi['CODE_GENDER'] != 'XNA']

        df4chi.loc[df4chi['Class'] == 0, 'Class'] = 'Acepted'
        df4chi.loc[df4chi['Class'] == 1, 'Class'] = 'Rejected'
    
    else:
        return None

    return df4chi

def get_res_df(X, y_true, y_pred):
    X = X.reset_index(drop=True)
    y_true = y_true.reset_index(drop=True)
    return pd.concat([X, y_true, pd.Series(y_pred, name='y_pred')], axis=1)

def decoding(df, df_processed, ignore_columns = []):
    joined_df = pd.concat([df, df_processed], axis=1)
    decoding_dict = {}
    
    for column_name in df_processed.columns:
        if column_name not in ignore_columns:
            dec_column_name = ''.join(i for i in column_name if not i.isdigit())[:-1]

            res = joined_df.loc[joined_df[column_name] == 1][dec_column_name].unique()
            decoding_dict[column_name] = res[0]
            
    return decoding_dict

def ignore_attribute_n_values(df, n = 10): # Move to script file
    
    selected_attributes = []
    
    for c in df.columns:
        if is_numeric_dtype(df[c]) or df[c].nunique() < n:
            selected_attributes.append(c)
    
    return df[selected_attributes]

def fill_nan(df):
    df[df.select_dtypes(include=np.number).columns] = df.select_dtypes(include=np.number).fillna(df.select_dtypes(include=np.number).median())
    df[df.select_dtypes(exclude=np.number).columns] = df.select_dtypes(exclude=np.number).fillna('nan_category')
    
    return df