import os
import pandas as pd
import numpy as np
import pickle
from statistics import median,mean,stdev
from scipy import stats as s
import math
from warnings import simplefilter
simplefilter(action='ignore')

import collections
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from utils.save_load import *

def load_save_df_all(data_dir, outcome_dir, summary_type):
    """
    :param data_dir: physionet challenge 2012 raw data directory
                      './predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/'
    :param outcome_dir: directory to save result
    :param summary_type: aggregation method : 'sttdev', 'median', 'mean
    :return: pandas dataframe df_full, features_a, features_b, features_full, outcomes_a, outcomes_b, outcomes_full
    """

    def parsing(input_dir, summary_type):
        """
        parsing txt data to padnas dataframe
        :param input_dir: physionet challenge 2012 raw data directory
                          './predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/'
        :param summary_type: aggregation method : 'sttdev', 'median', 'mean
        :return: dataframe (features)
        """
        attr_list = [
            "RecordID", "Age", "Gender", "Height", "ICUType", "Weight",
            "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol",
            "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT",
            "HR", "K", "Lactate", "Mg", "MAP", "MechVent", "Na", "NIDiasABP",
            "NIMAP", "NISysABP", "PaCO2", "PaO2", "pH", "Platelets",
            "RespRate", "SaO2", "SysABP", "Temp", "TroponinI", "TroponinT",
            "Urine", "WBC"
        ]

        dir_path = input_dir
        patients_dir = {}
        c = 0
        mylist = []
        for root, dirs, files in sorted(os.walk(dir_path, topdown=False)):
            for name in sorted(files):
                # Checking the filename it is has txt extension it is taken up for processing
                if 'txt' in name:
                    mylist.append(name)
                    f = open(os.path.join(root, name), 'r')
                    rows = []
                    for row in f.readlines():
                        rows.append(row)
                    p1 = {}
                    # Adding the time of each measurement
                    p1["time"] = []
                    for var in attr_list:
                        p1[var] = []
                    for row in rows[1:]:
                        p1["time"].append(row.split(',')[0])
                        p1[row.split(',')[1]].append([row.split(',')[0], row.rstrip().split(',')[2]])
                    patients_dir[c] = p1
                    c += 1

        dup_dir = patients_dir.copy()
        # Iterate over the patients dictionary for summarizing each feature
        for key, value in dup_dir.items():
            # As each value gives the patient dictionary the iterating on the attributes of that patient
            for key_, val in value.items():
                # Ignoring the time when measurement is made
                if 'time' not in key_:
                    # Some features may not have any values replace it with NA
                    if isinstance(val, (list)) and len(val) == 0:
                        value[key_] = 'NA'
                    # If only one value for a feature is available then take that value
                    elif isinstance(val, (list)) and len(val) == 1:
                        templist = val
                        res_ = [el[1] for el in templist]
                        value[key_] = res_[0]
                    # When feature has many values, then different types of summarization can be done like mean, median,mode, stddev
                    elif isinstance(val, (list)) and len(val) > 1:
                        templist = val
                        res = [float(el[1]) for el in templist]
                        if 'stddev' in summary_type:
                            value[key_] = stdev(res)
                        elif 'mean' in summary_type:
                            value[key_] = sum(res) / len(res)
                        elif 'mode' in summary_type:
                            # If multiple modes then take the first mode
                            value[key_] = float(s.mode(res)[0])

        ## Create a dataframe then add each patient, where each feature is a summary statistic
        df = pd.DataFrame(columns=attr_list)
        for key, value in patients_dir.items():
            df = df.append({'RecordID': value['RecordID'],
                            'Age': value['Age'],
                            'Gender': value['Gender'],
                            'Height': value['Height'],
                            'ICUType': value['ICUType'],
                            'Weight': value['Weight'],
                            'Albumin': value['Albumin'],
                            'ALP': value['ALP'],
                            'ALT': value['ALT'],
                            'AST': value['AST'],
                            'Bilirubin': value['Bilirubin'],
                            'BUN': value['BUN'],
                            'Cholesterol': value['Cholesterol'],
                            'Creatinine': value['Creatinine'],
                            'DiasABP': value['DiasABP'],
                            'FiO2': value['FiO2'],
                            'GCS': value['GCS'],
                            'Glucose': value['Glucose'],
                            'HCO3': value['HCO3'],
                            'HCT': value['HCT'],
                            'HR': value['HR'],
                            'K': value['K'],
                            'Lactate': value['Lactate'],
                            'Mg': value['Mg'],
                            'MAP': value['MAP'],
                            'MechVent': value['MechVent'],
                            'Na': value['Na'],
                            'NIDiasABP': value['NIDiasABP'],
                            'NIMAP': value['NIMAP'],
                            'NISysABP': value['NISysABP'],
                            'PaCO2': value['PaCO2'],
                            'PaO2': value['PaO2'],
                            'pH': value['pH'],
                            'Platelets': value['Platelets'],
                            'RespRate': value['RespRate'],
                            'SaO2': value['SaO2'],
                            'SysABP': value['SysABP'],
                            'Temp': value['Temp'],
                            'TroponinI': value['TroponinI'],
                            'TroponinT': value['TroponinT'],
                            'Urine': value['Urine'],
                            'WBC': value['WBC']},
                           ignore_index=True)
        return df

    a_dir = data_dir + 'set-a'
    b_dir = data_dir + 'set-b'

    features_a = parsing(a_dir, summary_type)
    features_b = parsing(b_dir, summary_type)
    features_full = pd.concat([features_a, features_b], axis=0)

    outcomes_a = pd.read_csv(data_dir + 'Outcomes-a.txt')
    outcomes_b = pd.read_csv(data_dir + 'Outcomes-b.txt')
    outcomes_full = pd.concat([outcomes_a, outcomes_b], axis=0)

    for tmp in [features_a, features_b, features_full, outcomes_a, outcomes_b, outcomes_full]:
        tmp['RecordID'] = tmp['RecordID'].astype(np.int64)

    df_full = pd.merge(features_full, outcomes_full, how='left', on='RecordID')

    if outcome_dir == None:
        pass
    elif outcome_dir != None:
        df_full.to_pickle(outcome_dir + 'phy_2012_full.pkl')
    else:
        print("please set outcome dir to save full df")

    return df_full, features_a, features_b, features_full, outcomes_a, outcomes_b, outcomes_full


def clean_df(df):
    """
    replace wrong na . repalce -1 to np.nan , repalce 'NA' to np.nan
    :param df: pandas dataframe features_full from the function load_save_df_all(data_dir, outcome_dir, summary_type)
    :return: pandas dataframe clean_df
    """
    ## -1 to np.nan , repalce 'NA' to np.nan
    clean_df = df.sort_values('RecordID').copy()
    # Clean the Data
    clean_df['Height'] = clean_df['Height'].apply(lambda x: np.nan if x == '-1' else x)
    clean_df['Weight'] = clean_df['Weight'].apply(lambda x: np.nan if x == '-1' else x)
    clean_df['Gender'] = clean_df['Gender'].apply(lambda x: np.nan if x == '-1' else x)
    clean_df.replace('NA', np.nan, inplace=True)

    clean_df = clean_df.sort_values("RecordID")
    return clean_df


def impute_scale(df):
    """
    imputing and scaling data
    :param df: pandas dataframe clean_df from function clean_df(df)
    :return: pandas dataframe
    """
    num_cols = [e for e in list(df.columns) if e not in ('RecordID', 'Gender', 'MechVent', 'ICUType')]
    # Preprocessor - Imputation and Scaling

    numeric_transformer = Pipeline(steps=[
        ('num_imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('scaler', MinMaxScaler())])

    non_numeric_transformer = Pipeline(steps=[
        ('non_num_imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent'))])

    procedure_transformer = Pipeline(steps=[
        ('procedure_imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, num_cols),
            ('non_numeric', non_numeric_transformer, ['Gender', 'ICUType']),
            ('procedure', procedure_transformer, ['MechVent'])
        ]
    )

    df_tmp = df.drop(columns=['RecordID']).reset_index(drop=True)
    df_ID = df['RecordID'].reset_index(drop=True)
    df_processed = pd.concat(
        [df_ID,
        pd.DataFrame(data=preprocessor.fit_transform(df_tmp), columns=num_cols + ['Gender', 'ICUType'] + ['MechVent'])],
        axis=1
    )
    return df_processed


# ICU Types : 1~4
def split_client_by_icu(df, outcome_df, label_name='In-hospital_death'):
    """
    create OrderedDict of each client's dataset (icu client)
    split client data by icu type. and split features data to vertical,
    :param df: padnas dataframe from function impute_scale(df)
    :param outcome_df: outcomes dataframe outcome_full from function load_save_df_all
    :param label_name: target label name, 'In-hospital_death', 'Survival', 'Length_of_stay', 'SOFA', 'SAPS-I'
                        default: 'In-hospital_death'
    :return: Liwth consisted of OrderedDict include the information about clients train, valid dataset
    """
    print("=" * 10, "Split Client By ICU Type")
    df = df.astype(np.float32)

    lab_list = [
        'ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin', 'Cholesterol', 'Creatinine', 'FiO2',
        'Glucose', 'HCO3', 'HCT', 'K', 'Lactate', 'Mg', 'Na', 'PaCO2', 'PaO2', 'Platelets', 'SaO2',
        'TroponinI', 'TroponinT', 'Urine', 'WBC', 'pH'
    ]

    vital_list = [
        'DiasABP', 'GCS', 'HR', 'MAP', 'NIDiasABP', 'NIMAP', 'NISysABP', 'RespRate', 'SysABP', 'Temp',
    ]
    procedure_list = ['MechVent']
    demo_list = ['Age', 'Gender', 'Height', 'Weight']

    client_dataset = collections.OrderedDict()
    for icu in range(1, 5):
        client_name = "client_" + str(icu)
        X = df[df['ICUType'] == icu].sort_values('RecordID')
        y = outcome_df[outcome_df['RecordID'].isin(X['RecordID'])].sort_values('RecordID')[label_name]

        X = X.drop(columns=['RecordID', 'ICUType'])

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.4, stratify=y,
                                                            random_state=42)

        X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test,
                                                            test_size=0.5, stratify=y_test,
                                                            random_state=42)

        print(f"Adding {len(X_train)} train data / {len(X_valid)} valid data for client {len(X_valid)} / test data for client : {client_name}")
        data = collections.OrderedDict((('input_train', X_train.values), ('label_train', y_train),
                                        ('input_valid', X_valid.values), ('label_valid', y_valid),
                                        ('input_test', X_test.values), ('label_test', y_test),
                                        ('input_demo_train', X_train[demo_list].values),
                                        ('input_demo_valid', X_valid[demo_list].values),
                                        ('input_demo_test', X_test[demo_list].values),
                                        ('input_lab_train', X_train[lab_list].values),
                                        ('input_lab_valid', X_valid[lab_list].values),
                                        ('input_lab_test', X_test[lab_list].values),
                                        ('input_vital_train', X_train[vital_list].values),
                                        ('input_vital_valid', X_valid[vital_list].values),
                                        ('input_vital_test', X_test[vital_list].values),
                                        ('input_procedure_train', X_train[procedure_list].values),
                                        ('input_procedure_valid', X_valid[procedure_list].values),
                                        ('input_procedure_test', X_test[procedure_list].values)
                                        ))
        client_dataset[client_name] = data
    return client_dataset


def create_client_scenario_dataset(client_dataset, scenario, split_type, vertical=False):
    """
    create OrderedDict of each client's dataset with scenario (vertical vs common / full)
    :param client_dataset: OrderedDict from function split_client_by_icu
    :param scenario: 'full', 'demo+procedure'
    :param split_type: 'train', 'valid', 'test'
    :param vertical: if 'True' then return vertical feature , else return common feature / default: False
    :return: OrderedDict
    """

    # create common feature dataset
    if vertical == False:
        client_common_dataset = collections.OrderedDict()
        if scenario == 'full':
            print("=" * 10, f"Create Clients' full {split_type} dataset")
            for client in client_dataset:
                data = collections.OrderedDict((
                    (f'input_{split_type}', client_dataset[client][f'input_{split_type}']),
                    ('label', client_dataset[client][f'label_{split_type}'])
                ))
                client_common_dataset[client] = data
                print(
                    f"{client}: input_{split_type}_shape: {client_common_dataset[client][f'input_{split_type}'].shape} label_shape: {client_common_dataset[client]['label'].shape}")

        elif scenario == 'demo+procedure':
            print("="*10, f"Create Clients' common {split_type} dataset")
            for client in client_dataset:
                data = collections.OrderedDict((
                    (f'input_{split_type}', np.concatenate([client_dataset[client][f'input_demo_{split_type}'],
                                                        client_dataset[client][f'input_procedure_{split_type}']],
                                                       axis=1)),
                    ('label', client_dataset[client][f'label_{split_type}'])
                ))
                client_common_dataset[client] = data
                print(
                    f"{client}: input_{split_type}_shape: {client_common_dataset[client][f'input_{split_type}'].shape} label_shape: {client_common_dataset[client]['label'].shape}")

        elif scenario == 'demo+lab':
            print("=" * 10, f"Create Clients' common {split_type} dataset")
            for client in client_dataset:
                data = collections.OrderedDict((
                    (f'input_{split_type}', np.concatenate([client_dataset[client][f'input_demo_{split_type}'],
                                                        client_dataset[client][f'input_lab_{split_type}']],
                                                       axis=1)),
                    ('label', client_dataset[client][f'label_{split_type}'])
                ))
                client_common_dataset[client] = data
                print(
                    f"{client}: input_{split_type}_shape: {client_common_dataset[client][f'input_{split_type}'].shape} label_shape: {client_common_dataset[client]['label'].shape}")
        elif scenario == 'demo':
            print("=" * 10, f"Create Clients' vertical {split_type} dataset")
            client_vertical_dataset = collections.OrderedDict()
            for client in client_dataset:
                data = collections.OrderedDict((
                    (f'input_{split_type}', client_dataset[client][f'input_demo_{split_type}']),
                    ('label', client_dataset[client][f'label_{split_type}'])
                ))
                client_vertical_dataset[client] = data
                print(
                    f"{client}: input_{split_type}_shape: {client_vertical_dataset[client][f'input_{split_type}'].shape} label_shape: {client_vertical_dataset[client]['label'].shape}")
        return client_common_dataset




    # create vertical feature dataset
    else:
        if scenario == 'demo+procedure':
            print("="*10, f"Create Clients' vertical {split_type} dataset")
            client_vertical_dataset = collections.OrderedDict()
            for client in client_dataset:
                data = collections.OrderedDict((
                    (f'input_{split_type}', np.concatenate([client_dataset[client][f'input_lab_{split_type}'],
                                                        client_dataset[client][f'input_vital_{split_type}']],
                                                       axis=1)),
                    ('label', client_dataset[client][f'label_{split_type}'])
                ))
                client_vertical_dataset[client] = data
                print(
                    f"{client}: input_{split_type}_shape: {client_vertical_dataset[client][f'input_{split_type}'].shape} label_shape: {client_vertical_dataset[client]['label'].shape}")
        elif scenario == 'demo':
            print("="*10, f"Create Clients' vertical {split_type} dataset")
            client_vertical_dataset = collections.OrderedDict()
            for client in client_dataset:
                data = collections.OrderedDict((
                    (f'input_{split_type}', np.concatenate([client_dataset[client][f'input_lab_{split_type}'],
                                                        client_dataset[client][f'input_vital_{split_type}']],
                                                       axis=1)),
                    ('label', client_dataset[client][f'label_{split_type}'])
                ))
                client_vertical_dataset[client] = data
                print(
                    f"{client}: input_{split_type}_shape: {client_vertical_dataset[client][f'input_{split_type}'].shape} label_shape: {client_vertical_dataset[client]['label'].shape}")
        elif scenario == 'demo+lab':
            print("="*10, f"Create Clients' vertical {split_type} dataset")
            client_vertical_dataset = collections.OrderedDict()
            for client in client_dataset:
                data = collections.OrderedDict((
                    (f'input_{split_type}', client_dataset[client][f'input_vital_{split_type}']),
                    ('label', client_dataset[client][f'label_{split_type}'])
                ))
                client_vertical_dataset[client] = data
                print(
                    f"{client}: input_{split_type}_shape: {client_vertical_dataset[client][f'input_{split_type}'].shape} label_shape: {client_vertical_dataset[client]['label'].shape}")

        return client_vertical_dataset

def create_external_scenario_dataset(df, outcome_df, label_name='In-hospital_death'):
    """
    create OrderedDict of each client's dataset (icu client)
    split client data by icu type. and split features data to vertical,
    :param df: padnas dataframe from function impute_scale(df)
    :param outcome_df: outcomes dataframe outcome_full from function load_save_df_all
    :param label_name: target label name, 'In-hospital_death', 'Survival', 'Length_of_stay', 'SOFA', 'SAPS-I'
                        default: 'In-hospital_death'
    :return: Liwth consisted of OrderedDict include the information about clients train, valid dataset
    """
    df = df.astype(np.float32)

    lab_list = [
        'ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin', 'Cholesterol', 'Creatinine', 'FiO2',
        'Glucose', 'HCO3', 'HCT', 'K', 'Lactate', 'Mg', 'Na', 'PaCO2', 'PaO2', 'Platelets', 'SaO2',
        'TroponinI', 'TroponinT', 'Urine', 'WBC', 'pH'
    ]

    vital_list = [
        'DiasABP', 'GCS', 'HR', 'MAP', 'NIDiasABP', 'NIMAP', 'NISysABP', 'RespRate', 'SysABP', 'Temp',
    ]
    procedure_list = ['MechVent']
    demo_list = ['Age', 'Gender', 'Height', 'Weight']

    X = df.sort_values('RecordID').drop(columns=['RecordID', 'ICUType'])
    y = outcome_df.sort_values('RecordID')[label_name]

    data = collections.OrderedDict((('input_full', X.values), ('label', y),
                                    ('input_common', np.concatenate([X[demo_list].values, X[procedure_list].values], axis=1)),
                                    ('input_vertical', np.concatenate([X[lab_list].values, X[vital_list].values], axis=1))
                                    ))
    print("=" * 10, f"Create External Clients dataset")
    print(f"input_full_shape: {data['input_full'].shape}, input_common_shape: {data['input_common'].shape}, input_vertical_shape: {data['input_vertical'].shape} \n "
          f"label_shape: {data['label'].shape}")
    return data

# def create_federated_data(client_train_dataset, num_epochs=200, batch_size=64, prefetch_buffer=10):
#     """
#     create federated data (PrefetchDataset for TFF)
#     :param client_train_dataset: OrderedDict from function created_client_traind_data
#     :param num_epochs: local epochs
#     :param batch_size: local batch size
#     :param prefetch_buffer: buffersize for prefetch
#     :return:
#     """
#     train_dataset = tff.simulation.FromTensorSlicesClientData(client_train_dataset)
#
#     def preprocess(dataset, shuffle_buffer):
#         # shuffle_buffer = len(train_dataset.create_tf_dataset_for_client(x))
#         def batch_format_fn(element):
#             return collections.OrderedDict(
#                 x=element['X_train'],
#                 y=element['label']
#             )
#
#         return dataset.repeat(num_epochs).shuffle(shuffle_buffer).batch(
#             batch_size).map(batch_format_fn).prefetch(prefetch_buffer)
#
#     def make_federated_data(client_data, client_ids):
#         return [preprocess(client_data.create_tf_dataset_for_client(x),
#                            len(client_data.create_tf_dataset_for_client(x))
#                            ) for x in client_ids]
#
#     federated_train_data = make_federated_data(train_dataset, train_dataset.client_ids)
#     return federated_train_data

if __name__ == "__main__":
    os.chdir('/Users/taehyun/PycharmProjects/FederatedLearningBase/')
    data_dir = '/Users/taehyun/PycharmProjects/FederatedLearningBase/processed_data/challenge-2012/'
    save_dir = '/Users/taehyun/PycharmProjects/FederatedLearningBase/processed_data/physionet2012/'
    df_full, features_a, features_b, features_full, outcomes_a, outcomes_b, outcomes_full = load_save_df_all(data_dir,
                                                                                                             outcome_dir=None,
                                                                                                             summary_type='mean')
    features_internal, features_external, outcomes_internal, outcomes_external = train_test_split(features_full,
                                                                                                  outcomes_full,
                                                                                                  test_size=0.25,
                                                                                                  stratify=
                                                                                                  outcomes_full[
                                                                                                      'In-hospital_death']
                                                                                                  )
    # internal
    internal_clean_features = clean_df(features_a)
    internal_df_processed = impute_scale(internal_clean_features)

    # internal client
    client_dataset = split_client_by_icu(internal_df_processed, outcomes_full, label_name="In-hospital_death")

    client_full_train = create_client_scenario_dataset(client_dataset, scenario="full", split_type='train',
                                                       vertical=False)

    client_common_train = create_client_scenario_dataset(client_dataset, scenario="demo+procedure", split_type='train',
                                                         vertical=False)
    client_vertical_train = create_client_scenario_dataset(client_dataset, scenario="demo+procedure",
                                                           split_type='train', vertical=True)

    client_full_valid = create_client_scenario_dataset(client_dataset, scenario="full", split_type='valid',
                                                       vertical=False)
    client_common_valid = create_client_scenario_dataset(client_dataset, scenario="demo+procedure", split_type='valid',
                                                         vertical=False)
    client_vertical_valid = create_client_scenario_dataset(client_dataset, scenario="demo+procedure",
                                                           split_type='valid', vertical=True)

    client_full_test = create_client_scenario_dataset(client_dataset, scenario="full", split_type='test',
                                                      vertical=False)
    client_common_test = create_client_scenario_dataset(client_dataset, scenario="demo+procedure", split_type='test',
                                                        vertical=False)
    client_vertical_test = create_client_scenario_dataset(client_dataset, scenario="demo+procedure", split_type='test',
                                                          vertical=True)
    # internal save
    OrderedDict_to_pkl(client_full_train, 'icu_client_full_train.pkl', save_dir=save_dir)
    OrderedDict_to_pkl(client_common_train, 'icu_client_common_train.pkl', save_dir=save_dir)
    OrderedDict_to_pkl(client_vertical_train, 'icu_client_vertical_train.pkl', save_dir=save_dir)

    OrderedDict_to_pkl(client_full_valid, 'icu_client_full_valid.pkl', save_dir=save_dir)
    OrderedDict_to_pkl(client_common_valid, 'icu_client_common_valid.pkl', save_dir=save_dir)
    OrderedDict_to_pkl(client_vertical_valid, 'icu_client_vertical_valid.pkl', save_dir=save_dir)

    OrderedDict_to_pkl(client_full_test, 'icu_client_full_test.pkl', save_dir=save_dir)
    OrderedDict_to_pkl(client_common_test, 'icu_client_common_test.pkl', save_dir=save_dir)
    OrderedDict_to_pkl(client_vertical_test, 'icu_client_vertical_test.pkl', save_dir=save_dir)

    # external
    external_clean_features = clean_df(features_external)
    # external dataset
    external_df_processed = impute_scale(external_clean_features)
    external_validation_data = create_external_scenario_dataset(external_df_processed, outcomes_external,
                                                                label_name='In-hospital_death')
    # external save
    OrderedDict_to_pkl(external_validation_data, 'icu_external_data.pkl', save_dir=save_dir)