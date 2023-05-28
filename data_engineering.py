from datetime import datetime

import pandas as pd
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def handle_missing_values(v_train_df, v_strategy):
    """
    Handle the missing value of the dataframe applying the selected strategy

    Inputs:
    - v_train_df: a dataframe containing the dataset
    - v_strategy: selected strategy

    Returns:
    - v_train_df: the original dataset with now the missing values handled
    """
    imputer = SimpleImputer(strategy=v_strategy)
    v_train_df = pd.DataFrame(imputer.fit_transform(v_train_df), columns=v_train_df.columns)
    return v_train_df


def encode_categorical_variables(v_train_df):
    """
    Encode the categorical variables

    Inputs:
    - v_train_df: a dataframe containing the dataset

    Returns:
    - v_train_df: the original dataset with now the missing values handled
    """
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    cat_cols = ['job', 'marital', 'education', 'device', 'outcome_old']
    encoded = pd.DataFrame(encoder.fit_transform(v_train_df[cat_cols]), columns=encoder.get_feature_names_out(cat_cols))
    v_train_df = pd.concat([v_train_df, encoded], axis=1)
    v_train_df.drop(cat_cols, axis=1, inplace=True)
    return v_train_df


def scale_numerical_features(v_train_df):
    """
    Scale the numerical features

    Inputs:
    - v_train_df: a dataframe containing the dataset

    Returns:
    - v_train_df: the original dataset with now the missing values handled
    """
    scaler = StandardScaler()
    num_cols = ['age', 'day', 'month', 'time_spent', 'banner_views', 'banner_views_old', 'days_elapsed_old', 'X4']
    scaled = pd.DataFrame(scaler.fit_transform(v_train_df[num_cols]), columns=[f'scaled_{col}' for col in num_cols])
    v_train_df = pd.concat([v_train_df, scaled], axis=1)
    v_train_df.drop(num_cols, axis=1, inplace=True)
    return v_train_df


predictor_models = dict()


def predict_na_method(
        dataframe,
        column,
        is_train,
        categorical_columns,
        model_class=None,
        model_params=None,
):
    if model_params is None:
        model_params = {}
    if model_class is None:
        model_class = CatBoostClassifier
        model_params['verbose'] = False
    columns_to_drop = categorical_columns + [column]
    if 'subscription' in dataframe.columns:
        columns_to_drop.append('subscription')
    X = dataframe.drop(columns=columns_to_drop)
    y = dataframe[column]
    na_index = y == 'na'
    X_train = X[~na_index]
    y_train = y[~na_index]
    if is_train:
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        predictor_models[(column, model_class.__name__)] = model
    else:
        model = predictor_models[(column, model_class.__name__)]
    pred = model.predict(X[na_index])
    return pred, na_index


def predict_na_for_col_list(
        dataframe,
        column_list,
        is_train,
        categorical_columns,
        model_class=None,
        model_params=None
):
    result = dataframe
    for col in column_list:
        result = predict_na_method(
            result,
            col,
            is_train,
            categorical_columns,
            model_class,
            model_params
        )
    return result


def encode_categorical_features(
        data,
        categorical_columns,
        predict_na,
        is_train,
        drop_columns=True
):
    for col in tqdm(categorical_columns):
        if predict_na and 'na' in data[col].unique():
            pred, na_index = predict_na_method(
                dataframe=data,
                column=col,
                is_train=is_train,
                categorical_columns=categorical_columns
            )
            data[f'{col}_is_na'] = data[col] == 'na'
            data.loc[na_index, col] = pred

        new_cols = pd.get_dummies(data[col])
        if 'na' in new_cols.columns:
            new_cols = new_cols.drop(columns=['na'])
        else:
            new_cols = new_cols[new_cols.columns[:-1]]
        columns = [f'{col}_{column}' for column in new_cols.columns]
        data[columns] = new_cols
    if drop_columns:
        data.drop(columns=categorical_columns, inplace=True)


def transform_data(
        data,
        is_train,
        categorical_columns,
        predict_na=False,
        add_weekday=True,
        transform_days_elapsed_old=True,
        extra_data=None
):
    dataframe = data.copy(deep=True)
    initial_length = len(data)
    if extra_data is not None:
        dataframe = pd.concat(
            [dataframe, extra_data],
            axis=0
        ).reset_index(drop=True)
    if 'Id' in dataframe.columns:
        dataframe.drop(columns=['Id'], inplace=True)
    if add_weekday:
        if 'weekday_2021' not in categorical_columns:
            categorical_columns = list(categorical_columns)
            categorical_columns.append('weekday_2021')
        dataframe['weekday_2021'] = dataframe[['month', 'day']].apply(
            lambda row: datetime(2021, row['month'], row['day']).weekday(),
            axis=1)

    encode_categorical_features(
        dataframe,
        categorical_columns,
        predict_na=predict_na,
        is_train=is_train,
    )
    if transform_days_elapsed_old:
        dataframe['never_saw_before'] = dataframe['days_elapsed_old'] == -1
        dataframe.loc[
            dataframe['days_elapsed_old'] == -1,
            'days_elapsed_old'
        ] = 365 * 3
    return dataframe.iloc[:initial_length, :]


def evaluate_model(
        model,
        X_train_eval,
        X_test_eval,
        y_train_eval,
        y_test_eval
):
    predict_test = model.predict(X_test_eval)
    predict_train = model.predict(X_train_eval)

    accuracy_train = round(accuracy_score(y_train_eval, predict_train), 3)
    accuracy_test = round(accuracy_score(y_test_eval, predict_test), 3)
    print(f'{accuracy_train=}\n{accuracy_test=}')

    cm = confusion_matrix(y_test_eval, predict_test, labels=model.classes_)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_
    )
    disp.plot(cmap='magma')
    plt.show()
