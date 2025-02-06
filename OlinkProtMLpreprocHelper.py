####################
# ML preprocessing #
####################
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

def label_encode(train_data, test_data, cols_to_enc):
    """
    Perform label encoding of train and test data
    :param train_data: training data
    :param test_data: test data
    :param cols_to_enc: array of columns that are categorical and should be label encoded
    :return: train and test data with categorical columns represented using label encoding
    """
    for curr_col in cols_to_enc:
        le = LabelEncoder()
        train_data[curr_col] = le.fit_transform(train_data[curr_col])
        test_data[curr_col] = le.transform(test_data[curr_col])
    return train_data, test_data


def ordinal_encode(train_data, test_data, cols_to_enc, **kwargs):
    """
    Perform ordinal encoding of train and test data
    :param train_data: training data
    :param test_data: test data
    :param cols_to_enc: dict where keys are names of columns that are categorical and should be ordinal encoded, and
    values are arrays representing the ordered categorical levels: e.g. {"Smoking Status": [Never, Previous, Current]}
    :param na_pol: string describing how to handle NAs
    :return: train and test data with categorical columns represented using ordinal encoding
    """
    # Create nested list defining the order of categorical levels for each categorical column
    categ_cols = list(cols_to_enc.keys())
    categ_orders = [curr_categ_order for curr_col, curr_categ_order in cols_to_enc.items()]

    # Perform encoding
    oe = OrdinalEncoder(categories=categ_orders, **kwargs)
    train_data[categ_cols] = oe.fit_transform(train_data[categ_cols])
    test_data[categ_cols] = oe.transform(test_data[categ_cols])
    return train_data, test_data


def standardise_continuous_vars(train_data, test_data, continuous_columns, hout_data=None):
    """
    Standardise continuous variables. Apply the parameters from the train set to the test and held-out sets
    :param train_data: train data (could be CV train fold)
    :param test_data: test data (could be CV test fold)
    :param continuous_columns: list of columns that contain continuous variables
    :param hout_data: held-out data
    :return: standardised train, standardised test, and standardised held-out (if provided) dataframes
    """
    sc = StandardScaler(with_mean=True, with_std=True)
    train_data[continuous_columns] = sc.fit_transform(train_data[continuous_columns])
    test_data[continuous_columns] = sc.transform(test_data[continuous_columns])
    if hout_data is not None:
        hout_data[continuous_columns] = sc.transform(hout_data[continuous_columns])
        return train_data, test_data, hout_data
    else:
        return train_data, test_data

