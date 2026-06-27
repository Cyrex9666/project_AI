import pandas as pd
from sklearn.preprocessing import StandardScaler

def split_features_and_target(dataset):
    X = dataset.drop(columns=["target"])
    y = dataset["target"].astype(int)

    return X, y


# time_train_test_split
#   - used to split data into training and testing 
def time_train_test_split(X, y, train_size):
    split_index = int(len(X) * train_size)

    # X_train takes index 0 to split_index for training data, eg the first 80% of data
    # X_test takes from split_index to the rest of data for testing purposes, eg the last 20%
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


# scale to reasonable size
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

# Final preparation of model-ready data
def prepare_model_data(dataset, train_size):
    X, y = split_features_and_target(dataset)

    X_train, X_test, y_train, y_test = time_train_test_split(
        X=X,
        y=y,
        train_size=train_size
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train=X_train,
        X_test=X_test
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler