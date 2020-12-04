import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from datascientist.feature_selection.recursive_feature_elimination import _rfe


def tests_rfe():
    player_df = pd.read_csv("datascientist/feature_selection/test/CSV/data.csv")

    # Taking only those columns which have numerical or categorical values since feature selection can be performed on
    # numerical data.

    numcols = ['Overall', 'Crossing', 'Finishing', 'ShortPassing', 'Dribbling', 'LongPassing', 'BallControl',
               'Acceleration', 'SprintSpeed', 'Agility', 'Stamina', 'Volleys', 'FKAccuracy', 'Reactions', 'Balance',
               'ShotPower', 'Strength', 'LongShots', 'Aggression', 'Interceptions']
    catcols = ['Preferred Foot', 'Position', 'Body Type', 'Nationality', 'Weak Foot']

    player_df = player_df[numcols + catcols]

    # encoding categorical values with one-hot encoding.

    traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])], axis=1)
    features = traindf.columns

    traindf = traindf.dropna()

    traindf = pd.DataFrame(traindf, columns=features)

    # Separating features(X) and target(y).
    y = traindf['Overall']
    X = traindf.copy()
    X = X.drop(['Overall'], axis=1)

    estimator = LinearRegression()
    mod = _rfe(x=X, y=y, estimator=estimator, n_features_to_select=5)
    assert mod[0] == ['Body Type_Akinfenwa', 'Body Type_Lean', 'Body Type_Normal', 'Body Type_Shaqiri', 'Body '
                                                                                                        'Type_Stocky']

    estimator = LinearRegression()
    mod = _rfe(x=X, y=y, estimator=estimator, n_features_to_select=7)
    assert mod[0] == ['Body Type_Akinfenwa', 'Body Type_Courtois', 'Body Type_Lean', 'Body Type_Normal', 'Body '
                      'Type_PLAYER_BODY_TYPE_25',
                      'Body Type_Shaqiri', 'Body Type_Stocky']

    estimator = Lasso()
    mod = _rfe(x=X, y=y, estimator=estimator, n_features_to_select=3)
    assert mod[0] == ['ShortPassing', 'Reactions', 'Strength']
