from sklearn.feature_selection import RFE


def _rfe(*, x, y, estimator, n_features_to_select=None, step=1, verbose=0):
    """For more info visit :
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
    """

    selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step, verbose=verbose)
    selector.fit(x, y)

    rfe_support = selector.get_support()
    rfe_feature = x.loc[:, rfe_support].columns.tolist()
    rfe_ranking = selector.ranking_

    return rfe_feature, rfe_ranking
