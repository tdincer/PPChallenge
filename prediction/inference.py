import os
import joblib
import pandas as pd
import utility as ut
import warnings
warnings.filterwarnings("ignore")


def submit(y_pred, est_name):
    data = pd.read_csv('../data/inference.csv')
    data['Price ($/MWh)'] = y_pred
    data.to_csv(est_name + '_submission.csv', index=False)


def main():
    x_train, y_train, x_test, y_test, NUMS, CATS = ut.get_train_test_data()

    lgbparams = ut.get_best_params('lgb')
    xgbparams = ut.get_best_params('xgb')

    transformer_lgb = ut.get_preprocessor('lgb', NUMS, CATS)
    transformer_xgb = ut.get_preprocessor('xgb', NUMS, CATS)

    clf_lgb = ut.MeanRegressor(transformer_lgb, 'lgb', lgbparams)
    clf_xgb = ut.MeanRegressor(transformer_xgb, 'xgb', xgbparams)

    clf_lgb.fit(x_train, y_train)
    clf_xgb.fit(x_train, y_train)

    # Save models
    for clf in [clf_lgb, clf_xgb]:
        file_name = os.path.join('models', clf.est_name + '.model')
        joblib.dump(clf, file_name, compress=1)

    # Save Test Results
    y_test_lgb = clf_lgb.predict(x_test)
    y_test_xgb = clf_xgb.predict(x_test)

    submit(y_test_lgb, 'lgb')
    submit(y_test_xgb, 'xgb')


if __name__ == '__main__':
    main()
