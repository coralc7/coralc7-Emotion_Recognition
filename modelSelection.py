import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from preparation import *

class ModelSelection:
    path = Preparation.path

    def __init__(self):
        self.Xtrain = load_pkls_file(Preprocessing.path, "Xtrain_resampled")
        self.ytrain = load_pkls_file(Preprocessing.path, "ytrain_resampled")
        self.Xtest = load_pkls_file(Preparation.path, "Xtest_reduced")
        self.ytest = load_pkls_file(Preparation.path, "Xtest_static_with_labels")["label"]

    def get_cv_iter(self, random_state=2):
        y = self.ytrain.replace(Preprocessing.dict_convert_labels_2_numeric).to_list()
        entities_list = Preprocessing().get_entities(movie_id_list=list(self.Xtrain.index))
        groups = []
        for e in entities_list:
            #e = entities_list[0]
            groups.append(int(e[1:]))
        groups = pd.Series(groups).to_list()
        # split train and test
        cv_iter = StratifiedGroupKFold(n_splits=10, random_state=random_state, shuffle=True).split(X=self.Xtrain, y=y, groups=groups)
        return cv_iter, groups

    def hyperparameter_tuning(self, params, model, cv, groups):
        model_search = GridSearchCV(param_grid=params, estimator=model, scoring="f1_micro", cv=cv)
        model_search.fit(X=m.Xtrain, y=m.ytrain, groups=groups)
        return model_search.best_params_, model_search.best_score_

m = ModelSelection()
cv_iter, groups = m.get_cv_iter(random_state=2)
rf_param = {'max_depth' : list(range(170, 190, 2)),
             'n_estimators': list(range(20, 30, 1)),
             'max_features': ['sqrt', 'auto', 'log2'],
             'min_samples_split': list(range(10, 20, 1)),
             'min_samples_leaf': list(range(12, 16, 1)),
             'bootstrap': [True, False],
             'criterion':['gini', 'entropy']
              }
RF = RandomForestClassifier(class_weight='balanced')
start = time.time()
model_best_params, model_best_score = m.hyperparameter_tuning(rf_param, RF, cv_iter, groups)
end = time.time()
print("total time is: {} sec".format(end-start))

"""
rf_param = {'max_depth' : list(range(100, 200, 10)),
             'n_estimators': list(range(5, 30, 5)),
             'max_features': ['sqrt','auto','log2'],
             'min_samples_split': list(range(5, 30, 5)),
             'min_samples_leaf': list(range(2, 20, 2)),
             'bootstrap': [True, False],
             'criterion':['gini','entropy']
              } # 18478 sec -> 308 -> min -> 5.2 hours

    def hyperparameter_tuning(self, model):
        # xgboost with default hyperparameters for binary classification
        from sklearn.datasets import make_classification
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import RepeatedStratifiedKFold
        from xgboost import XGBClassifier
        # define dataset
        X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
        # define model
        model = XGBClassifier()
        # define evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate model
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        # report result
        print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
"""
