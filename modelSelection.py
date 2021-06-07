from sklearn.model_selection import GridSearchCV
from preparation import *

class ModelSelection:
    path = Preparation.path

    def __init__(self, method):
        self.method = method
        self.Xtrain = load_pkls_file(Preprocessing.path, "Xtrain_{}_resampled".format(method))
        self.ytrain = load_pkls_file(Preprocessing.path, "ytrain_{}_resampled".format(method))
        self.Xtest = load_pkls_file(Preparation.path, "Xtest_{}_reduced".format(method))
        self.ytest = load_pkls_file(Preparation.path, "Xtest_static_with_labels_{}".format(method))["label"]

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
        model_search.fit(X=self.Xtrain, y=self.ytrain, groups=groups)
        return model_search.best_params_, model_search.best_score_

