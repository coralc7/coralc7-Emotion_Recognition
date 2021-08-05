from modelSelection import *
import time
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from numpy.random import seed
from sklearn.metrics import plot_confusion_matrix


def train_set_preparation_process(Preparation_instance):
    method = Preparation_instance.method
    Xtrain_static = load_pkls_file(Preparation.path, "Xtrain_static_with_labels_{}".format(method))
    if Xtrain_static is None:
        print("Train set: extracting and selecting features")
        Xtrain_static_selected, ytrain_static = Preparation_instance.Xtrain_feature_extraction_selection()
    else:
        print("Train set: loading extracted and selected features")
        Xtrain_static_selected = load_pkls_file(Preparation.path, "Xtrain_static_selected_{}".format(method))
        ytrain_static = Xtrain_static["label"]
    chosen_features = Xtrain_static_selected.columns
    print("Train set: reducing dimensions")
    if method == "Tsfresh":
        dimension_reduction_pca_general(Xtrain_static_selected, 0.8, method)
        pca, Xtrain_static_reduced = dimension_reduction_pca_specific(Xtrain_static_selected, 0.8, method)
    else:
        dimension_reduction_pca_general(Xtrain_static_selected, 0.95, method)
        pca, Xtrain_static_reduced = dimension_reduction_pca_specific(Xtrain_static_selected, 0.95, method)
    print("Train set: handling imbalanced data")
    Xtrain_resampled, ytrain_resampled = handling_imbalanced_data(Xtrain_static_reduced, ytrain_static)
    pd.DataFrame(Xtrain_resampled).to_pickle("Xtrain_{}_resampled.pkl".format(method))
    ytrain_resampled.to_pickle("ytrain_{}_resampled.pkl".format(method))
    return chosen_features, pca, method, Xtrain_resampled, ytrain_resampled

def preparation_process(Preparation_instance):
    # Xtrain preparation
    chosen_features, pca, method, Xtrain_resampled, ytrain_resampled = train_set_preparation_process(
        Preparation_instance)
    # test_preparation - the same wat as Xtrain - by Naive method
    Xtest_static = load_pkls_file(Preparation.path, "Xtest_static_with_labels_{}".format(method))
    if Xtest_static is None:
        print("Test set: extracting, selecting features and reducing dimensions as the same way as X train")
        Xtest_reduced, ytest_static = Preparation_instance.test_preparation(chosen_features=chosen_features, pca=pca)
    else:
        print("Test set: loading extracted, selected features and reduced dimensions")
        Xtest_reduced = load_pkls_file(Preparation.path, "Xtest_{}_reduced".format(method))
        ytest_static = load_pkls_file(Preparation.path, "Xtest_static_with_labels_{}".format(method))["label"]
    return Xtrain_resampled, ytrain_resampled, Xtest_reduced, ytest_static

def evaluation(model, Xtrain, ytrain, Xtest, ytest, model_name):
    model.fit(Xtrain, ytrain)
    ytest_pred = model.predict(Xtest)
    #ytrain_pred = Naive_GBM.predict(Xtrain)
    ytest_pred_series = pd.Series(ytest_pred)
    ytest_pred_series.to_csv('ytest_pred_' + model_name + '.csv', index=False, header=False)
    print("The accuracy score of " + model_name, str(np.round(accuracy_score(ytest, ytest_pred), 2)))
    print("The f1 score of model " + model_name,
          str(np.round(f1_score(ytest, ytest_pred, average='weighted'), 2)))
    print("The f1 micro of model " + model_name,
          str(np.round(f1_score(ytest, ytest_pred, average='micro'), 2)))
    #print("The accuracy score of model " + model_name +" for train set",
          #str(np.round(accuracy_score(ytrain, ytrain_pred), 2)))
    #print("The f1 score of model " + model_name +" for train set",
          #str(np.round(f1_score(ytrain, ytrain_pred, average='weighted'), 2)))
    #print("The f1 micro of model " + model_name +" for train set",
          #str(np.round(f1_score(ytrain, ytrain_pred, average='micro'), 2)))


# Preprocessing and Visualization
p = Preprocessing()
p.get_movies_id_with_NaN()

# data understanding - visualization
v = Visualization()
v.plot_labels_distribution()
v.cor_matrix()
vif_data = v.get_VIF_data()
vif_data.to_csv("first_VIF_data.csv")
vif_above_10 = vif_data[vif_data > 10]
v.cor_matrix(is_mean_movie=True)
v.test_train_distribution()
for movie_id in np.unique(v.Xtrain["Name"]):
    v.plot_features_of_label(movie_id)
# adding new feature:
v.Xtrain["head_features_norm"] = get_vector_norm("Pitch", "Yaw", "Roll", v.Xtrain)
v.Xtest["head_features_norm"] = get_vector_norm("Pitch", "Yaw", "Roll", v.Xtest)
# statistics_summary
Xtrain_statistics_summary = v.Xtrain_statistics_summary()
Xtrain_statistics_summary.to_csv("Xtrain_statistics_summary.csv")
# v.cor_matrix()
features_2_boxplot = list(set(v.Xtrain.columns) - set(Visualization.feature_2_drop))
for feature in features_2_boxplot:
    v.boxplot("head_features_norm")
# WHAT TO DO ??
p.to_pkl_train_test(Xtrain=v.Xtrain, ytrain=v.ytrain, Xtest=v.Xtest, ytest=v.ytest)

# Preparation - feature extraction by Tsfresh method
p = Preparation(method="Tsfresh")
Xtrain_resampled_tsfresh, ytrain_resampled_tsfresh, Xtest_reduced_tsfresh, ytest_static_tsfresh = preparation_process(
    Preparation_instance=p)

# Preparation - feature extraction by Naive method
p = Preparation(method="Naive")
Xtrain_resampled_naive, ytrain_resampled_naive, Xtest_reduced_naive, ytest_static_naive = preparation_process(
    Preparation_instance=p)

# Example of hyperparameter Tuning Naive GBM
m = ModelSelection("Naive")
cv_iter, groups = m.get_cv_iter(random_state=2)
param_lgb = {"max_depth": [12, 15, 20, 25],
             "learning_rate": [0.3, 0.4, 0.5],
             "num_leaves": [34, 36, 50, 55],
             "n_estimators": [145, 155, 166],
             "boosting_type": ['gbdt', 'goss'],
             'min_child_samples': [15, 20, 35, 45],
             'subsample': [0.1, 0.01],
             'reg_alpha': [0]
             }
lg = lgb.LGBMClassifier(class_weight='balanced')
start = time.time()
model_best_params, model_best_score = m.hyperparameter_tuning(param_lgb, lg, cv_iter, groups)
end = time.time()
total_time = end - start
print("total time is: {} sec".format(total_time))
print("total time is: {} hours".format(total_time / 3600))
print("The best params are:")
print(model_best_params)
print("The best score is: ", model_best_score)

# Example of hyperparameter Tuning Naive RF
m = ModelSelection("Naive")
cv_iter, groups = m.get_cv_iter(random_state=2)
rf_param = {'max_depth': list(range(80, 100, 1)),
            'n_estimators': list(range(30, 40, 1)),
            'max_features': ['sqrt', 'auto', 'log2'],
            'min_samples_split': list(range(5, 20, 1)),
            'min_samples_leaf': list(range(3, 10, 1)),
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy']
            }
RF = RandomForestClassifier(class_weight='balanced')
start = time.time()
model_best_params, model_best_score = m.hyperparameter_tuning(rf_param, RF, cv_iter, groups)
end = time.time()
total_time = end - start
print("total time is: {} sec".format(total_time))
print("total time is: {} hours".format(total_time / 3600))
print("The best params are:")
print(model_best_params)
print("The best score is: ", model_best_score)

# Example of hyperparameter Tuning Tsfresh RF
m = ModelSelection("Tsfresh")
cv_iter, groups = m.get_cv_iter(random_state=2)
rf_param = {'max_depth': list(range(20, 200, 20)) + [None],
            'n_estimators': list(range(5, 40, 5)),
            'max_features': ['sqrt', 'auto', 'log2'],
            'min_samples_split': list(range(5, 40, 5)),
            'min_samples_leaf': list(range(5, 40, 5)),
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy']
            }
RF = RandomForestClassifier(class_weight='balanced')
model_best_params, model_best_score = m.hyperparameter_tuning(rf_param, RF, cv_iter, groups)

# Example of hyperparameter Tuning Tsfresh GBM
m = ModelSelection("Tsfresh")
cv_iter, groups = m.get_cv_iter(random_state=2)
param_lgb = {"max_depth": [10, 15, 80, 90],
             "learning_rate": [0.15, 0.1, 0.2],
             "num_leaves": [10, 20, 30],
             "n_estimators": [80, 100, 200],
             "boosting_type": ['gbdt', 'goss'],
             'min_child_samples': [40, 50, 60],
             'subsample': [0.1, 0.05],
             'reg_alpha': [0, 0.05, 0.1]
             }
lg = lgb.LGBMClassifier(class_weight='balanced')
start = time.time()
model_best_params, model_best_score = m.hyperparameter_tuning(param_lgb, lg, cv_iter, groups)
end = time.time()
total_time = end - start
print("total time is: {} sec".format(total_time))
print("total time is: {} hours".format(total_time / 3600))
print("The best params are:")
print(model_best_params)
print("The best score is: ", model_best_score)

# The 4 final models:
tsfresh_RF = RandomForestClassifier(n_estimators=20,
                                    min_samples_split=6,
                                    min_samples_leaf=4,
                                    max_features='auto',
                                    max_depth=164,
                                    bootstrap=False,
                                    criterion='gini')
Naive_RF = RandomForestClassifier(n_estimators=34,
                                  min_samples_split=14,
                                  min_samples_leaf=6,
                                  max_features='auto',
                                  max_depth=90,
                                  bootstrap=False,
                                  criterion='entropy')
tsfresh_GBM = lgb.LGBMClassifier(max_depth=8,
                                 learning_rate=0.19,
                                 num_leaves=8,
                                 n_estimators=70,
                                 boosting_type='gbdt',
                                 min_child_samples=60,
                                 subsample=0.1,
                                 reg_alpha=0)
Naive_GBM = lgb.LGBMClassifier(max_depth=12,
                               learning_rate=0.2,
                               num_leaves=35,
                               n_estimators=150,
                               boosting_type='goss',
                               min_child_samples=20,
                               subsample=0.1,
                               reg_alpha=0)

ModelSelection_instance = ModelSelection("Naive")
Xtrain_Naive = ModelSelection_instance.Xtrain
ytrain_Naive = ModelSelection_instance.ytrain
Xtest_Naive = ModelSelection_instance.Xtest
ytest_Naive = ModelSelection_instance.ytest

ModelSelection_instance = ModelSelection("Tsfresh")
Xtrain_tsfresh = ModelSelection_instance.Xtrain
ytrain_tsfresh = ModelSelection_instance.ytrain
Xtest_tsfresh = ModelSelection_instance.Xtest
ytest_tsfresh = ModelSelection_instance.ytest

seed(1)
# fit and predict on test set - GBM - Naive
evaluation(model=Naive_GBM, Xtrain=Xtrain_Naive, ytrain=ytrain_Naive, Xtest=Xtest_Naive, ytest=ytest_Naive, model_name="Naive_GBM")
# fit and predict on test set - RF - Naive
evaluation(model=Naive_RF, Xtrain=Xtrain_Naive, ytrain=ytrain_Naive, Xtest=Xtest_Naive, ytest=ytest_Naive, model_name="Naive_RF")
# fit and predict on test set - GBM - Tsfresh
evaluation(model=tsfresh_GBM, Xtrain=Xtrain_tsfresh, ytrain=ytrain_tsfresh, Xtest=Xtest_tsfresh, ytest=ytest_tsfresh, model_name="tsfresh_GBM")
# fit and predict on test set - RF - Tsfresh
evaluation(model=tsfresh_RF, Xtrain=Xtrain_tsfresh, ytrain=ytrain_tsfresh, Xtest=Xtest_tsfresh, ytest=ytest_tsfresh, model_name="tsfresh_RF")

# confusion matrix - the best model - RF-Tsfresh
plot_confusion_matrix(tsfresh_RF, Xtest_tsfresh, ytest_tsfresh, cmap=plt.cm.Blues,  normalize='pred')
plt.title("confusion matrix - the best model - RF-Tsfresh")
plt.show()