from preparation import *


def train_set_preparation_process(Preparation_instance):
    method = Preparation_instance.method
    Xtrain_static = load_pkls_file(Preparation.path, "Xtrain_static_with_labels_{}".format(method))
    if Xtrain_static is None:
        print("Train set: extracting and selecting features")
        Xtrain_static_selected, ytrain_static = Preparation_instance.feature_extraction_selection()
    else:
        print("Train set: loading extracted and selected features")
        Xtrain_static_selected = load_pkls_file(Preparation.path, "Xtrain_static_selected_{}".format(method))
        ytrain_static = Xtrain_static["label"]
    chosen_features = Xtrain_static_selected.columns
    print("Train set: reducing dimensions")
    if method == "Tsfresh":
        pca, Xtrain_static_reduced = dimension_reduction_pca(Xtrain_static_selected, 0.8, method)
    else:
        pca, Xtrain_static_reduced = dimension_reduction_pca(Xtrain_static_selected, 0.95, method)
    print("Train set: handling imbalanced data")
    Xtrain_resampled, ytrain_resampled = handling_imbalanced_data(Xtrain_static_reduced, ytrain_static)
    pd.DataFrame(Xtrain_resampled).to_pickle("Xtrain_{}_resampled.pkl".format(method))
    ytrain_resampled.to_pickle("ytrain_{}_resampled.pkl".format(method))
    return chosen_features, pca, method, Xtrain_resampled, ytrain_resampled

def preparation_process(Preparation_instance):
    # Xtrain preparation
    chosen_features, pca, method, Xtrain_resampled, ytrain_resampled = train_set_preparation_process(Preparation_instance)
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
p.to_pkl_train_test(Xtrain=v.Xtrain, ytrain=v.ytrain, Xtest=v.Xtest, ytest=v.ytest)

# Preparation - feature extraction by Tsfresh method
p = Preparation(method="Tsfresh")
Xtrain_resampled_tsfresh, ytrain_resampled_tsfresh, Xtest_reduced_tsfresh, ytest_static_tsfresh = preparation_process(Preparation_instance=p)

# Preparation - feature extraction by Naive method
p = Preparation(method="Naive")
Xtrain_resampled_naive, ytrain_resampled_naive, Xtest_reduced_naive, ytest_static_naive = preparation_process(Preparation_instance=p)









