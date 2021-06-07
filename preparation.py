from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from visualization import *
from imblearn.over_sampling import SMOTE

def load_pkls_file(path, pkl_name):
    pkl_dir = os.path.join(path, "{}.pkl".format(pkl_name))
    if os.path.isfile(pkl_dir):
        pkl_file = open(pkl_dir, 'rb')
        pkl_data = pkl.load(pkl_file)
        pkl_file.close()
        return pkl_data
    else:
        print("There is no pickle file with the name {} in the directory {}".format(pkl_name, path))
        return None

def middle(df):
    if df.shape[0] % 2 == 0:
        return df.iloc[int(len(df) / 2) - 1:int(len(df) / 2) + 1]
    else:
        return df.iloc[int((len(df) / 2 - 0.5)) - 1:int(len(df) / 2 + 0.5)]

def add_number_each_element_string_list(string_list, mumber):
    new_string_list = []
    for elemnt in string_list:
        new_string_list.append(elemnt + str(mumber))
    return new_string_list

def concat_number_2_string(string):
    new_string = "S" + str(Preparation.number_2_concate) + "-" + string
    Preparation.number_2_concate += 1
    return new_string

def get_labels(movie_id_list):
    labels_list = []
    for movie_id in movie_id_list:
        movie_label = movie_id.split("-")[1]
        labels_list.append(movie_label)
    return labels_list

def dimension_reduction_pca_general(data, n_components, method):
    # scaling
    StSc = StandardScaler()
    data_scaled = StSc.fit_transform(data)
    # Fitting the PCA algorithm with our Data
    pca = PCA().fit(data_scaled)
    # Plotting the Cumulative Summation of the Explained Variance
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.axhline(y=n_components, color='r', linestyle='-')
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')  # for each component
    plt.title('Movies dataset explained variance')
    plt.show()
    plt.savefig('Movies_dataset_explained_variance_all_VAR_{}.png'.format(method))

def dimension_reduction_pca_specific(data, n_components, method):
    # scaling
    StSc = StandardScaler()
    data_scaled = StSc.fit_transform(data)
    # Fitting the PCA algorithm with our Data
    pca = PCA(n_components).fit(data_scaled)
    data_reduced = pd.DataFrame(pca.transform(data_scaled), index=data.index)
    # Plotting the Cumulative Summation of the Explained Variance
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')  # for each component
    plt.title('Movies dataset explained variance')
    plt.show()
    plt.savefig('Movies_dataset_explained_variance_{}_VAR_{}.png'.format(n_components, method))
    return pca, data_reduced

def handling_imbalanced_data(Xtrain, ytrain):
    X_resampled, y_resampled = SMOTE(sampling_strategy='minority').fit_resample(Xtrain, ytrain)
    new_labels = y_resampled.iloc[Xtrain.shape[0]:]
    new_labels = new_labels.apply(concat_number_2_string)
    Preparation.number_2_concate = 501
    X_resampled.index = list(Xtrain.index) + new_labels.to_numpy().tolist()
    y_resampled.index = X_resampled.index
    return X_resampled, y_resampled


class Preparation:
    path = Preprocessing.path
    dict_convert_labels_2_numeric = Preprocessing.dict_convert_labels_2_numeric
    feature_2_drop = Visualization.feature_2_drop
    unique_labels = Preprocessing.unique_labels
    number_2_concate = 501

    def __init__(self, method):
        self.method = method
        self.Xtrain = load_pkls_file(Preprocessing.path, "Xtrain")
        self.ytrain = load_pkls_file(Preprocessing.path, "ytrain")
        self.Xtest = load_pkls_file(Preprocessing.path, "Xtest")
        self.ytest = load_pkls_file(Preprocessing.path, "ytest")

    def tsfresh_feature_extraction_selection(self):
        extraction_settings = ComprehensiveFCParameters()
        Xtrain_static = extract_features(self.Xtrain, column_id='Name', column_sort='FrameIndex', default_fc_parameters=extraction_settings, impute_function=impute)
        labels = get_labels(list(Xtrain_static.index))
        Xtrain_static["label"] = labels
        Xtrain_static.to_pickle("Xtrain_static_with_labels_{}.pkl".format(self.method))
        Xtrain_static_selected = select_features(Xtrain_static.drop(["label"], axis=1), Xtrain_static["label"])
        Xtrain_static_selected.to_pickle("Xtrain_static_selected_{}.pkl".format(self.method))
        return Xtrain_static_selected, Xtrain_static["label"]

    def naive_feature_extraction_selection(self, data, data_name):
        movie_id_list = data["Name"].unique()
        features_name = data.drop(Preparation.feature_2_drop, axis=1).columns
        new_features_name = []
        for i in np.arange(1, 7):
            new_features_name = new_features_name + add_number_each_element_string_list(string_list=features_name, mumber=i)
        data_static = pd.DataFrame([], columns=new_features_name)
        for movie_id in movie_id_list:
            # movie_id = movie_id_list[0]
            m_data = data[data["Name"] == movie_id].drop(Preparation.feature_2_drop, axis=1).reset_index(drop=True)
            first_2_frames = m_data.iloc[0:2, :]
            last_2_frames = m_data.iloc[-2:, :]
            middle_2_frames = middle(m_data)
            concated_data = pd.concat([first_2_frames, middle_2_frames, last_2_frames], axis=0)
            new_row = pd.DataFrame([np.arange(len(new_features_name))], columns=new_features_name, index=[movie_id])
            for i, row_index in enumerate(concated_data.index):
                current_new_features_name = add_number_each_element_string_list(string_list=features_name, mumber=i + 1)
                for j, feature in enumerate(current_new_features_name):
                    new_row.loc[movie_id, feature] = concated_data.loc[row_index, features_name[j]]
            data_static = data_static.append(new_row)
        labels = get_labels(list(data_static.index))
        data_static.to_pickle("{}_static_selected_{}.pkl".format(data_name, self.method))
        data_static_with_labels = data_static.copy()
        data_static_with_labels["label"] = labels
        data_static_with_labels.to_pickle("{}_static_with_labels_{}.pkl".format(data_name, self.method))
        return data_static, data_static_with_labels["label"]

    def feature_extraction_selection(self):
        if self.method == "Tsfresh":
            return self.tsfresh_feature_extraction_selection()
        else:
            return self.naive_feature_extraction_selection(data=self.Xtrain, data_name="Xtrain")

    def test_PCA(self, pca, Xtest):
        StSc = StandardScaler()
        Xtest_static_scaled = StSc.fit_transform(Xtest)
        Xtest_reduced = pca.transform(Xtest_static_scaled)
        Xtest_reduced = pd.DataFrame(Xtest_reduced, index=Xtest.index)
        Xtest_reduced.to_pickle("Xtest_{}_reduced.pkl".format(self.method))
        return Xtest_reduced

    def test_preparation(self, chosen_features, pca):
        if self.method != "Naive":
            extraction_settings = ComprehensiveFCParameters()
            Xtest_static = extract_features(self.Xtest, column_id='Name', column_sort='FrameIndex', default_fc_parameters=extraction_settings, impute_function=impute)
            labels = get_labels(list(Xtest_static.index))
            Xtest_static["label"] = labels
            Xtest_static.to_pickle("Xtest_static_with_labels_{}.pkl".format(self.method))
            Xtest_static_selected = Xtest_static[chosen_features]
            Xtest_static_selected.to_pickle("Xtest_static_selected_{}.pkl".format(self.method))
            # pca
            Xtest_reduced = self.test_PCA(pca=pca, Xtest=Xtest_static_selected)
            return Xtest_reduced, Xtest_static["label"]
        else:
            Xtest_static, ytest_static = self.naive_feature_extraction_selection(self.Xtest, "Xtest")
            Xtest_reduced = self.test_PCA(pca=pca, Xtest=Xtest_static)
            return Xtest_reduced, ytest_static
