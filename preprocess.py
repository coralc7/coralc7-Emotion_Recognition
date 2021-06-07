import os, glob
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import check_random_state
from collections import Counter, defaultdict


class StratifiedGroupKFold(_BaseKFold):
    """Stratified K-Folds iterator variant with non-overlapping groups.
    This cross-validation object is a variation of StratifiedKFold that returns
    stratified folds with non-overlapping groups. The folds are made by
    preserving the percentage of samples for each class.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    The difference between GroupKFold and StratifiedGroupKFold is that
    the former attempts to create balanced folds such that the number of
    distinct groups is approximately the same in each fold, whereas
    StratifiedGroupKFold attempts to create folds which preserve the
    percentage of samples for each class.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
    random_state : int or RandomState instance, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Examples
    --------
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold
    X = np.ones((17, 2))
    y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    groups = np.array([1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8])
    cv = StratifiedGroupKFold(n_splits=3)
    for train_idxs, test_idxs in cv.split(X, y, groups):
    ...     print("TRAIN:", groups[train_idxs])
    ...     print("      ", y[train_idxs])
    ...     print(" TEST:", groups[test_idxs])
    ...     print("      ", y[test_idxs])
    TRAIN: [2 2 4 5 5 5 5 6 6 7]
           [1 1 1 0 0 0 0 0 0 0]
     TEST: [1 1 3 3 3 8 8]
           [0 0 1 1 1 0 0]
    TRAIN: [1 1 3 3 3 4 5 5 5 5 8 8]
           [0 0 1 1 1 1 0 0 0 0 0 0]
     TEST: [2 2 6 6 7]
           [1 1 0 0 0]
    TRAIN: [1 1 2 2 3 3 3 6 6 7 8 8]
           [0 0 1 1 1 1 1 0 0 0 0 0]
     TEST: [4 5 5 5 5]
           [1 0 0 0 0]
    See also
    --------
    StratifiedKFold: Takes class information into account to build folds which
        retain class distributions (for binary or multiclass classification
        tasks).
    GroupKFold: K-fold iterator variant with non-overlapping groups.
    """

    def __init__(self, n_splits=10, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

    # Implementation based on this kaggle kernel:
    # https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    def _iter_test_indices(self, X, y, groups):
        labels_num = len(np.unique(y))
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, group in zip(y, groups):
            #print(group)
            y_counts_per_group[group][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        groups_and_y_counts = list(y_counts_per_group.items())
        rng = check_random_state(self.random_state)
        if self.shuffle:
            rng.shuffle(groups_and_y_counts)

        for group, y_counts in sorted(groups_and_y_counts,
                                      key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(self.n_splits):
                y_counts_per_fold[i] += y_counts
                std_per_label = []
                for label in range(labels_num):
                    std_per_label.append(np.std(
                        [y_counts_per_fold[j][label] / y_distr[label]
                         for j in range(self.n_splits)]))
                y_counts_per_fold[i] -= y_counts
                fold_eval = np.mean(std_per_label)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(group)

        for i in range(self.n_splits):
            test_indices = [idx for idx, group in enumerate(groups)
                            if group in groups_per_fold[i]]
            yield test_indices


class Preprocessing:
    path = 'C:\\Users\\Coral\\Desktop\\Emotion_Recognition'
    features_2_load = ['Name', 'FrameIndex', 'Brow Furrow', 'Brow Raise', 'Lip Corner Depressor', 'Smile',
                       'InnerBrowRaise', 'EyeClosure', 'NoseWrinkle', 'UpperLipRaise', 'LipSuck', 'LipPress', 'MouthOpen', 'ChinRaise',
                       'Smirk', 'LipPucker', 'Cheek Raise', 'Dimpler', 'Eye Widen', 'Lid Tighten', 'Lip Stretch',
                       'Jaw Drop', 'Pitch', 'Yaw', 'Roll', 'Anger', 'Sadness', 'Disgust', 'Joy', 'Surprise', 'Fear',
                       'Contempt']
    Affectiva_emotions = ['Anger', 'Sadness', 'Disgust', 'Joy', 'Surprise', 'Fear', 'Contempt']
    unique_labels = ['happy', 'surprise', 'anger', 'disgust', 'fear', 'sadness', 'contempt']
    dict_convert_labels_2_numeric = {'happy': 0, 'surprise': 1, 'anger': 2, 'disgust': 3, 'fear': 4, 'sadness': 5, 'contempt': 6}

    def __init__(self):
        self.load_data_and_set_attrs()
        self.split_train_test()

    def get_labels(self, data):
        movie_id_list = data["Name"].to_numpy()
        labels_list = []
        for movie_id in movie_id_list:
            movie_label = movie_id.split("-")[1]
            labels_list.append(movie_label)
        return movie_id_list, labels_list

    def get_entities(self, movie_id_list):
        entities_list = []
        for movie_id in movie_id_list:
            #movie_id = movie_id_list[0]
            movie_entity = movie_id.split("-")[0]
            entities_list.append(movie_entity)
        return entities_list

    def load_data_and_set_attrs(self):
        data_dir = os.path.join(Preprocessing.path, "data")
        pkl_dir = os.path.join(Preprocessing.path, "movies_data.pkl")
        if os.path.isfile(pkl_dir):
            pkl_file = open(pkl_dir, 'rb')
            data = pkl.load(pkl_file)
            pkl_file.close()
        else:
            movies_data_dir_list = sorted(list(f for f in glob.glob(os.path.join(data_dir, "*.txt"))))
            data = pd.DataFrame()
            for movie_dir in movies_data_dir_list:
                movie_data = pd.read_csv(movie_dir, sep="\t", skiprows=5, header=0)
                data = data.append(movie_data[Preprocessing.features_2_load])
            data.to_pickle(pkl_dir)
        movie_id_list, labels_list = self.get_labels(data)
        data_no_labels = data.drop(Preprocessing.Affectiva_emotions, axis=1).reset_index(drop=True)
        self.data = data_no_labels.copy()
        self.labels = labels_list
        self.movie_id_list = movie_id_list
        self.movies_length = data_no_labels["Name"].value_counts()
        self.entities_list = self.get_entities(movie_id_list)

    def get_movies_id_with_NaN(self):
        data = self.data.drop(["FrameIndex"], axis=1)
        movies_id_with_NaN = []
        for movie_id in self.movie_id_list:
            # movie_id = p.movie_id_list[0]
            movie_data = data[data["Name"] == movie_id].drop(["Name"], axis=1)
            movie_data_NaN = movie_data[movie_data.isnull().all(axis=1)]
            if movie_data_NaN.empty:
                continue
            movies_id_with_NaN.append(movie_id)
        return movies_id_with_NaN

    def split_train_test(self):
        #p = Preprocessing()
        # convert y, groups (entities) to numeric for using StratifiedGroupKFold
        X = self.movie_id_list
        labels = pd.Series(self.labels)
        y = labels.replace(Preprocessing.dict_convert_labels_2_numeric).to_list()
        groups = []
        for e in self.entities_list:
            groups.append(int(e[1:]))
        groups = pd.Series(groups).to_list()
        # split train and test
        cv_iter = StratifiedGroupKFold(n_splits=5, random_state=1, shuffle=True).split(X=X, y=y, groups=groups)
        for train_idx, test_idx in cv_iter:
            train = train_idx
            test = test_idx
            break
            # print("TRAIN:", train_idx, "TEST:", test_idx)
            #print("TRAIN:", len(train_idx), "TEST:", len(test_idx))
        self.Xtrain = self.data.iloc[train, ].reset_index(drop=True)
        self.ytrain = labels.iloc[train].reset_index(drop=True)
        self.Xtest = self.data.iloc[test, ].reset_index(drop=True)
        self.ytest = labels.iloc[test].reset_index(drop=True)

    def to_pkl_train_test(self, Xtrain, ytrain, Xtest, ytest):
        Xtrain.to_pickle("Xtrain.pkl")
        ytrain.to_pickle("ytrain.pkl")
        Xtest.to_pickle("Xtest.pkl")
        ytest.to_pickle("ytest.pkl")
