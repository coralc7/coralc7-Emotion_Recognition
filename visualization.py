from preprocess import *
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from matplotlib import pyplot as plt


def get_vector_norm(x, y, z, df):
    return np.sqrt(df[x] ** 2 + df[y] ** 2 + df[z] ** 2)


class Visualization:
    Preprocessing_instance = Preprocessing()
    feature_2_drop = ["Name", "FrameIndex"]
    unique_labels = Preprocessing_instance.unique_labels
    AUs = ['Brow Furrow', 'Brow Raise', 'Lip Corner Depressor', 'Smile', 'InnerBrowRaise', 'EyeClosure', 'NoseWrinkle',
           'UpperLipRaise', 'LipSuck', 'LipPress', 'MouthOpen', 'ChinRaise', 'Smirk', 'LipPucker', 'Cheek Raise',
           'Dimpler', 'Eye Widen', 'Lid Tighten', 'Lip Stretch', 'Jaw Drop']
    head_features = ['Pitch', 'Yaw', 'Roll']

    def __init__(self):
        self.data = Visualization.Preprocessing_instance.data
        self.labels = Visualization.Preprocessing_instance.labels
        self.movie_id_list = Visualization.Preprocessing_instance.movie_id_list
        self.entities_list = Visualization.Preprocessing_instance.entities_list
        self.Xtrain = Visualization.Preprocessing_instance.Xtrain
        self.ytrain = Visualization.Preprocessing_instance.ytrain
        self.Xtest = Visualization.Preprocessing_instance.Xtest
        self.ytest = Visualization.Preprocessing_instance.ytest

    def plot_labels_distribution(self):
        plot = sns.countplot(x=self.labels)
        total = len(self.labels)
        for patch in plot.patches:
            percentage = '{:.1f}%'.format(100 * (patch.get_height() / total))
            x = patch.get_x() + patch.get_width() / 6
            y = patch.get_y() + patch.get_height() + 0.5
            plot.annotate(percentage, (x, y), size=12)
        title = "Labels Distribution"
        plt.title(title)
        plt.xlabel("Labels")
        plt.savefig(title + ".png")

    def cor_matrix(self, is_mean_movie=False):
        fig, ax = plt.subplots(1, 1, figsize=(13, 9))
        if is_mean_movie:
            data = self.Xtrain.drop(["FrameIndex"], axis=1)
            data = data.groupby("Name").mean()
        else:
            data = self.Xtrain.drop(Visualization.feature_2_drop, axis=1)
        sns.heatmap(data.corr(),
                    ax=ax,
                    cmap='coolwarm',
                    annot=True,
                    fmt='.2f',
                    linewidths=0.05)
        fig.subplots_adjust(top=0.93)
        if is_mean_movie:
            title = 'Correlation Heatmap for average features in movie Train dataset'
        else:
            title = 'Correlation Heatmap for Train dataset'
        fig.suptitle(title,
                     fontsize=14,
                     fontweight='bold')
        plt.savefig(title + ".png")

    def get_VIF_data(self):
        data = self.Xtrain.drop(Visualization.feature_2_drop, axis=1)
        vif_data = pd.Series([variance_inflation_factor(data.values, i) for i in range(data.shape[1])], index=data.columns)
        return vif_data

    def drop_features(self, features_list, is_Xtrain=True):
        if is_Xtrain:
            self.Xtrain = self.Xtrain.drop(features_list, axis=1)
        else:
            self.Xtest = self.Xtest.drop(features_list, axis=1)

    def test_train_distribution(self):
        features = self.Xtrain.drop(Visualization.feature_2_drop, axis=1).columns.to_numpy()
        sns.set_style('whitegrid')
        for feature in features:
            #feature = features[0]
            plt.figure()
            train_2_plot = self.Xtrain[feature].reset_index(drop=True).to_numpy(dtype='float')
            test_2_plot = self.Xtest[feature].reset_index(drop=True).to_numpy(dtype='float')
            sns.kdeplot(train_2_plot, label='Train')
            sns.kdeplot(test_2_plot, label='Test')
            plt.xlabel(feature)
            plt.legend()
            title = 'Distribution of feature: {}'.format(feature)
            plt.title(title)
            plt.savefig('train_test_{}_distribution.png'.format(feature))

    def plot_features_of_label(self, movie_id):
        #movies_id_list_train = np.unique(v.Xtrain["Name"])
        #movie_id = movies_id_list_train[0]
        data_2_plot = self.Xtrain[self.Xtrain["Name"] == movie_id]
        x_values = data_2_plot["FrameIndex"]
        sns.set_style('whitegrid')
        AUs_2_plot = list(set(data_2_plot.columns) - set(Visualization.feature_2_drop + Visualization.head_features))
        plt.figure(figsize=(9,7))
        for feature in AUs_2_plot:
            plt.plot(x_values, data_2_plot[feature].to_numpy(), label=feature)
        plt.xlabel("Frame Index")
        plt.ylabel("AUs")
        plt.legend()
        splitted_movie_id = movie_id.split("-")
        plt.title("Facial Features, Entity: {}, Label: {}".format(splitted_movie_id[0], splitted_movie_id[1]))
        plt.savefig("Facial Features_Entity_{}_Label_{}_.png".format(splitted_movie_id[0], splitted_movie_id[1]))
        plt.figure(figsize=(9, 7))
        for feature in Visualization.head_features:
            plt.plot(x_values, data_2_plot[feature].to_numpy(), label=feature)
        plt.xlabel("Frame Index")
        plt.ylabel("Head features")
        plt.legend()
        plt.title("Head Features, Entity: {}, Label: {}".format(splitted_movie_id[0], splitted_movie_id[1]))
        plt.savefig("Head Features_Entity_{}_Label_{}_.png".format(splitted_movie_id[0], splitted_movie_id[1]))

    def Xtrain_summary_statistics(self):
        return self.Xtrain.drop(Visualization.feature_2_drop, axis=1).describe()

    def boxplot(self, feature):
        #feature = "head_features_norm"
        new_df = self.Xtrain.groupby('Name')[feature].mean().reset_index()
        _, labels_list = Visualization.Preprocessing_instance.get_labels(new_df)
        new_df['Emotion'] = labels_list
        plt.figure()
        sns.boxplot(x='Emotion', y=feature, data=new_df)
        plt.title('{} vs emotion'.format(feature))
        plt.savefig("boxplot_{}.png".format(feature))

