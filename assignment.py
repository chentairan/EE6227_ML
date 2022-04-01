import scipy.io as scio
from numpy import mean
from numpy import std
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

def import_datasets():
    train_data = scio.loadmat("data/Data_Train.mat")
    train_label = scio.loadmat("data/Label_Train.mat")
    test_data = scio.loadmat("data/Data_test.mat")
    return train_data['Data_Train'], train_label['Label_Train'], test_data['Data_test']

def LDA(train_data, train_label, test_data):
    # define model
    model = LinearDiscriminantAnalysis()
    
    X_r_lda = model.fit(train_data, train_label.ravel()).transform(train_data)
    test_label = model.predict(test_data)
    Test_r_lda = model.transform(test_data)


    with plt.style.context('seaborn-talk'):
        fig, axes = plt.subplots(1,1,figsize=[15,15],dpi=300)
        colors = ['navy', 'turquoise', 'darkorange']
        for color, i, target_name in zip(colors, [1, 2, 3], 'test'):
            axes.scatter(X_r_lda[train_label.ravel() == i, 0], X_r_lda[train_label.ravel() == i, 1], alpha=.8, label=target_name, color=color)
            # axes[1].scatter(X_r_pca[y == i, 0], X_r_pca[y == i, 1], alpha=.8, label=target_name, color=color)
        axes.title.set_text('LDA visualization for training dataset')
        # axes[1].title.set_text('PCA for Wine dataset')
        axes.set_xlabel('Discriminant Coordinate 1')
        axes.set_ylabel('Discriminant Coordinate 2')
    plt.savefig('LDA_train.png')

    with plt.style.context('seaborn-talk'):
        fig, axes = plt.subplots(1,1,figsize=[15,15],dpi=300)
        colors = ['navy', 'turquoise', 'darkorange']
        for color, i, target_name in zip(colors, [1, 2, 3], 'test'):
            axes.scatter(Test_r_lda[test_label.ravel() == i, 0], Test_r_lda[test_label.ravel() == i, 1], alpha=.8, label=target_name, color=color)
            # axes[1].scatter(X_r_pca[y == i, 0], X_r_pca[y == i, 1], alpha=.8, label=target_name, color=color)
        axes.title.set_text('LDA visualization for test dataset')
        # axes[1].title.set_text('PCA for Wine dataset')
        axes.set_xlabel('Discriminant Coordinate 1')
        axes.set_ylabel('Discriminant Coordinate 2')
    plt.savefig('LDA_test.png')

    print(test_label)

def Bayes(train_data, train_label, test_data):
    model = GaussianNB()
    y_pred = model.fit(train_data, train_label.ravel()).predict(test_data)
    print(y_pred)

def DTree(train_data, train_label, test_data):
    model = DecisionTreeClassifier()
    y_pred = model.fit(train_data, train_label.ravel()).predict(test_data)
    print(y_pred)
    fig = plt.figure(figsize=(25,20), dpi=300)
    _ = plot_tree(model, 
                    feature_names=['feature 1', 'feature 2', 'feature 3', 'feature 4'],  
                    class_names=['class 1', 'class 2', 'class 2'],
                    filled=True)
    plt.savefig('decision_tree.png')

if __name__ == '__main__':
    train_data, train_label, test_data = import_datasets()
    Bayes(train_data, train_label, test_data)
    LDA(train_data, train_label, test_data)
    DTree(train_data, train_label, test_data)



