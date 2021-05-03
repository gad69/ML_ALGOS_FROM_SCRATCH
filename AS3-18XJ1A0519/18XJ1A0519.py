import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier#used for graphs 
from sklearn import tree
import pandas as pd
from timeit import default_timer as timer
from sklearn.ensemble import BaggingClassifier
from sklearn import model_selection
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

def accuracy(y_true, y_pred):
   
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return (accuracy)
class node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:

    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self.grow_tree(X, y)

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])

    def grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split):
            leaf_value = self.most_common_label(y)
            return node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedily select the best split according to information gain
        best_feat, best_thresh = self.best_criteria(X, y, feat_idxs)
        
        # grow the children that result from the split
        left_idxs, right_idxs = self.split(X[:, best_feat], best_thresh)
        left = self.grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self.grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return node(best_feat, best_thresh, left, right)

    def best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self.information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def information_gain(self, y, X_column, split_thresh):
        # parent loss
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self.split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    def split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)

    def most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

if __name__ == "__main__":
    start = timer()
    print("NOTE : BANK_NOTE_DATA_SET IS COMMA SEPARATED AND SDD IS SPACE SEPARATED AND TO TRAIN SENSORY DATA SET IT TAKES LONGER TIME AS USUAL BUT THE PRERECORDER ACCURACIES ARE RECORDER IN WORD DOCUMENT")
    choose = int(input("Choose from below\n1.Basic Decision AND bagging  for bank data set (can be any comma separated data set) \n2.Basic Decision tree AND bagging for Sensory (can be any space separated data set) SDD DATA SET TAKES LONGER TIME THAN USUAL \n3.visualization for BNA or (any comma separated data set)\n4.visualization for SDD or (any space separated)\n"))
    filename = input("Enter the file name: ")
    if choose == 1:
        n = int(input("Enter no of columns EX 5 COLUMNS FOR BANKNOTE DATA SET"))
        u = int(input("Enter the tree depth "))
        z = int(input("Enter no of trees K"))
        o = int(input("Enter percent of overlap"))
        data = pd.read_csv(filename)
        x = data.iloc[:,0:n-1].values #separating x and y 
        y = data.iloc[:,-1].values
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1234)
        clf = DecisionTree(max_depth=u)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("missclassification of each class " ,1-(cm.diagonal()))
        print ("Accuracy for basictree:", acc)
        print ("misclassification for basictree:",1-acc)
        kfold = model_selection.KFold(n_splits =z)
        clf = BaggingClassifier(n_estimators=o, random_state=0).fit(x, y)
        results = model_selection.cross_val_score(clf, X_test,y_test, cv=kfold)
        print("accuracies of the individual trees " ,results)
        
        print ("Accuracy for bagging:", results.mean())
        
    elif choose == 2:
        n = int(input("Enter no of columns EX 49 COLUMNS FOR Sensory data set {takes longer time than usual}"))
        u = int(input("Enter the tree depth "))
        z = int(input("Enter no of trees K"))
        o = int(input("Enter percent of overlap"))
        data = pd.read_csv(filename,sep='\s+')
        x = data.iloc[:,0:n-1].values #separating x and y 
        y = data.iloc[:,-1].values
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1234)
        clf = DecisionTree(max_depth=u)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("missclassification of each class " ,1-(cm.diagonal()))
        print ("Accuracy for basictree:", acc)
        print ("misclassification for basictree:",1-acc)
        kfold = model_selection.KFold(n_splits =z)
        clf = BaggingClassifier(n_estimators=o, random_state=0).fit(x, y)
        results = model_selection.cross_val_score(clf, X_test,y_test, cv=kfold)
        print("accuracies of the individual trees " ,results)
        print ("Accuracy for bagging:", results.mean())

    elif choose == 3: #graph for BN DATA
        data = pd.read_csv(filename)
        #data.columns = ['feature1','feature2','feature3','feature4','class']
        n = int(input("Enter no of columns EX 5 COLUMNS FOR BANKNOTE DATA SET"))
        x = data.iloc[:,0:n-1] #separating x and y 
        y = data.iloc[:,-1]
        clf = DecisionTreeClassifier(random_state=1234)
        model = clf.fit(x, y)
        text_representation = tree.export_text(clf)
        print(text_representation)
       
        dt_feature_names = list(data.columns)
        dt_target_names = [str(s) for s in y.unique()]
        fig = plt.figure(figsize=(50,40))
        _ = tree.plot_tree(clf, 
                   feature_names=dt_feature_names,
                   class_names=dt_target_names,
                   filled=True)
    elif choose == 4: #GRAPH FOR SDD
        data = pd.read_csv(filename,sep='\s+')
        #data.columns = ['feature1','feature2','feature3','feature4','class']
        n = int(input("Enter no of columns EX 49 COLUMNS FOR SDD"))
        
        x = data.iloc[:,0:n-1]
        y = data.iloc[:,-1]
        clf = DecisionTreeClassifier(random_state=1234)
        model = clf.fit(x, y)
        text_representation = tree.export_text(clf)
        print(text_representation)
        dt_feature_names = list(data.columns)
        dt_target_names = [str(s) for s in y.unique()]
        fig = plt.figure(figsize=(25,20))
        _ = tree.plot_tree(clf, 
                   feature_names=dt_feature_names,
                   class_names=dt_target_names,
                   filled=True)
        