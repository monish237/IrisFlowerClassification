# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Import Modules
import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

def printModule():
    print('Python: {}'.format(sys.version))
    print('scipy: {}'.format(scipy.__version__))
    print('numpy: {}'.format(numpy.__version__))
    print('matplotlib: {}'.format(matplotlib.__version__))
    print('pandas: {}'.format(pandas.__version__))
    print('sklearn: {}'.format(sklearn.__version__))
#printModule();

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
#print(dataset);

# Data Visualization (41-65):

# Shape: (Instance, Attributes)
#print(dataset.shape)

# head - Peek at Data - 20 rows
#print(dataset.head(20))

# numeric descriptions of data
#print(dataset.describe())

# class distribution - number of iris per group
#print(dataset.groupby('class').size())

# box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#pyplot.show()

# histograms
#dataset.hist()
#pyplot.show()

# scatter plot matrix
#scatter_matrix(dataset)
#pyplot.show()


# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

#arrays for checkBestAlgorithm
models = []
results = []
names = []

def checkBestAlgorithm():
    # Spot Check Algorithms
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    # evaluate each model in turn
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        # Prints accuracy of each algorithm
        #print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Checks for the best algorithm out of 'LR, LDA, KNN, CART, NB, SVM'
checkBestAlgorithm();

# Compares Algorithms using boxplots
#pyplot.boxplot(results, labels=names)
#pyplot.title('Algorithm Comparison')
#pyplot.show()

# Result: Support Vector Machines(SVM) was the most accurate - 98%
# SVM - An algorithm to find a hyperplane* in an N**-dimensional space.
# Hyperplane - An 'N minus 1' dimensional line/plane used to separate data
# N - amount of features of the data

# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
# Final Conclusion/ Result of an SVM algorithm
# Overall Accuracy: 96.66%
# Precision: 100% for Setosa and Versicolor, 86% for Virginica - Lower percentage = Higher false positives
# Recall: 100% for Setosa and Virginica, 92% for versicolor - Lower percentage = Higher false negatives
# f1-score: 100% for Setosa, 96% for versicolor, 92% for Viriginca - Combination of Precision and Recall