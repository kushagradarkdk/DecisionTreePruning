from DecisionTree import *
import pandas as pd
from sklearn import model_selection

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

filename = Path(r"C:\Users\darku\OneDrive\Desktop\DecisionTree\diabetes.csv")
#header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']
header = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
#df = pd.read_csv('abc', header=None, names=['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label'])
#header = ['A1','A2','A3','A4','A5','A6','A7','A8','Class']
df = pd.read_csv(r"C:\Users\darku\OneDrive\Desktop\DecisionTree\diabetes.csv", encoding='utf-8', names=['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label'])
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data',header=None, names=['A1','A2','A3','A4','A5','A6','A7','A8','Class'])
#df_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.tst',header=None, names=['A1','A2','A3','A4','A5','A6','A7','A8','Class'])
lst = df.values.tolist()
t = build_tree(lst, header)
print_tree(t)

print("********** Leaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t)
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))

trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)
print("*************Tree before pruning*******")
print_tree(t)
testacc = computeAccuracy(test, t)
print("Accuracy on test = " + str(testacc))

# t_pruned = prune_tree(t, [26, 11, 5])
print("*************Tree after pruning*******")
# print_tree(t_pruned)
# acc = computeAccuracy(test, t_pruned)
# print("Accuracy on test after Pruning = " + str(acc))
## TODO: You have to decide on a pruning strategy
nodesId = []
for nodes in innerNodes:
    nodesId.append(nodes.id)
for nodesId in nodesId:
    print(nodesId)
    t_pruned = prune_tree(t, [nodesId])
    acc_afterpruning = computeAccuracy(test, t_pruned)
    if(acc_afterpruning>testacc):
        print_tree(t_pruned)
        print("Accuracy on test after Pruning = " + str(acc_afterpruning))
    t = build_tree(lst, header)