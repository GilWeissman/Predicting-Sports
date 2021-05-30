import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree

def filter_data1(data):
    filter_arr = []
    for game in data:
        if game[9] > 2 and game[15] > 2:
            filter_arr.append(True)
        else:
            filter_arr.append(False)
    data = data[filter_arr]
    selections11 = np.array([False, True, True, True, True, True, True, True, True, False, True, True, True,
                            True, True, False])
    selections1 = np.array([False, True, True, True, False, False, False, False, False, False, False, False, False,
                            False, False, False])
    selections2 = np.array([True, False, False, False, False, False, False, False, False, False, False, False, False,
                            False, False, False])
    selections3 = np.array([False, False, False, False, False, False, True, False, False, False, False, False, False,
                            False, False, False])
    y1 = data[:, selections2]
    y3 = data[:, selections3]
    y2 = []
    for y in y1:
        y2.append(y[0])
    y4 = []
    for y in y3:
        y4.append(y[0])
    return data[:, selections1], y2, y4

def filter_data2(data):
    filter_arr = []
    for game in data:
        if 37 > game[9] > 2 < game[15] < 37:
            filter_arr.append(True)
        else:
            filter_arr.append(False)
    data = data[filter_arr]
    selections11 = np.array([False, True, True, True, True, True, True, True, True, False, True, True, True,
                            True, True, False])
    selections1 = np.array([False, True, True, True, False, False, False, False, False, False, False, False, False,
                            False, False, False])
    selections2 = np.array([True, False, False, False, False, False, False, False, False, False, False, False, False,
                            False, False, False])
    selections3 = np.array([False, False, False, False, False, False, True, False, False, False, False, False, False,
                            False, False, False])
    y1 = data[:, selections2]
    y3 = data[:, selections3]
    y2 = []
    for y in y1:
        y2.append(y[0])
    y4 = []
    for y in y3:
        y4.append(y[0])
    return data[:, selections1], y2, y4

adress_train_PL = 'trainingSetPL.csv'
adress_test_PL = 'testSetPL.csv'
adress_current_test_PL = 'currentTestSetPL.csv'
names1 = ['', 'date', 'HomeTeam', 'GoalsFH', 'GoalsAH', 'ShootsTargetFH', 'ShootsTargetAH', 'B365H', 'B365D', 'B365A',
          'FTR', 'Venue', 'ptsH', 'CT', 'Count', 'RDate', 'Games', 'GlspgaH', 'GlspgfH', 'STpgaH', 'STpgfH', 'PtspgH',
          'CumptsH', 'CumG', 'CumglsAH', 'CumglsfH', 'CumsTAH', 'CumsTfH', 'date', 'AwayTeam', 'GoalsFA',
          'GoalsAA', 'ShootsTargetFA', 'ShootsTargetAA', 'B365H', 'B365D', 'B365A', 'FTR', 'Venue', 'ptsA', 'CT',
          'Count', 'RDate', 'Games', 'GlspgaA', 'GlspgfA', 'STpgaA', 'STpgfA', 'PtspgA', 'CumptsA', 'CumG',
          'CumglsAA', 'CumglsfA', 'CumsTAA', 'CumsTfA', 'year']
trainingSet = pd.read_csv(adress_train_PL, header=0)
testSet = pd.read_csv(adress_test_PL, header=0)
train = trainingSet.iloc[:, [10, 7, 8, 9, 17, 18, 19, 20, 21, 23, 44, 45, 46, 47, 48, 50]].values
X_train, y_train, check_train = filter_data1(train)
test = testSet.iloc[:, [10, 7, 8, 9, 17, 18, 19, 20, 21, 23, 44, 45, 46, 47, 48, 50]].values
X_test, y_test, check_test = filter_data2(test)

# clf1 = LogisticRegression(max_iter=100000, class_weight={'H': 0, 'D': 0.7, 'A': 0.3}).fit(X_train, y_train)
clf2 = LogisticRegression(max_iter=100000).fit(X_train, y_train)
# clf2 = KNeighborsClassifier(n_neighbors=15).fit(X_train, y_train)
# clf3 = tree.DecisionTreeClassifier().fit(X_train, y_train)
# clf4 = SGDClassifier(loss="hinge", penalty="l2", max_iter=10000).fit(X_train, y_train)

# predicted_y = clf1.predict(X_test)
# print(classification_report(y_test, predicted_y))
# print(confusion_matrix(y_test, predicted_y))

value = 0
count = 0
count2 = 0
sum = 0
for i in range(len(X_test)):
    game = X_test[i]
    bet365 = [game[0], game[1], game[2]]
    # probs1 = clf1.predict_proba([game])[0]
    # ratios1 = [1/probs1[2], 1/probs1[1], 1/probs1[0]]
    # dRatios1 = [bet365[0] - ratios1[0], bet365[1] - ratios1[1], bet365[2] - ratios1[2]]
    probs2 = clf2.predict_proba([game])[0]
    ratios2 = [1 / probs2[2], 1 / probs2[1], 1 / probs2[0]]
    dRatios2 = [bet365[0] - ratios2[0], bet365[1] - ratios2[1], bet365[2] - ratios2[2]]
    distance1 = abs(4.902 - check_test[i])
    distance2 = abs(4.570 - check_test[i])
    if dRatios2[0] > 0.55 and distance2 - distance1 < -0.2:
        count += 1
        if y_test[i] != 'H':
            value -= 1
            print("no", bet365)
        else:
            value += (bet365[0] - 1)
            print("yes", bet365)
        sum += (value / count)
        print("current ROI", value / count)
        print("profit", value)
        print("num bets", count)
        print("avg ROI", sum / count)
    # if dRatios1[2] > 3.5:
        # count += 1
        # if y_test[i] != 'A':
            # value -= 1
        # else:`
            # value += (bet365[2] - 1)
        # print(value / count)
        # print(value)
        # print(count)
