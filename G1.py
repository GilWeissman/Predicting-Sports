import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier

def filter_data1(data):
    filter_arr = []
    for game in data:
        if game[9] > 0 and game[15] > 0:
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
    selections3 = np.array([False, False, False, False, False, False, False, False, False, False, False, False, False,
                            False, False, True])
    y1 = data[:, selections2]
    y3 = data[:, selections3]
    y2 = []
    for y in y1:
        y2.append(y[0])
    y4 = []
    for y in y3:
        y4.append(y[0])
    return data[:, selections1], y2, y4


adress_train_G1 = 'trainingSetG1.csv'
adress_test_G12019 = 'testSetG12019.csv'
adress_current_test_F1 = 'currentTestSetPL.csv'
names1 = ['', 'date', 'HomeTeam', 'GoalsFH', 'GoalsAH', 'ShootsTargetFH', 'ShootsTargetAH', 'B365H', 'B365D', 'B365A',
          'FTR', 'Venue', 'ptsH', 'CT', 'Count', 'RDate', 'Games', 'GlspgaH', 'GlspgfH', 'STpgaH', 'STpgfH', 'PtspgH',
          'CumptsH', 'CumG', 'CumglsAH', 'CumglsfH', 'CumsTAH', 'CumsTfH', 'date', 'AwayTeam', 'GoalsFA',
          'GoalsAA', 'ShootsTargetFA', 'ShootsTargetAA', 'B365H', 'B365D', 'B365A', 'FTR', 'Venue', 'ptsA', 'CT',
          'Count', 'RDate', 'Games', 'GlspgaA', 'GlspgfA', 'STpgaA', 'STpgfA', 'PtspgA', 'CumptsA', 'CumG',
          'CumglsAA', 'CumglsfA', 'CumsTAA', 'CumsTfA', 'year']
trainingSet = pd.read_csv(adress_train_G1, header=0)
testSet = pd.read_csv(adress_test_G12019, header=0)
train = trainingSet.iloc[:, [10, 7, 8, 9, 17, 18, 19, 20, 21, 23, 44, 45, 46, 47, 48, 50]].values
X_train, y_train, check_train = filter_data1(train)
test = testSet.iloc[:, [10, 7, 8, 9, 17, 18, 19, 20, 21, 23, 44, 45, 46, 47, 48, 50]].values
X_test, y_test, check_test = filter_data1(test)

clf1 = LogisticRegression(max_iter=100000, class_weight={'H': 0.6, 'D': 0.3, 'A': 0.1}).fit(X_train, y_train)
clf2 = LogisticRegression(max_iter=100000, class_weight={'H': 0.25, 'D': 0.5, 'A': 0.25}).fit(X_train, y_train)
clf3 = LogisticRegression(max_iter=100000, class_weight={'H': 0.3, 'D': 0.3, 'A': 0.4}).fit(X_train, y_train)

value = 0
count = 0
count2 = 0
sum = 0
for i in range(len(X_test)):
    game = X_test[i]
    bet365 = [game[0], game[1], game[2]]
    probs1 = clf1.predict_proba([game])[0]
    ratios1 = [1 / probs1[2], 1 / probs1[1], 1 / probs1[0]]
    dRatios1 = [bet365[0] - ratios1[0], bet365[1] - ratios1[1], bet365[2] - ratios1[2]]
    if 6 < dRatios1[0] < 6.8 and bet365[0] > 5:
        count += 1
        if y_test[i] != 'H':
            print("no", bet365)
            value -= 1
        else:
            value += (bet365[0] - 1)
            print("yes", bet365)
        sum += (value / count)
        print("current ROI", value / count)
        print("profit", value)
        print("num bets", count)
        print("avg ROI", sum / count)

new_game = [1.44, 4.75, 6.5]
new_bet365 = [new_game[0], new_game[1], new_game[2]]
probs1 = clf1.predict_proba([new_game])[0]
ratios1 = [1 / probs1[2], 1 / probs1[1], 1 / probs1[0]]
dRatios1 = [new_bet365[0] - ratios1[0], new_bet365[1] - ratios1[1], new_bet365[2] - ratios1[2]]
probs2 = clf2.predict_proba([new_game])[0]
ratios2 = [1 / probs2[2], 1 / probs2[1], 1 / probs2[0]]
dRatios2 = [new_bet365[0] - ratios2[0], new_bet365[1] - ratios2[1], new_bet365[2] - ratios2[2]]
probs3 = clf3.predict_proba([new_game])[0]
ratios3 = [1 / probs3[2], 1 / probs3[1], 1 / probs3[0]]
dRatios3 = [new_bet365[0] - ratios3[0], new_bet365[1] - ratios3[1], new_bet365[2] - ratios3[2]]
if 6 < dRatios1[0] < 6.8 and new_bet365[0] > 5:
    print("bet H")
if 1.75 < dRatios2[1] < 2.1 and new_bet365[1] > 4.75:
    print("Bet D")
if 1.05 < dRatios3[2] < 1.1 and new_bet365[2] > 4:
    print("Bet A")

