import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

adress_ds1 = 'ds1.csv'
adress_ds2 = 'ds2.csv'
names1 = ['avgH', 'avgA', 'OvUn2.5']
names2 = ['avgH', 'avgA', 'PDH', 'PDA', 'OvUn2.5']
dataset = pd.read_csv(adress_ds1, names=names1, header=0)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 2].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

classifier1 = LogisticRegression(random_state=0).fit(X_train, y_train)
classifier2 = KNeighborsClassifier(n_neighbors=15).fit(X_train, y_train)

predicted_y = classifier1.predict(X_test)
print(classification_report(y_test, predicted_y))
print(confusion_matrix(y_test, predicted_y))

