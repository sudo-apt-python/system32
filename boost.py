!pip install -q hvplot
import pandas as pd
import numpy as np
from time import time
import hvplot.pandas
import matplotlib.pyplot as plt


data = pd.read_csv('data.csv')
data.loc[data.grad_rate > 100, 'grad_rate'] = 100
accuracy = {}
speed = {}
X = data.drop('private', axis=1)
y = data.private.map({"Yes": 1, "No": 0})
data.private.value_counts()

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
model = GradientBoostingClassifier()

start = time()
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
score = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1)
speed['GradientBoosting'] = np.round(time() - start, 3)
accuracy['GradientBoosting'] = (np.mean(score) * 100).round(3)

print(f"Mean F1 score: {accuracy['GradientBoosting']}")
print(f"STD: {np.std(score):.3f}")
print(f"Run Time: {speed['GradientBoosting']}s")


from xgboost import XGBClassifier
model = XGBClassifier()
start = time()
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
score = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1)
speed['XGBoost'] = np.round(time() - start, 3)
accuracy['XGBoost'] = (np.mean(score) * 100).round(3)
print(f"Mean F1 score: {accuracy['XGBoost']}")
print(f"STD: {np.std(score):.3f}")
print(f"Run Time: {speed['XGBoost']}s")

!pip install catboost
from catboost import CatBoostClassifier
model = CatBoostClassifier(silent=True)
start = time()
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
score = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1)
speed['CatBoost'] = np.round(time() - start, 3)
accuracy['CatBoost'] = (np.mean(score) * 100).round(3)
print(f"Mean F1 score: {accuracy['CatBoost']}")
print(f"STD: {np.std(score):.3f}")
print(f"Run Time: {speed['CatBoost']}s")
#ada
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
start = time()
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
score = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1)
speed['AdaBoost'] = np.round(time() - start, 3)
accuracy['AdaBoost'] = (np.mean(score) * 100).round(3)
print(f"Mean F1 score: {accuracy['AdaBoost']}")
print(f"STD: {np.std(score):.3f}")
print(f"Run Time: {speed['AdaBoost']}s")
for algo, result in accuracy.items():
    print(f"{algo:{20}}: Score: {result}, Speed: {speed[algo]}")
# Visualize Comparative Analysis Results
# Bar plot for Accuracy
plt.figure(figsize=(12, 6))
plt.bar(accuracy.keys(), accuracy.values(), color=['blue', 'orange', 'green',
'red'])
plt.title('Accuracy Comparison of Ensemble Techniques')
plt.xlabel('Ensemble Techniques')
plt.ylabel('Mean F1 Score (%)')
plt.ylim(0, 100)
plt.show()
# Bar plot for Speed
plt.figure(figsize=(12, 6))
plt.bar(speed.keys(), speed.values(), color=['blue', 'orange', 'green', 'red'])
plt.title('Speed Comparison of Ensemble Techniques')
plt.xlabel('Ensemble Techniques')
plt.ylabel('Run Time (s)')
plt.show()


