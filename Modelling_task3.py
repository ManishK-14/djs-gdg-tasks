import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

f1_comp = pd.read_csv('f1_dnf.csv',index_col=0)

print(f1_comp.describe())

f1_comp.replace('/N',np.nan,inplace=True)

f1_comp['points'] = pd.to_numeric(f1_comp['points'], errors='coerce')
f1_comp['laps'] = pd.to_numeric(f1_comp['laps'], errors='coerce')
f1_comp['milliseconds'] = pd.to_numeric(f1_comp['milliseconds'], errors='coerce')
f1_comp['fastestLapSpeed'] = pd.to_numeric(f1_comp['fastestLapSpeed'], errors='coerce')

print(f1_comp.isnull().sum())
f1_comp['points'] = f1_comp['points'].fillna(f1_comp['points'].mean())
f1_comp['laps'] = f1_comp['laps'].fillna(f1_comp['laps'].mode()[0])
f1_comp['milliseconds'] = f1_comp['milliseconds'].fillna(f1_comp['milliseconds'].median())
f1_comp['fastestLapSpeed'] = f1_comp['fastestLapSpeed'].fillna(f1_comp['fastestLapSpeed'].mean())

print('-----------------')
print("After Handling Missing Values ")
print(f1_comp.isnull().sum())

print(f1_comp.dtypes)

#Histogram 
plt.figure(figsize=(8,4))
plt.hist(f1_comp['points'], bins=30, alpha=0.6, color='skyblue', edgecolor='black')
sns.kdeplot(f1_comp['points'], color='red', linewidth=2)
plt.title("Distribution of Points")
plt.xlabel("Points")
plt.ylabel("Count")
plt.show()

#Boxplott
sns.boxplot(x=f1_comp['milliseconds'], color='lightgreen')
plt.title("Boxplot of Race Completion Time (ms)")
plt.show()

#Checking position and the points
plt.scatter(f1_comp['positionOrder'], f1_comp['points'], color='orange')
plt.title("Position vs Points")
plt.xlabel("Position Order")
plt.ylabel("Points")
plt.show()

points_by_year = f1_comp.groupby('year')['points'].mean()
plt.plot(points_by_year.index, points_by_year.values, marker='o', color='red')
plt.title("Average Points per Year")
plt.xlabel("Year")
plt.ylabel("Average Points")
plt.show()

#Top drivers
top_drivers = f1_comp.groupby('surname')['points'].sum()
top_10_drivers = top_drivers.sort_values(ascending=False).head(10)
sns.barplot(x=top_10_drivers.values, y=top_10_drivers.index, palette='cool')
plt.title("Top 10 Drivers by Total Points")
plt.xlabel("Total Points")
plt.ylabel("Driver")
plt.show()

#Finishing the target 
sns.countplot(x='target_finish', data=f1_comp, palette='pastel')
plt.title("Target Finish Distribution")
plt.show()

#Fastest LAp speed distribution 
sns.boxplot(x='fastestLapSpeed', data=f1_comp, color='yellow')
plt.title("Distribution of Fastest Lap Speed")
plt.show()

#Nationality of top drivers
top_countries = f1_comp['nationality_x'].value_counts().head(10)
sns.barplot(x=top_countries.values, y=top_countries.index, palette='pastel')
plt.title("Top 10 Driver Nationalities")
plt.xlabel("Number of Drivers")
plt.show()


##Modelling 

#USing Lr to predict the finish line (0 or  1)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
#Selecting the relevant feature 
features = ['year', 'laps', 'points', 'fastestLapSpeed', 'lat', 'lng', 'alt', 'grid']
X = f1_comp[features]
y = f1_comp['target_finish']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Feature scaling
numeric_features = ['year', 'laps', 'points', 'fastestLapSpeed', 'lat', 'lng', 'alt', 'grid']
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

#Training model
model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

#Prediction
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:,1]

#perfomance metrics
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

print("Classification Report:\n", classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_prob)
print("ROC-AUC Score:", roc_auc)
