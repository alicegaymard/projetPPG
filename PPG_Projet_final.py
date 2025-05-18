# -*- coding: utf-8 -*-
"""
Created on Sun May 18 23:20:10 2025

@author: callu
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, entropy
from sklearn.metrics import classification_report

df = pd.read_csv("PPG_Dataset.csv")
df.head()

df.columns = df.columns.astype(str)


example_signal = df.iloc[0, :-1].astype(float).values  
label = df.iloc[0, -1]  # 'MI' or not

# On fit un modèle d'auto régression d'ordre 10 
p = 10
model = AutoReg(example_signal, lags=p, old_names=False)
model_fit = model.fit()
predicted_signal = model_fit.predict(start=p, end=len(example_signal)-1)

# Vrai signal pour comparer aux prédictions
true_signal_segment = example_signal[p:]

# Calcul du bruit
noise = true_signal_segment - predicted_signal

# Afficher un signal aléatoire et ses prédictions AR
random_idx = np.random.randint(len(df))
example_signal = df.iloc[random_idx, :-1].astype(float).values
label = df.iloc[random_idx, -1]

model = AutoReg(example_signal, lags=p, old_names=False)
model_fit = model.fit()
predicted_signal = model_fit.predict(start=p, end=len(example_signal)-1)
true_signal_segment = example_signal[p:]

fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axs[0].plot(true_signal_segment, label="Signal d'origine")
axs[0].set_title(f'Vrai signal (Index {random_idx}) - Label: {label}')
axs[0].legend()

axs[1].plot(predicted_signal, label='Signal Prédit (AR)', color='orange')
axs[1].set_title('Prédiction Auto-Régressive')
axs[1].legend()

plt.tight_layout()
plt.show()

# Affichage des coefficients AR
ar_coeffs = model_fit.params[1:]  # on ignore l'intercept (premier terme a_0)
lags = np.arange(1, len(ar_coeffs) + 1)

plt.figure(figsize=(8, 4))
plt.bar(lags, ar_coeffs, color='purple')
plt.xlabel('Lag')
plt.ylabel('Coefficient')
plt.title(f'Coefficients AR (p = {p}) pour le signal index {random_idx}')
plt.grid(True)
plt.tight_layout()
plt.show()


# Fonction pour extraire des features à partir du bruit entre signal réel et signal AR
def extract_features(signal, p=10, use_ar_coeffs=True):
    model = AutoReg(signal, lags=p, old_names=False)
    model_fit = model.fit()
    predicted = model_fit.predict(start=p, end=len(signal)-1)
    true_segment = signal[p:]
    noise = true_segment - predicted

    if use_ar_coeffs:
        ar_coeffs = model_fit.params[1:]  # on enlève l’intercept
        return ar_coeffs.tolist()
    else:
        energy = np.sum(noise**2)
        std_dev = np.std(noise)
        skewness = skew(noise)
        ent = entropy(np.histogram(noise, bins=30, density=True)[0] + 1e-6)
        return [energy, std_dev, skewness, ent]


# Préparation des features et labels pour tout le dataset
X = []
y = []

for idx, row in df.iterrows():
    signal = row[:-1].astype(float).values
    label = 1 if row['Label'] == 'MI' else 0
    features = extract_features(signal, p=10, use_ar_coeffs=True)  # <- True pour utiliser les coeffs AR
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

from sklearn.model_selection import GridSearchCV

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Définir les valeurs possibles de C à tester
param_grid = {'C': [0.01, 0.1, 1, 10, 100],'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]}

svm = SVC(kernel='rbf')
grid = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Meilleurs hyperparamètres :", grid.best_params_)

# SVM + GridSearch
svm = SVC(kernel='rbf', gamma='scale')
grid = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# Meilleur modèle
best_svm = grid.best_estimator_
print("Meilleur C trouvé :", grid.best_params_)

# Évaluation
y_pred = best_svm.predict(X_test)
report = classification_report(y_test, y_pred, target_names=['Non-MI', 'MI'])
print(report)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

models = {
    "Régression Logistique": LogisticRegression(max_iter=1000),
    "Arbre de Décision": DecisionTreeClassifier(),
    "Forêt Aléatoire": RandomForestClassifier(n_estimators=100)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n==== {name} ====")
    print(classification_report(y_test, y_pred, target_names=['Non-MI', 'MI']))


# Nouveau rapport au format tableau
from sklearn.metrics import classification_report
import pandas as pd

report_dict = classification_report(y_test, y_pred, target_names=['Non-MI', 'MI'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

# Arrondir les valeurs à 3 décimales
report_df = report_df.round(3)

# Afficher le tableau
print("Rapport de classification (SVM optimisé avec GridSearch) :")
print(report_df)
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calcul de la matrice de confusion
cm = confusion_matrix(y_test, y_pred)

# Affichage graphique
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-MI', 'MI'],
            yticklabels=['Non-MI', 'MI'])
plt.xlabel('Prédit')
plt.ylabel('Vrai')
plt.title('Matrice de confusion - SVM optimisé')
plt.tight_layout()
plt.show()

# Calcul de la matrice de corrélation pour les features
feature_names = [f'AR({i+1})' for i in range(X.shape[1])]
correlation_matrix = pd.DataFrame(X, columns=feature_names).corr()

# Affichage graphique de la matrice de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True,
            xticklabels=feature_names,
            yticklabels=feature_names)
plt.title('Matrice de Corrélation des Coefficients AR')
plt.tight_layout()
plt.show()