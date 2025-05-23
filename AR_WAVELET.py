# -*- coding: utf-8 -*-
"""
Pipeline de classification PPG MI en combinant coefficients AR et coefficients ondelette

@author: callu
"""

import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
import random

from statsmodels.tsa.ar_model import AutoReg
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# --- Chargement des données ---
df = pd.read_csv("PPG_Dataset.csv")
df.columns = df.columns.astype(str)

# --- 1) Visualisation : signal réel vs prédiction AR en 2 sous-graphes ---
p = 13
idx = random.randint(0, len(df) - 1)
signal = df.iloc[idx, :-1].astype(float).values
label = df.iloc[idx]['Label']

ar_model = AutoReg(signal, lags=p, old_names=False)
ar_fit = ar_model.fit()
pred = ar_fit.predict(start=p, end=len(signal)-1)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
axes[0].plot(signal, label="Signal d'origine", linewidth=1)
axes[0].set_title(f"Vrai signal (Index {idx}) - Label: {label}")
axes[0].set_ylabel("Amplitude")
axes[0].legend(loc='upper right')
axes[1].plot(range(p, len(signal)), pred, label="Signal Prédit (AR)", linewidth=1, color='darkorange')
axes[1].set_title("Prédiction Auto-Régressive")
axes[1].set_xlabel("Échantillon")
axes[1].set_ylabel("Amplitude")
axes[1].legend(loc='upper right')
plt.tight_layout(); plt.show()

# --- 2) Histogramme des coefficients AR ---
ar_coefs = ar_fit.params[1:]
plt.figure(figsize=(8, 4))
plt.bar(np.arange(1, p+1), ar_coefs, color='mediumorchid')
plt.xlabel("Lag")
plt.ylabel("Coefficient AR")
plt.title("Histogramme des coefficients AR")
plt.tight_layout(); plt.show()

# --- 3) Fonctions ondelette mère (db4) ---
wavelet = pywt.Wavelet('db4')
phi, psi, x = wavelet.wavefun(level=5)
plt.figure(figsize=(8, 4)); plt.plot(x, phi, color='red')
plt.title("Fonction d'échelle (phi) db4"); plt.xlabel("Temps"); plt.ylabel("Amplitude")
plt.tight_layout(); plt.show()
plt.figure(figsize=(8, 4)); plt.plot(x, psi, color='red')
plt.title("Fonction ondelette mère (psi) db4"); plt.xlabel("Temps"); plt.ylabel("Amplitude")
plt.tight_layout(); plt.show()

# --- Extraction des features AR + Wavelet ---
def extract_ar_coeffs(sig, p=13):
    fit = AutoReg(sig, lags=p, old_names=False).fit()
    return fit.params[1:].tolist()

def extract_wavelet_features(sig, wavelet='db4', level=5):
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    feats = []
    for c in coeffs:
        feats += [np.mean(c), np.std(c), np.max(c), np.min(c), np.sum(c**2)]
    return feats

X, y = [], []
for _, row in df.iterrows():
    sig = row[:-1].astype(float).values
    lab = 1 if row['Label']=='MI' else 0
    X.append(extract_ar_coeffs(sig) + extract_wavelet_features(sig))
    y.append(lab)
X = np.array(X); y = np.array(y)
print(f"Dimensions des features : {X.shape[1]} (AR + Wavelet)")

# --- Préparation et split ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# --- Recherche hyperparamètres SVM RBF ---
param_grid = {'C':[0.01,0.1,1,10,100], 'gamma':[1e-4,1e-3,1e-2,1e-1,1]}
svm = SVC(kernel='rbf', probability=True)
grid = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
best_svm = grid.best_estimator_
print("Meilleurs paramètres SVM:", grid.best_params_)

# --- Évaluation ---
y_pred = best_svm.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Normal','MI']))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal','MI'], yticklabels=['Normal','MI'])
plt.title('Matrice de confusion - SVM AR+Wavelet')
plt.tight_layout(); plt.show()

# --- Courbe ROC & AUC ---
y_score = best_svm.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.2f})', color='darkred')
plt.plot([0,1], [0,1], linestyle='--', lw=1, color='gray')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Courbe ROC - SVM AR+Wavelet')
plt.legend(loc='lower right'); plt.tight_layout(); plt.show()
