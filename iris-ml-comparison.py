import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import shap
import seaborn as sns
import platform
import psutil

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier

# Sistem bilgisi
print(f"CPU: {platform.processor()}, Physical cores: {psutil.cpu_count(logical=False)}, Logical cores: {psutil.cpu_count(logical=True)}")

# 📥 Veri setini yükle
iris = load_iris()
X, y = iris.data, iris.target

# label binarize ROC AUC için
y_bin = label_binarize(y, classes=np.unique(y))
n_classes = y_bin.shape[1]

# 📚 Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ⚙️ Modeller ve parametreler (geliştirilmiş parametre aralıkları, pipeline kullanımı)
models_params = {
    "K-Nearest Neighbors": {
        "model": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "params": {
            "kneighborsclassifier__n_neighbors": [3, 5, 7, 9],
            "kneighborsclassifier__weights": ["uniform", "distance"]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5]
        }
    },
    "Support Vector Machine": {
        "model": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
        "params": {
            "svc__C": [0.1, 1, 10],
            "svc__kernel": ["linear", "rbf"],
            "svc__gamma": ["scale", "auto"]
        }
    },
    "Neural Network": {
        "model": make_pipeline(StandardScaler(), MLPClassifier(max_iter=3000, random_state=42)),
        "params": {
            "mlpclassifier__hidden_layer_sizes": [(50,), (100,), (150,)],
            "mlpclassifier__activation": ["relu", "tanh"],
            "mlpclassifier__solver": ["adam"],
            "mlpclassifier__alpha": [0.0001, 0.001, 0.01]
        }
    }
}

best_models = {}
metrics_summary = {}
train_times = {}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, mp in models_params.items():
    print(f"🔍 {name} için GridSearch başlatılıyor...")
    start = time.time()
    grid = GridSearchCV(mp["model"], mp["params"], cv=cv, n_jobs=-1, scoring='accuracy')
    grid.fit(X_train, y_train)
    duration = time.time() - start

    best_model = grid.best_estimator_
    best_models[name] = best_model
    train_times[name] = duration

    # Tahmin ve metrikler
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Cross-val skorlar (accuracy)
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

    metrics_summary[name] = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "train_time": duration,
        "best_params": grid.best_params_,
        "cv_mean_acc": cv_scores.mean(),
        "cv_std_acc": cv_scores.std()
    }

    print(f"✅ {name} tamamlandı.")
    print(f"    - Test Set Doğruluk: {acc:.4f}")
    print(f"    - Ortalama CV Doğruluk: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    - Eğitim Süresi: {duration:.2f} saniye")
    print(f"    - En iyi parametreler: {grid.best_params_}")
    print("")

# 📊 Accuracy ve F1 Score grafiği
model_names = list(metrics_summary.keys())
accuracy_vals = [metrics_summary[m]["accuracy"] for m in model_names]
f1_vals = [metrics_summary[m]["f1_score"] for m in model_names]

plt.figure(figsize=(10,6))
plt.bar(np.arange(len(model_names)), accuracy_vals, width=0.4, label="Accuracy")
plt.bar(np.arange(len(model_names)) + 0.4, f1_vals, width=0.4, label="F1 Score")
plt.xticks(np.arange(len(model_names)) + 0.2, model_names, rotation=15)
plt.ylim(0.8, 1.05)
plt.ylabel("Skor")
plt.title("Accuracy vs F1 Score")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 🔎 En iyi model için confusion matrix
best_model_name = max(metrics_summary, key=lambda x: metrics_summary[x]['accuracy'])
best_model = best_models[best_model_name]
y_pred_best = best_model.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred_best)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title(f"{best_model_name} - Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.tight_layout()
plt.show()


# 🧠 SHAP analizi

def get_final_estimator(model):
    if hasattr(model, "named_steps"):
        last_step_name = list(model.named_steps.keys())[-1]
        return model.named_steps[last_step_name]
    else:
        return model

def shap_analysis(model, model_name, X_train_data, X_test_data):
    print(f"{model_name} için SHAP analizi başlıyor...")
    final_model = get_final_estimator(model)

    try:
        if isinstance(final_model, RandomForestClassifier):
            explainer = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(X_test_data)
            shap.summary_plot(shap_values, X_test_data, feature_names=iris.feature_names, show=False)
            plt.title(f"{model_name} - SHAP summary plot")
            plt.tight_layout()
            plt.show()
        else:
            # KernelExplainer ağırdır, örneklem sınırla
            background = shap.sample(X_train_data, 50, random_state=42)
            explainer = shap.KernelExplainer(final_model.predict_proba, background)
            shap_values = explainer.shap_values(X_test_data[:20])
            shap.summary_plot(shap_values, X_test_data[:20], feature_names=iris.feature_names, show=False)
            plt.title(f"{model_name} - SHAP summary plot (KernelExplainer, 20 örneklem)")
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"SHAP analizi yapılamadı: {model_name} - Hata: {e}")

X_train_df = pd.DataFrame(X_train, columns=iris.feature_names)
X_test_df = pd.DataFrame(X_test, columns=iris.feature_names)

for name, mdl in best_models.items():
    shap_analysis(mdl, name, X_train_df, X_test_df)


# ROC AUC (Multiclass) - Random Forest için

if "Random Forest" in best_models:
    print("🔷 Random Forest ROC AUC (one-vs-rest) grafiği oluşturuluyor...")
    rf_model = best_models["Random Forest"]
    y_score = None

    if hasattr(rf_model, "predict_proba"):
        y_score = rf_model.predict_proba(X_test)
    elif hasattr(rf_model, "named_steps"):
        last_step = list(rf_model.named_steps.keys())[-1]
        if hasattr(rf_model.named_steps[last_step], "predict_proba"):
            y_score = rf_model.named_steps[last_step].predict_proba(X_test)

    if y_score is not None:
        y_test_bin = label_binarize(y_test, classes=np.unique(y))
        for i in range(n_classes):
            RocCurveDisplay.from_predictions(y_test_bin[:, i], y_score[:, i], name=f"Class {iris.target_names[i]}")
        plt.title(f"{best_model_name} - ROC Curve")
        plt.tight_layout()
        plt.show()
    else:
        print("ROC AUC için predict_proba desteklenmiyor.")


