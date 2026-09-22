import time

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

train_path = "data/processed/train.csv"
test_path = "data/processed/test.csv"


# ============================================================
# CARGAR DATOS
# ============================================================

print("=" * 60)
print("COMPARACIÓN DE MODELOS")
print("=" * 60)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train_text = train_df["Email Text"]
y_train = train_df["label"]

X_test_text = test_df["Email Text"]
y_test = test_df["label"]

print(f"\nRegistros Train: {len(train_df):,}")
print(f"Registros Test:  {len(test_df):,}")


# ============================================================
# TF-IDF
# ============================================================

print("\nTransformando textos con TF-IDF...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode"
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print(f"Train: {X_train.shape}")
print(f"Test:  {X_test.shape}")


# ============================================================
# MODELOS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Multinomial Naive Bayes": MultinomialNB(),

    "Linear SVM": LinearSVC(
        random_state=42
    )
}


# ============================================================
# ENTRENAMIENTO Y EVALUACIÓN
# ============================================================

results = []

for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"MODELO: {name}")
    print("=" * 60)

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    results.append({
        "Modelo": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Tiempo (s)": training_time
    })

    print(f"Tiempo entrenamiento: {training_time:.4f} s")
    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"Recall:               {recall:.4f}")
    print(f"F1-Score:             {f1:.4f}")


# ============================================================
# TABLA FINAL
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 80)
print("COMPARACIÓN FINAL")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1-Score": "{:.4f}".format,
            "Tiempo (s)": "{:.4f}".format
        }
    )
)