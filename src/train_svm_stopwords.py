import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

train_path = "data/processed/train.csv"
test_path = "data/processed/test.csv"

model_path = "models/svm_stopwords_model.pkl"
vectorizer_path = "models/tfidf_vectorizer_svm_stopwords.pkl"


# ============================================================
# CARGAR DATOS
# ============================================================

print("=" * 60)
print("EXPERIMENTO B — LINEAR SVM + STOP WORDS")
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
# TF-IDF CON STOP WORDS
# ============================================================

print("\nTransformando textos con TF-IDF...")
print("Eliminando stop words en inglés...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    stop_words="english"
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print(f"Train: {X_train.shape}")
print(f"Test:  {X_test.shape}")
print(
    f"Características aprendidas: "
    f"{len(vectorizer.get_feature_names_out()):,}"
)


# ============================================================
# ENTRENAR SVM
# ============================================================

print("\nEntrenando Linear SVM...")

model = LinearSVC(
    random_state=42
)

model.fit(X_train, y_train)

print("✓ Modelo entrenado.")


# ============================================================
# GUARDAR MODELO
# ============================================================

os.makedirs("models", exist_ok=True)

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"\n✓ Modelo guardado en: {model_path}")
print(f"✓ Vectorizador guardado en: {vectorizer_path}")


# ============================================================
# PREDICCIONES
# ============================================================

print("\nGenerando predicciones...")

y_pred = model.predict(X_test)


# ============================================================
# MÉTRICAS
# ============================================================

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


print("\n")
print("=" * 60)
print("RESULTADOS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Safe Email",
            "Phishing Email"
        ],
        zero_division=0
    )
)


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\n")
print("=" * 60)
print("MATRIZ DE CONFUSIÓN")
print("=" * 60)

print(cm)

print("\nInterpretación:")
print("                Predicho")
print("              Safe  Phishing")
print(f"Real Safe     {tn:4d}     {fp:4d}")
print(f"Real Phishing {fn:4d}     {tp:4d}")


# ============================================================
# RESUMEN
# ============================================================

print("\n")
print("=" * 60)
print("ANÁLISIS DE ERRORES")
print("=" * 60)

print(f"True Negatives  (Safe → Safe):         {tn}")
print(f"False Positives (Safe → Phishing):     {fp}")
print(f"False Negatives (Phishing → Safe):     {fn}")
print(f"True Positives  (Phishing → Phishing): {tp}")

print(f"\nTotal de errores: {fp + fn}")
print(f"Total correctas:  {tn + tp}")