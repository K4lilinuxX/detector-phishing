import joblib
import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. CARGAR TRAIN Y TEST
# ============================================================

train_path = "data/processed/train.csv"
test_path = "data/processed/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("=" * 60)
print("ENTRENAMIENTO DEL MODELO")
print("=" * 60)

print(f"Registros Train: {len(train_df):,}")
print(f"Registros Test:  {len(test_df):,}")


# ============================================================
# 2. SEPARAR TEXTO Y ETIQUETA
# ============================================================

X_train_text = train_df["Email Text"]
y_train = train_df["label"]

X_test_text = test_df["Email Text"]
y_test = test_df["label"]


# ============================================================
# 3. CREAR VECTORIZADOR TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode"
)


# ============================================================
# 4. TRANSFORMAR TRAIN
# ============================================================

print("\nTransformando Train con TF-IDF...")

X_train = vectorizer.fit_transform(
    X_train_text
)


# ============================================================
# 5. TRANSFORMAR TEST
# ============================================================

print("Transformando Test con TF-IDF...")

X_test = vectorizer.transform(
    X_test_text
)


# ============================================================
# 6. INFORMACIÓN DE TF-IDF
# ============================================================

print("\n" + "=" * 60)
print("MATRICES TF-IDF")
print("=" * 60)

print(f"Train: {X_train.shape}")
print(f"Test:  {X_test.shape}")

print(
    f"Características aprendidas: "
    f"{len(vectorizer.vocabulary_):,}"
)


# ============================================================
# 7. CREAR MODELO
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# ============================================================
# 8. ENTRENAR
# ============================================================

print("\nEntrenando Logistic Regression...")

model.fit(
    X_train,
    y_train
)

print("✓ Modelo entrenado.")

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/phishing_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\n✓ Modelo guardado en: models/phishing_model.pkl")
print("✓ Vectorizador guardado en: models/tfidf_vectorizer.pkl")


# ============================================================
# 9. PREDICCIONES
# ============================================================

print("\nGenerando predicciones...")

y_pred = model.predict(
    X_test
)


# ============================================================
# 10. MÉTRICAS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

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


# ============================================================
# 11. MOSTRAR MÉTRICAS
# ============================================================

print("\n" + "=" * 60)
print("RESULTADOS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
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
# 13. MATRIZ DE CONFUSIÓN
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("MATRIZ DE CONFUSIÓN")
print("=" * 60)

print(cm)

print("\nInterpretación:")
print("                Predicho")
print("              Safe  Phishing")
print(f"Real Safe     {cm[0, 0]:4}  {cm[0, 1]:8}")
print(f"Real Phishing {cm[1, 0]:4}  {cm[1, 1]:8}")