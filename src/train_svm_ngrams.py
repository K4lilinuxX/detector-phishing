import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. CARGAR DATOS
# ============================================================

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

X_train_text = train_df["Email Text"].fillna("").astype(str)
X_test_text = test_df["Email Text"].fillna("").astype(str)

y_train = train_df["label"]
y_test = test_df["label"]

print(f"Registros Train: {len(train_df)}")
print(f"Registros Test:  {len(test_df)}")


# ============================================================
# 2. TF-IDF CON WORD N-GRAMS
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    stop_words="english",
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print(f"\nTF-IDF Train: {X_train.shape}")
print(f"TF-IDF Test:  {X_test.shape}")

print(f"Características aprendidas: {X_train.shape[1]}")
print("Configuración: unigramas + bigramas")


# ============================================================
# 3. ENTRENAR LINEAR SVM
# ============================================================

model = LinearSVC(
    random_state=42,
    max_iter=5000
)

print("\nEntrenando SVM...")

model.fit(
    X_train,
    y_train
)

print("✓ Entrenamiento terminado.")


# ============================================================
# 4. PREDICCIONES
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 5. MÉTRICAS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# 6. RESULTADOS
# ============================================================

print("\n" + "=" * 50)
print("RESULTADOS — SVM + WORD N-GRAMS")
print("=" * 50)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")


# ============================================================
# 7. MATRIZ DE CONFUSIÓN
# ============================================================

print("\nMATRIZ DE CONFUSIÓN")

print(cm)


# ============================================================
# 8. DETALLES
# ============================================================

print("\nDETALLES")

print(f"True Negatives : {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives : {tp}")

print(f"\nTotal errores  : {fp + fn}")
print(f"Total correctos: {tn + tp}")


# ============================================================
# 9. GUARDAR MODELO
# ============================================================

joblib.dump(
    model,
    "models/svm_ngrams_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer_svm_ngrams.pkl"
)


# ============================================================
# 10. CONFIRMACIÓN
# ============================================================

print("\n✓ Modelo guardado en:")
print("  models/svm_ngrams_model.pkl")

print("\n✓ Vectorizador guardado en:")
print("  models/tfidf_vectorizer_svm_ngrams.pkl")