import pandas as pd
import numpy as np
import re
import joblib

from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
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
# 2. TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

print(f"\nTF-IDF Train: {X_train_tfidf.shape}")
print(f"TF-IDF Test:  {X_test_tfidf.shape}")


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def extract_features(text):
    """
    Extrae características estructurales de un correo.
    """

    text = str(text)

    # --------------------------------------------------------
    # URLs
    # --------------------------------------------------------

    urls = re.findall(
        r"https?://\S+|www\.\S+",
        text,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # Palabras
    # --------------------------------------------------------

    words = re.findall(r"\b\w+\b", text)

    # --------------------------------------------------------
    # Texto en minúsculas
    # --------------------------------------------------------

    text_lower = text.lower()

    # --------------------------------------------------------
    # Lenguaje de urgencia
    # --------------------------------------------------------

    urgent_words = [
        "urgent",
        "immediately",
        "urgent action",
        "act now",
        "asap",
        "within 24 hours",
        "suspended",
        "expire",
        "expired",
        "warning"
    ]

    has_urgent = int(
        any(phrase in text_lower for phrase in urgent_words)
    )

    # --------------------------------------------------------
    # Lenguaje relacionado con credenciales
    # --------------------------------------------------------

    credential_words = [
        "password",
        "passwd",
        "login",
        "username",
        "credentials",
        "verify your identity",
        "verification",
        "confirm your password"
    ]

    has_credentials = int(
        any(phrase in text_lower for phrase in credential_words)
    )

    # --------------------------------------------------------
    # Longitud del correo
    # --------------------------------------------------------

    email_length = len(text)

    # --------------------------------------------------------
    # Número de palabras
    # --------------------------------------------------------

    word_count = len(words)

    # --------------------------------------------------------
    # Número de URLs
    # --------------------------------------------------------

    url_count = len(urls)

    # --------------------------------------------------------
    # Presencia de URL
    # --------------------------------------------------------

    has_url = int(url_count > 0)

    # --------------------------------------------------------
    # Símbolos
    # --------------------------------------------------------

    exclamation_count = text.count("!")
    question_count = text.count("?")
    dollar_count = text.count("$")

    # --------------------------------------------------------
    # Número de dígitos
    # --------------------------------------------------------

    digit_count = sum(
        char.isdigit()
        for char in text
    )

    # --------------------------------------------------------
    # Proporción de mayúsculas
    # --------------------------------------------------------

    letters = [
        char
        for char in text
        if char.isalpha()
    ]

    if letters:
        uppercase_ratio = (
            sum(char.isupper() for char in letters)
            / len(letters)
        )
    else:
        uppercase_ratio = 0

    # --------------------------------------------------------
    # Regresar todas las características
    # --------------------------------------------------------

    return [
        email_length,
        word_count,
        url_count,
        has_url,
        exclamation_count,
        question_count,
        dollar_count,
        digit_count,
        uppercase_ratio,
        has_urgent,
        has_credentials
    ]


# ============================================================
# 4. NOMBRES DE LAS FEATURES
# ============================================================

feature_names = [
    "email_length",
    "word_count",
    "url_count",
    "has_url",
    "exclamation_count",
    "question_count",
    "dollar_count",
    "digit_count",
    "uppercase_ratio",
    "has_urgent",
    "has_credentials"
]


# ============================================================
# 5. CREAR FEATURES
# ============================================================

X_train_features = np.array(
    [
        extract_features(text)
        for text in X_train_text
    ],
    dtype=float
)

X_test_features = np.array(
    [
        extract_features(text)
        for text in X_test_text
    ],
    dtype=float
)

print(f"\nFeatures extra: {X_train_features.shape[1]}")

print("Características:")

for feature in feature_names:
    print(f" - {feature}")


# ============================================================
# 6. CONVERTIR A MATRIZ SPARSE
# ============================================================

X_train_features = csr_matrix(X_train_features)
X_test_features = csr_matrix(X_test_features)


# ============================================================
# 7. ESCALAR FEATURES
# ============================================================

scaler = MaxAbsScaler()

X_train_features = scaler.fit_transform(
    X_train_features
)

X_test_features = scaler.transform(
    X_test_features
)

print("\n✓ Features extra escaladas con MaxAbsScaler.")


# ============================================================
# 8. COMBINAR TF-IDF + FEATURES
# ============================================================

X_train = hstack([
    X_train_tfidf,
    X_train_features
])

X_test = hstack([
    X_test_tfidf,
    X_test_features
])

print(f"\nMatriz final Train: {X_train.shape}")
print(f"Matriz final Test:  {X_test.shape}")


# ============================================================
# 9. ENTRENAR SVM
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
# 10. PREDICCIONES
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 11. MÉTRICAS
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
# 12. RESULTADOS
# ============================================================

print("\n" + "=" * 50)
print("RESULTADOS — SVM + FEATURES")
print("=" * 50)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")


# ============================================================
# 13. MATRIZ DE CONFUSIÓN
# ============================================================

print("\nMATRIZ DE CONFUSIÓN")

print(cm)


# ============================================================
# 14. DETALLES DE LA MATRIZ
# ============================================================

print("\nDETALLES")

print(f"True Negatives : {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives : {tp}")

print(f"\nTotal errores  : {fp + fn}")
print(f"Total correctos: {tn + tp}")


# ============================================================
# 15. GUARDAR MODELO
# ============================================================

joblib.dump(
    model,
    "models/svm_features_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer_svm_features.pkl"
)

joblib.dump(
    scaler,
    "models/scaler_svm_features.pkl"
)


# ============================================================
# 16. CONFIRMACIÓN
# ============================================================

print("\n✓ Modelo guardado en:")
print("  models/svm_features_model.pkl")

print("\n✓ Vectorizador guardado en:")
print("  models/tfidf_vectorizer_svm_features.pkl")

print("\n✓ Scaler guardado en:")
print("  models/scaler_svm_features.pkl")