import pandas as pd
import joblib
import os

from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. CARGAR DATOS
# ==========================================

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

X_train_text = train_df["Email Text"].fillna("")
X_test_text = test_df["Email Text"].fillna("")

y_train = train_df["label"]
y_test = test_df["label"]

print(f"Registros Train: {len(train_df)}")
print(f"Registros Test:  {len(test_df)}")


# ==========================================
# 2. WORD-LEVEL TF-IDF
# ==========================================

print("\nGenerando TF-IDF de palabras...")

word_vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    stop_words="english",
    ngram_range=(1, 1),
    sublinear_tf=True,
    min_df=2
)

X_train_word = word_vectorizer.fit_transform(X_train_text)
X_test_word = word_vectorizer.transform(X_test_text)

print(f"Word TF-IDF Train: {X_train_word.shape}")
print(f"Word TF-IDF Test:  {X_test_word.shape}")
print(f"Características Word: {X_train_word.shape[1]}")


# ==========================================
# 3. CHARACTER-LEVEL TF-IDF
# ==========================================

print("\nGenerando TF-IDF de caracteres...")

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    lowercase=True,
    sublinear_tf=True,
    min_df=2
)

X_train_char = char_vectorizer.fit_transform(X_train_text)
X_test_char = char_vectorizer.transform(X_test_text)

print(f"Char TF-IDF Train: {X_train_char.shape}")
print(f"Char TF-IDF Test:  {X_test_char.shape}")
print(f"Características Char: {X_train_char.shape[1]}")


# ==========================================
# 4. FEATURES ESTRUCTURALES
# ==========================================

def extract_features(text):
    text = str(text)

    words = text.split()

    email_length = len(text)
    word_count = len(words)

    url_count = (
        text.lower().count("http://")
        + text.lower().count("https://")
        + text.lower().count("www.")
    )

    has_url = int(url_count > 0)

    exclamation_count = text.count("!")
    question_count = text.count("?")
    dollar_count = text.count("$")

    digit_count = sum(char.isdigit() for char in text)

    uppercase_count = sum(char.isupper() for char in text)

    if len(text) > 0:
        uppercase_ratio = uppercase_count / len(text)
    else:
        uppercase_ratio = 0

    has_urgent = int(
        any(
            word in text.lower()
            for word in [
                "urgent",
                "immediately",
                "action required",
                "verify now",
                "account suspended"
            ]
        )
    )

    has_credentials = int(
        any(
            word in text.lower()
            for word in [
                "password",
                "username",
                "login",
                "credentials",
                "verify your identity"
            ]
        )
    )

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


print("\nExtrayendo features estructurales...")

X_train_features = csr_matrix(
    [extract_features(text) for text in X_train_text]
)

X_test_features = csr_matrix(
    [extract_features(text) for text in X_test_text]
)

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

print(f"Features extra: {len(feature_names)}")

print("Características:")
for feature in feature_names:
    print(f" - {feature}")


# ==========================================
# 5. ESCALAR FEATURES
# ==========================================

scaler = MaxAbsScaler()

X_train_features = scaler.fit_transform(
    X_train_features
)

X_test_features = scaler.transform(
    X_test_features
)

print("\n✓ Features extra escaladas con MaxAbsScaler.")


# ==========================================
# 6. COMBINAR TODAS LAS FEATURES
# ==========================================

print("\nCombinando Word + Character + Features...")

X_train_final = hstack(
    [
        X_train_word,
        X_train_char,
        X_train_features
    ]
).tocsr()

X_test_final = hstack(
    [
        X_test_word,
        X_test_char,
        X_test_features
    ]
).tocsr()

print(f"\nMatriz final Train: {X_train_final.shape}")
print(f"Matriz final Test:  {X_test_final.shape}")


# ==========================================
# 7. ENTRENAR LINEAR SVM
# ==========================================

print("\nEntrenando SVM...")

model = LinearSVC(
    random_state=42,
    max_iter=5000
)

model.fit(
    X_train_final,
    y_train
)

print("✓ Entrenamiento terminado.")


# ==========================================
# 8. PREDICCIONES
# ==========================================

y_pred = model.predict(X_test_final)


# ==========================================
# 9. MÉTRICAS
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 55)
print("RESULTADOS — WORD + CHAR + FEATURES + SVM")
print("=" * 55)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")


# ==========================================
# 10. CLASSIFICATION REPORT
# ==========================================

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Safe Email",
            "Phishing Email"
        ]
    )
)


# ==========================================
# 11. MATRIZ DE CONFUSIÓN
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("MATRIZ DE CONFUSIÓN")
print(cm)

tn, fp, fn, tp = cm.ravel()

print(f"\nTrue Negatives : {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives : {tp}")

print(f"\nTotal errores  : {fp + fn}")
print(f"Total correctos: {tn + tp}")


# ==========================================
# 12. GUARDAR MODELO
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/svm_hybrid_model.pkl"
)

joblib.dump(
    word_vectorizer,
    "models/tfidf_vectorizer_word_hybrid.pkl"
)

joblib.dump(
    char_vectorizer,
    "models/tfidf_vectorizer_char_hybrid.pkl"
)

joblib.dump(
    scaler,
    "models/scaler_hybrid.pkl"
)

print("\n✓ Modelo guardado en:")
print("  models/svm_hybrid_model.pkl")

print("\n✓ Word vectorizer guardado en:")
print("  models/tfidf_vectorizer_word_hybrid.pkl")

print("\n✓ Character vectorizer guardado en:")
print("  models/tfidf_vectorizer_char_hybrid.pkl")

print("\n✓ Scaler guardado en:")
print("  models/scaler_hybrid.pkl")