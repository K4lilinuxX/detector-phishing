import pandas as pd
import numpy as np
import joblib
import os

from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. CARGAR DATOS
# ==========================================

train_df = pd.read_csv(
    "data/processed/train.csv"
)

test_df = pd.read_csv(
    "data/processed/test.csv"
)

X_train_text = train_df["Email Text"].fillna("")
y_train = train_df["label"]

X_test_text = test_df["Email Text"].fillna("")
y_test = test_df["label"]

print(f"Registros Train: {len(train_df)}")
print(f"Registros Test:  {len(test_df)}")


# ==========================================
# 2. WORD TF-IDF
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

X_train_word = word_vectorizer.fit_transform(
    X_train_text
)

X_test_word = word_vectorizer.transform(
    X_test_text
)

print(
    f"Word TF-IDF Train: "
    f"{X_train_word.shape}"
)

print(
    f"Word TF-IDF Test: "
    f"{X_test_word.shape}"
)

print(
    f"Características Word: "
    f"{X_train_word.shape[1]}"
)


# ==========================================
# 3. CHARACTER TF-IDF
# ==========================================

print("\nGenerando TF-IDF de caracteres...")

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    lowercase=True,
    sublinear_tf=True,
    min_df=2
)

X_train_char = char_vectorizer.fit_transform(
    X_train_text
)

X_test_char = char_vectorizer.transform(
    X_test_text
)

print(
    f"Char TF-IDF Train: "
    f"{X_train_char.shape}"
)

print(
    f"Char TF-IDF Test: "
    f"{X_test_char.shape}"
)

print(
    f"Características Char: "
    f"{X_train_char.shape[1]}"
)


# ==========================================
# 4. FEATURES
# ==========================================

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

    # Features originales
    "has_urgent",
    "has_credentials",

    # Nuevas features
    "has_password_request",
    "has_login_request",
    "has_identity_verification",
    "has_account_threat",
    "has_reward_claim",
    "has_financial_request",
    "has_personal_info_request"
]


def extract_features(text):

    text = str(text)

    lower = text.lower()

    words = text.split()

    email_length = len(text)
    word_count = len(words)

    # --------------------------------------
    # Estructurales
    # --------------------------------------

    url_count = (
        lower.count("http://")
        + lower.count("https://")
        + lower.count("www.")
    )

    has_url = int(
        url_count > 0
    )

    exclamation_count = text.count("!")
    question_count = text.count("?")
    dollar_count = text.count("$")

    digit_count = sum(
        char.isdigit()
        for char in text
    )

    uppercase_count = sum(
        char.isupper()
        for char in text
    )

    if len(text) > 0:

        uppercase_ratio = (
            uppercase_count / len(text)
        )

    else:

        uppercase_ratio = 0

    # --------------------------------------
    # Features originales
    # --------------------------------------

    has_urgent = int(
        any(
            phrase in lower
            for phrase in [
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
            word in lower
            for word in [
                "password",
                "username",
                "login",
                "credentials",
                "verify your identity"
            ]
        )
    )

    # --------------------------------------
    # Nuevas features contextuales
    # --------------------------------------

    password_request_patterns = [
        "enter your password",
        "provide your password",
        "confirm your password",
        "submit your password",
        "type your password",
        "password below"
    ]

    has_password_request = int(
        any(
            phrase in lower
            for phrase in password_request_patterns
        )
    )

    login_patterns = [
        "login to your account",
        "log in to your account",
        "login using the link",
        "log in using the link",
        "sign in to your account",
        "sign into your account"
    ]

    has_login_request = int(
        any(
            phrase in lower
            for phrase in login_patterns
        )
    )

    identity_patterns = [
        "verify your identity",
        "confirm your identity",
        "identity verification",
        "verify your account",
        "confirm your account"
    ]

    has_identity_verification = int(
        any(
            phrase in lower
            for phrase in identity_patterns
        )
    )

    account_threat_patterns = [
        "account will be locked",
        "account will be suspended",
        "account has been suspended",
        "account will be closed",
        "account has been locked",
        "failure to verify",
        "your account is at risk"
    ]

    has_account_threat = int(
        any(
            phrase in lower
            for phrase in account_threat_patterns
        )
    )

    reward_patterns = [
        "claim your reward",
        "claim your prize",
        "you have won",
        "you won",
        "selected to receive",
        "receive a bonus",
        "claim your bonus"
    ]

    has_reward_claim = int(
        any(
            phrase in lower
            for phrase in reward_patterns
        )
    )

    financial_patterns = [
        "bank account",
        "credit card",
        "account information",
        "investment bonus",
        "financial information",
        "payment information",
        "billing information"
    ]

    has_financial_request = int(
        any(
            phrase in lower
            for phrase in financial_patterns
        )
    )

    personal_info_patterns = [
        "personal information",
        "personal details",
        "provide your information",
        "provide your details",
        "social security",
        "date of birth",
        "phone number",
        "home address"
    ]

    has_personal_info_request = int(
        any(
            phrase in lower
            for phrase in personal_info_patterns
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
        has_credentials,
        has_password_request,
        has_login_request,
        has_identity_verification,
        has_account_threat,
        has_reward_claim,
        has_financial_request,
        has_personal_info_request
    ]


# ==========================================
# 5. EXTRAER FEATURES TRAIN / TEST
# ==========================================

print("\nExtrayendo features estructurales y contextuales...")

X_train_features = np.array([
    extract_features(text)
    for text in X_train_text
])

X_test_features = np.array([
    extract_features(text)
    for text in X_test_text
])

print(
    f"Features extra: "
    f"{X_train_features.shape[1]}"
)

print("\nCaracterísticas:")

for name in feature_names:

    print(f" - {name}")


# ==========================================
# 6. ESCALAR FEATURES
# ==========================================

scaler = MaxAbsScaler()

X_train_features = csr_matrix(
    X_train_features
)

X_test_features = csr_matrix(
    X_test_features
)

X_train_features = scaler.fit_transform(
    X_train_features
)

X_test_features = scaler.transform(
    X_test_features
)

print(
    "\n✓ Features escaladas con "
    "MaxAbsScaler."
)


# ==========================================
# 7. COMBINAR TODO
# ==========================================

print(
    "\nCombinando Word + Character + Features..."
)

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

print(
    f"\nMatriz final Train: "
    f"{X_train_final.shape}"
)

print(
    f"Matriz final Test: "
    f"{X_test_final.shape}"
)


# ==========================================
# 8. ENTRENAR SVM
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
# 9. PREDICCIONES
# ==========================================

y_pred = model.predict(
    X_test_final
)


# ==========================================
# 10. MÉTRICAS
# ==========================================

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


print(
    "\n" + "=" * 70
)

print(
    "RESULTADOS — HYBRID V2 + SVM"
)

print(
    "=" * 70
)

print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1-Score : {f1:.4f}"
)


# ==========================================
# 11. CLASSIFICATION REPORT
# ==========================================

print(
    "\nCLASSIFICATION REPORT"
)

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
# 12. MATRIZ DE CONFUSIÓN
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

print(
    "\nMATRIZ DE CONFUSIÓN"
)

print(cm)

print(
    f"\nTrue Negatives : {tn}"
)

print(
    f"False Positives: {fp}"
)

print(
    f"False Negatives: {fn}"
)

print(
    f"True Positives : {tp}"
)

print(
    f"Total errores  : {fp + fn}"
)

print(
    f"Total correctos: {tn + tp}"
)


# ==========================================
# 13. GUARDAR MODELO
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/svm_hybrid_v2_model.pkl"
)

joblib.dump(
    word_vectorizer,
    "models/tfidf_vectorizer_word_hybrid_v2.pkl"
)

joblib.dump(
    char_vectorizer,
    "models/tfidf_vectorizer_char_hybrid_v2.pkl"
)

joblib.dump(
    scaler,
    "models/scaler_hybrid_v2.pkl"
)

print(
    "\n✓ Modelo guardado en:"
)

print(
    "  models/svm_hybrid_v2_model.pkl"
)

print(
    "✓ Word vectorizer guardado en:"
)

print(
    "  models/tfidf_vectorizer_word_hybrid_v2.pkl"
)

print(
    "✓ Char vectorizer guardado en:"
)

print(
    "  models/tfidf_vectorizer_char_hybrid_v2.pkl"
)

print(
    "✓ Scaler guardado en:"
)

print(
    "  models/scaler_hybrid_v2.pkl"
)