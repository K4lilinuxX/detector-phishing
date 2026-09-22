import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix


# ============================================================
# CONFIGURACIÓN
# ============================================================

MODEL_PATH = "models/svm_hybrid_v2_model.pkl"
WORD_VECTORIZER_PATH = "models/tfidf_vectorizer_word_hybrid_v2.pkl"
CHAR_VECTORIZER_PATH = "models/tfidf_vectorizer_char_hybrid_v2.pkl"
SCALER_PATH = "models/scaler_hybrid_v2.pkl"


# ============================================================
# CARGAR MODELO Y TRANSFORMADORES
# ============================================================

print("\nCargando modelo...")

model = joblib.load(MODEL_PATH)
word_vectorizer = joblib.load(WORD_VECTORIZER_PATH)
char_vectorizer = joblib.load(CHAR_VECTORIZER_PATH)
scaler = joblib.load(SCALER_PATH)

print("✓ Modelo cargado.")
print("✓ Vectorizador de palabras cargado.")
print("✓ Vectorizador de caracteres cargado.")
print("✓ Scaler cargado.")


# ============================================================
# EXTRACCIÓN DE FEATURES
# ============================================================

def extract_features(text):
    """
    Extrae las mismas features utilizadas durante el entrenamiento.
    """

    text_lower = text.lower()

    words = text.split()

    email_length = len(text)
    word_count = len(words)

    url_count = (
        text_lower.count("http://")
        + text_lower.count("https://")
        + text_lower.count("www.")
    )

    has_url = int(url_count > 0)

    exclamation_count = text.count("!")
    question_count = text.count("?")
    dollar_count = text.count("$")

    digit_count = sum(char.isdigit() for char in text)

    if len(text) > 0:
        uppercase_ratio = sum(
            char.isupper() for char in text
        ) / len(text)
    else:
        uppercase_ratio = 0

    has_urgent = int(
        any(
            word in text_lower
            for word in [
                "urgent",
                "urgente",
                "immediately",
                "immediately",
                "asap"
            ]
        )
    )

    has_credentials = int(
        any(
            phrase in text_lower
            for phrase in [
                "password",
                "contraseña",
                "username",
                "usuario",
                "credentials",
                "credenciales"
            ]
        )
    )

    has_password_request = int(
        any(
            phrase in text_lower
            for phrase in [
                "enter your password",
                "provide your password",
                "confirm your password",
                "ingresa tu contraseña",
                "introduce tu contraseña",
                "proporciona tu contraseña"
            ]
        )
    )

    has_login_request = int(
        any(
            phrase in text_lower
            for phrase in [
                "log in",
                "login",
                "sign in",
                "inicia sesión",
                "iniciar sesión",
                "accede a tu cuenta"
            ]
        )
    )

    has_identity_verification = int(
        any(
            phrase in text_lower
            for phrase in [
                "verify your identity",
                "identity verification",
                "verify your account",
                "verifica tu identidad",
                "verificación de identidad",
                "verifica tu cuenta"
            ]
        )
    )

    has_account_threat = int(
        any(
            phrase in text_lower
            for phrase in [
                "account will be closed",
                "account will be suspended",
                "account has been suspended",
                "your account will be locked",
                "cuenta será cerrada",
                "cuenta será suspendida",
                "cuenta será bloqueada",
                "tu cuenta será bloqueada"
            ]
        )
    )

    has_reward_claim = int(
        any(
            phrase in text_lower
            for phrase in [
                "you have won",
                "you won",
                "claim your prize",
                "claim your reward",
                "has ganado",
                "ganaste",
                "reclama tu premio",
                "reclama tu recompensa"
            ]
        )
    )

    has_financial_request = int(
        any(
            phrase in text_lower
            for phrase in [
                "bank account",
                "credit card",
                "card number",
                "bank transfer",
                "cuenta bancaria",
                "tarjeta de crédito",
                "número de tarjeta",
                "transferencia bancaria"
            ]
        )
    )

    has_personal_info_request = int(
        any(
            phrase in text_lower
            for phrase in [
                "personal information",
                "personal details",
                "date of birth",
                "social security",
                "información personal",
                "datos personales",
                "fecha de nacimiento"
            ]
        )
    )

    features = np.array([[
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
    ]])

    return features


# ============================================================
# PREDICCIÓN
# ============================================================

def predict_email(email_text):

    # TF-IDF de palabras
    X_word = word_vectorizer.transform([email_text])

    # TF-IDF de caracteres
    X_char = char_vectorizer.transform([email_text])

    # Features estructurales/contextuales
    features = extract_features(email_text)

    features = csr_matrix(features)

    # Escalado
    features_scaled = scaler.transform(features)

    # Combinar todas las features
    X_final = hstack([
        X_word,
        X_char,
        features_scaled
    ]).tocsr()

    # Predicción
    prediction = model.predict(X_final)[0]

    # Decision score
    score = model.decision_function(X_final)[0]

    return prediction, score


# ============================================================
# INTERFAZ
# ============================================================

print("\n" + "=" * 70)
print("           DETECTOR DE CORREOS DE PHISHING")
print("=" * 70)

print("\nPega el contenido del correo.")
print("Cuando termines, presiona ENTER dos veces.")
print("Escribe 'salir' para cerrar el programa.\n")


while True:

    lines = []

    while True:

        line = input()

        if not lines and line.lower() == "salir":
            print("\nPrograma finalizado.")
            exit()

        if line == "":
            break

        lines.append(line)

    email_text = "\n".join(lines).strip()

    if not email_text:
        print("\n⚠️ No ingresaste ningún correo.\n")
        continue

    prediction, score = predict_email(email_text)

    print("\n" + "-" * 70)
    print("RESULTADO")
    print("-" * 70)

    if prediction == 1:

        print("🔴 PHISHING")

    else:

        print("🟢 SAFE")

    print(f"Decision Score: {score:.4f}")

    if score > 0:
        print("El modelo lo clasificó como phishing.")

    else:
        print("El modelo lo clasificó como correo seguro.")

    print("-" * 70)
    print("\nPuedes analizar otro correo o escribir 'salir'.\n")
