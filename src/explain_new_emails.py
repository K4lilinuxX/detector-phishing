import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack


# ==========================================
# 1. CARGAR MODELO Y TRANSFORMADORES
# ==========================================

print("Cargando modelo híbrido...")

model = joblib.load(
    "models/svm_hybrid_model.pkl"
)

word_vectorizer = joblib.load(
    "models/tfidf_vectorizer_word_hybrid.pkl"
)

char_vectorizer = joblib.load(
    "models/tfidf_vectorizer_char_hybrid.pkl"
)

scaler = joblib.load(
    "models/scaler_hybrid.pkl"
)

print("✓ Modelo cargado.")
print("✓ Vectorizadores cargados.")
print("✓ Scaler cargado.")


# ==========================================
# 2. FEATURES ESTRUCTURALES
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
    "has_urgent",
    "has_credentials"
]


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


# ==========================================
# 3. FUNCIÓN DE EXPLICACIÓN
# ==========================================

def explain_email(email, title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    # --------------------------------------
    # Transformaciones
    # --------------------------------------

    X_word = word_vectorizer.transform(
        [email]
    )

    X_char = char_vectorizer.transform(
        [email]
    )

    raw_features = extract_features(
        email
    )

    X_features = csr_matrix(
        [raw_features]
    )

    X_features = scaler.transform(
        X_features
    )

    X_final = hstack(
        [
            X_word,
            X_char,
            X_features
        ]
    ).tocsr()

    # --------------------------------------
    # Predicción
    # --------------------------------------

    prediction = model.predict(
        X_final
    )[0]

    score = model.decision_function(
        X_final
    )[0]

    prediction_name = (
        "PHISHING"
        if prediction == 1
        else "SAFE"
    )

    print(f"\nPredicción: {prediction_name}")
    print(f"Decision Score: {score:.4f}")

    # --------------------------------------
    # Coeficientes
    # --------------------------------------

    coefficients = model.coef_[0]

    word_size = len(
        word_vectorizer.get_feature_names_out()
    )

    char_size = len(
        char_vectorizer.get_feature_names_out()
    )

    # ======================================
    # PALABRAS
    # ======================================

    word_names = (
        word_vectorizer
        .get_feature_names_out()
    )

    word_values = X_word.toarray()[0]

    word_contributions = (
        word_values * coefficients[:word_size]
    )

    word_indexes = np.where(
        word_values != 0
    )[0]

    word_results = [
        (
            word_names[i],
            word_contributions[i]
        )
        for i in word_indexes
    ]

    word_results = sorted(
        word_results,
        key=lambda x: x[1],
        reverse=True
    )

    # ======================================
    # CARACTERES
    # ======================================

    char_names = (
        char_vectorizer
        .get_feature_names_out()
    )

    char_values = X_char.toarray()[0]

    char_start = word_size

    char_end = word_size + char_size

    char_contributions = (
        char_values
        * coefficients[
            char_start:char_end
        ]
    )

    char_indexes = np.where(
        char_values != 0
    )[0]

    char_results = [
        (
            char_names[i],
            char_contributions[i]
        )
        for i in char_indexes
    ]

    char_results = sorted(
        char_results,
        key=lambda x: x[1],
        reverse=True
    )

    # ======================================
    # FEATURES ESTRUCTURALES
    # ======================================

    feature_coefficients = coefficients[
        word_size + char_size:
    ]

    feature_values = (
        X_features.toarray()[0]
    )

    feature_contributions = (
        feature_values
        * feature_coefficients
    )

    feature_results = list(
        zip(
            feature_names,
            feature_values,
            feature_contributions
        )
    )

    feature_results = sorted(
        feature_results,
        key=lambda x: x[2],
        reverse=True
    )

    # ======================================
    # MOSTRAR PALABRAS
    # ======================================

    print("\n" + "-" * 70)
    print("PALABRAS QUE EMPUJAN HACIA PHISHING")
    print("-" * 70)

    phishing_words = [
        x for x in word_results
        if x[1] > 0
    ]

    for name, contribution in phishing_words[:10]:

        print(
            f"{name:<25} "
            f"+{contribution:.4f}"
        )

    print("\n" + "-" * 70)
    print("PALABRAS QUE EMPUJAN HACIA SAFE")
    print("-" * 70)

    safe_words = [
        x for x in word_results
        if x[1] < 0
    ]

    for name, contribution in safe_words[-10:][::-1]:

        print(
            f"{name:<25} "
            f"{contribution:.4f}"
        )

    # ======================================
    # CARACTERES
    # ======================================

    print("\n" + "-" * 70)
    print("CARACTERÍSTICAS DE CARACTERES")
    print("-" * 70)

    print("\nHacia PHISHING:")

    char_phishing = [
        x for x in char_results
        if x[1] > 0
    ]

    for name, contribution in char_phishing[:10]:

        print(
            f"{repr(name):<15} "
            f"+{contribution:.4f}"
        )

    print("\nHacia SAFE:")

    char_safe = [
        x for x in char_results
        if x[1] < 0
    ]

    for name, contribution in char_safe[:10]:

        print(
            f"{repr(name):<15} "
            f"{contribution:.4f}"
        )

    # ======================================
    # FEATURES ESTRUCTURALES
    # ======================================

    print("\n" + "-" * 70)
    print("FEATURES ESTRUCTURALES")
    print("-" * 70)

    for name, value, contribution in feature_results:

        direction = (
            "PHISHING"
            if contribution > 0
            else "SAFE"
        )

        print(
            f"{name:<22} "
            f"valor={value:.4f} "
            f"contribución={contribution:+.4f} "
            f"→ {direction}"
        )


# ==========================================
# 4. CORREO #2
# ==========================================

email_2 = """
Dear user,

Your account requires a security verification.

Please login using the link below and confirm
your username and password:

http://account-security.example.com/login

If you do not complete this process today,
your account will be locked.

Thank you.
"""


# ==========================================
# 5. CORREO #7
# ==========================================

email_7 = """
Summer Sale!

Enjoy 30% off selected products this weekend.

Visit our store to discover our latest
products and special offers.

Thank you for being a customer.
"""


# ==========================================
# 6. EJECUTAR
# ==========================================

explain_email(
    email_2,
    "CASO #2 — PHISHING DE CREDENCIALES"
)

explain_email(
    email_7,
    "CASO #7 — MARKETING LEGÍTIMO"
)