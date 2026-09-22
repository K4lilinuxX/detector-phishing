import joblib


# ==========================================
# 1. CARGAR MODELO
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
# 3. FUNCIÓN DE PREDICCIÓN
# ==========================================

def predict_email(email):

    X_word = word_vectorizer.transform(
        [email]
    )

    X_char = char_vectorizer.transform(
        [email]
    )

    from scipy.sparse import csr_matrix, hstack

    X_features = csr_matrix(
        [extract_features(email)]
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

    prediction = model.predict(
        X_final
    )[0]

    score = model.decision_function(
        X_final
    )[0]

    return prediction, score


# ==========================================
# 4. CORREOS NUEVOS
# ==========================================

emails = [

    {
        "name": "Phishing obvio",
        "expected": 1,
        "text": """
URGENT! Your bank account has been suspended.

We detected unusual activity on your account.
You must verify your identity immediately.

Click here to restore access:
http://secure-bank-verification.example.com

Failure to verify your account within 24 hours
will result in permanent suspension.
"""
    },

    {
        "name": "Phishing de credenciales",
        "expected": 1,
        "text": """
Dear user,

Your account requires a security verification.

Please login using the link below and confirm
your username and password:

http://account-security.example.com/login

If you do not complete this process today,
your account will be locked.

Thank you.
"""
    },

    {
        "name": "Phishing financiero",
        "expected": 1,
        "text": """
Congratulations!

You have been selected to receive a $5,000
investment bonus.

Click the link below to claim your reward
and provide your account information:

http://claim-reward.example.com

This offer expires today.
"""
    },

    {
        "name": "Phishing sutil",
        "expected": 1,
        "text": """
Hello,

We noticed a recent sign-in to your account
from a new device.

If this was not you, please review your
account activity and confirm your identity:

http://security-check.example.com

Thank you.
"""
    },

    {
        "name": "Correo personal",
        "expected": 0,
        "text": """
Hi John,

Just wanted to confirm that we're still meeting
tomorrow at 10:00 AM.

I'll bring the documents we discussed.

See you tomorrow,
Michael
"""
    },

    {
        "name": "Correo académico",
        "expected": 0,
        "text": """
Hello Professor,

I am writing to confirm that I will attend
tomorrow's class.

I have completed the assignment and will bring
a printed copy to the classroom.

Best regards,
Alex
"""
    },

    {
        "name": "Marketing legítimo",
        "expected": 0,
        "text": """
Summer Sale!

Enjoy 30% off selected products this weekend.

Visit our store to discover our latest
products and special offers.

Thank you for being a customer.
"""
    },

    {
        "name": "Newsletter legítima",
        "expected": 0,
        "text": """
Thank you for subscribing to our monthly newsletter.

This month we are sharing our latest articles,
community updates, and upcoming events.

You can unsubscribe at any time using the link
at the bottom of this email.

Have a great day!
"""
    }

]


# ==========================================
# 5. EJECUTAR PRUEBAS
# ==========================================

print("\n" + "=" * 65)
print("EXPERIMENTO H — CORREOS NUEVOS")
print("=" * 65)


correct = 0


for i, email in enumerate(emails, start=1):

    prediction, score = predict_email(
        email["text"]
    )

    predicted_name = (
        "PHISHING"
        if prediction == 1
        else "SAFE"
    )

    expected_name = (
        "PHISHING"
        if email["expected"] == 1
        else "SAFE"
    )

    is_correct = (
        prediction == email["expected"]
    )

    if is_correct:
        correct += 1

    result = "✓ CORRECTO" if is_correct else "✗ ERROR"

    print("\n" + "-" * 65)

    print(
        f"Prueba #{i}: {email['name']}"
    )

    print(
        f"Esperado : {expected_name}"
    )

    print(
        f"Predicción: {predicted_name}"
    )

    print(
        f"Decision Score: {score:.4f}"
    )

    print(
        f"Resultado: {result}"
    )


# ==========================================
# 6. RESUMEN
# ==========================================

total = len(emails)

print("\n" + "=" * 65)
print("RESUMEN")
print("=" * 65)

print(
    f"\nPruebas correctas: "
    f"{correct}/{total}"
)

print(
    f"Pruebas incorrectas: "
    f"{total - correct}/{total}"
)

print(
    f"Accuracy del conjunto manual: "
    f"{correct / total:.2%}"
)