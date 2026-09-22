import pandas as pd
import joblib

from scipy.sparse import hstack, csr_matrix


# ==========================================
# 1. CARGAR DATOS
# ==========================================

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

X_train_text = train_df["Email Text"].fillna("")
X_test_text = test_df["Email Text"].fillna("")

y_train = train_df["label"]
y_test = test_df["label"]


# ==========================================
# 2. CARGAR MODELOS
# ==========================================

print("Cargando modelos...")

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
print("✓ Word vectorizer cargado.")
print("✓ Character vectorizer cargado.")
print("✓ Scaler cargado.")


# ==========================================
# 3. FEATURES ESTRUCTURALES
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
# 4. TRANSFORMAR TEST
# ==========================================

print("\nTransformando datos de prueba...")

X_test_word = word_vectorizer.transform(
    X_test_text
)

X_test_char = char_vectorizer.transform(
    X_test_text
)

X_test_features = csr_matrix(
    [
        extract_features(text)
        for text in X_test_text
    ]
)

X_test_features = scaler.transform(
    X_test_features
)

X_test_final = hstack(
    [
        X_test_word,
        X_test_char,
        X_test_features
    ]
).tocsr()

print("✓ Datos transformados.")


# ==========================================
# 5. PREDICCIONES
# ==========================================

y_pred = model.predict(
    X_test_final
)

decision_scores = model.decision_function(
    X_test_final
)


# ==========================================
# 6. CREAR DATAFRAME DE RESULTADOS
# ==========================================

results = test_df.copy()

results["real_label"] = y_test.values
results["predicted_label"] = y_pred
results["decision_score"] = decision_scores

results["real_class"] = results[
    "real_label"
].map({
    0: "Safe Email",
    1: "Phishing Email"
})

results["predicted_class"] = results[
    "predicted_label"
].map({
    0: "Safe Email",
    1: "Phishing Email"
})


# ==========================================
# 7. IDENTIFICAR ERRORES
# ==========================================

false_positives = results[
    (results["real_label"] == 0)
    &
    (results["predicted_label"] == 1)
].copy()

false_negatives = results[
    (results["real_label"] == 1)
    &
    (results["predicted_label"] == 0)
].copy()


# ==========================================
# 8. RESUMEN
# ==========================================

print("\n" + "=" * 60)
print("ANÁLISIS DE ERRORES — MODELO HÍBRIDO")
print("=" * 60)

print(
    f"\nFalsos positivos: "
    f"{len(false_positives)}"
)

print(
    f"Falsos negativos: "
    f"{len(false_negatives)}"
)

print(
    f"Total errores: "
    f"{len(false_positives) + len(false_negatives)}"
)


# ==========================================
# 9. FALSOS POSITIVOS
# ==========================================

print("\n" + "=" * 60)
print("FALSOS POSITIVOS")
print("=" * 60)

print(
    "\nCorreos seguros que el modelo clasificó como phishing.\n"
)

false_positives = false_positives.sort_values(
    "decision_score",
    ascending=False
)

for i, (_, row) in enumerate(
    false_positives.iterrows(),
    start=1
):

    print("\n" + "-" * 60)

    print(
        f"ERROR #{i}"
    )

    print(
        f"ID: {row['Unnamed: 0']}"
    )

    print(
        f"Decision Score: "
        f"{row['decision_score']:.4f}"
    )

    print(
        f"Clase real: "
        f"{row['real_class']}"
    )

    print(
        f"Clase predicha: "
        f"{row['predicted_class']}"
    )

    text = str(row["Email Text"])

    print("\nEMAIL:")

    print(text[:1500])

    if len(text) > 1500:
        print("\n[Texto recortado...]")

    if i >= 19:
        break


# ==========================================
# 10. FALSOS NEGATIVOS
# ==========================================

print("\n" + "=" * 60)
print("FALSOS NEGATIVOS")
print("=" * 60)

print(
    "\nCorreos phishing que el modelo clasificó como seguros.\n"
)

false_negatives = false_negatives.sort_values(
    "decision_score",
    ascending=True
)

for i, (_, row) in enumerate(
    false_negatives.iterrows(),
    start=1
):

    print("\n" + "-" * 60)

    print(
        f"ERROR #{i}"
    )

    print(
        f"ID: {row['Unnamed: 0']}"
    )

    print(
        f"Decision Score: "
        f"{row['decision_score']:.4f}"
    )

    print(
        f"Clase real: "
        f"{row['real_class']}"
    )

    print(
        f"Clase predicha: "
        f"{row['predicted_class']}"
    )

    text = str(row["Email Text"])

    print("\nEMAIL:")

    print(text[:1500])

    if len(text) > 1500:
        print("\n[Texto recortado...]")

    if i >= 10:
        break


# ==========================================
# 11. GUARDAR ERRORES
# ==========================================

false_positives.to_csv(
    "data/processed/false_positives_hybrid.csv",
    index=False
)

false_negatives.to_csv(
    "data/processed/false_negatives_hybrid.csv",
    index=False
)

print("\n" + "=" * 60)
print("ARCHIVOS GENERADOS")
print("=" * 60)

print(
    "\n✓ data/processed/"
    "false_positives_hybrid.csv"
)

print(
    "✓ data/processed/"
    "false_negatives_hybrid.csv"
)

print("\nAnálisis terminado.")