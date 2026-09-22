import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# CONFIGURACIÓN
# ============================================================

train_path = "data/processed/train.csv"
test_path = "data/processed/test.csv"


# ============================================================
# CARGAR DATOS
# ============================================================

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train_text = train_df["Email Text"]
y_train = train_df["label"]

X_test_text = test_df["Email Text"]
y_test = test_df["label"]


# ============================================================
# TF-IDF
# ============================================================

print("=" * 60)
print("ENTRENAMIENTO DEL MODELO")
print("=" * 60)

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode"
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


# ============================================================
# MODELO
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

print("✓ Modelo entrenado.")


# ============================================================
# PREDICCIONES
# ============================================================

y_pred = model.predict(X_test)

# Probabilidad de que sea phishing
y_proba = model.predict_proba(X_test)[:, 1]


# ============================================================
# CREAR DATAFRAME DE RESULTADOS
# ============================================================

results = test_df.copy()

results["prediction"] = y_pred
results["phishing_probability"] = y_proba


# ============================================================
# FALSOS POSITIVOS
# ============================================================

false_positives = results[
    (results["label"] == 0) &
    (results["prediction"] == 1)
].copy()


# ============================================================
# FALSOS NEGATIVOS
# ============================================================

false_negatives = results[
    (results["label"] == 1) &
    (results["prediction"] == 0)
].copy()


print("\n" + "=" * 60)
print("RESUMEN DE ERRORES")
print("=" * 60)

print(f"Falsos positivos: {len(false_positives)}")
print(f"Falsos negativos: {len(false_negatives)}")


# ============================================================
# MOSTRAR FALSOS POSITIVOS
# ============================================================

print("\n" + "=" * 60)
print("FALSOS POSITIVOS")
print("=" * 60)

print(
    "\nEstos correos eran SAFE, "
    "pero el modelo los clasificó como PHISHING."
)

# Ordenamos por mayor probabilidad de phishing
false_positives = false_positives.sort_values(
    "phishing_probability",
    ascending=False
)


for i, (_, row) in enumerate(false_positives.head(10).iterrows(), start=1):

    print("\n" + "-" * 60)
    print(f"FALSO POSITIVO #{i}")
    print("-" * 60)

    print(f"ID: {row['Unnamed: 0']}")
    print(
        f"Probabilidad phishing: "
        f"{row['phishing_probability']:.2%}"
    )

    print("\nTexto del correo:")
    print(row["Email Text"][:1500])


# ============================================================
# MOSTRAR FALSOS NEGATIVOS
# ============================================================

print("\n" + "=" * 60)
print("FALSOS NEGATIVOS")
print("=" * 60)

print(
    "\nEstos correos eran PHISHING, "
    "pero el modelo los clasificó como SAFE."
)

# Ordenamos por menor probabilidad de phishing
false_negatives = false_negatives.sort_values(
    "phishing_probability",
    ascending=True
)


for i, (_, row) in enumerate(false_negatives.head(10).iterrows(), start=1):

    print("\n" + "-" * 60)
    print(f"FALSO NEGATIVO #{i}")
    print("-" * 60)

    print(f"ID: {row['Unnamed: 0']}")
    print(
        f"Probabilidad phishing: "
        f"{row['phishing_probability']:.2%}"
    )

    print("\nTexto del correo:")
    print(row["Email Text"][:1500])