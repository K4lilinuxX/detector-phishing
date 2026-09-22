import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# CONFIGURACIÓN
# ============================================================

train_path = "data/processed/train.csv"


# ============================================================
# CARGAR DATOS
# ============================================================

train_df = pd.read_csv(train_path)

X_text = train_df["Email Text"]
y = train_df["label"]


# ============================================================
# TF-IDF
# ============================================================

print("=" * 60)
print("ENTRENANDO MODELO PARA INSPECCIÓN")
print("=" * 60)

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode"
)

X = vectorizer.fit_transform(X_text)

print(f"Registros: {X.shape[0]}")
print(f"Características: {X.shape[1]}")


# ============================================================
# ENTRENAR MODELO
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X, y)

print("✓ Modelo entrenado.")


# ============================================================
# OBTENER CARACTERÍSTICAS
# ============================================================

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]


# ============================================================
# TOP CARACTERÍSTICAS → PHISHING
# ============================================================

top_phishing_indices = coefficients.argsort()[-20:][::-1]

print("\n" + "=" * 60)
print("TOP 20 CARACTERÍSTICAS ASOCIADAS CON PHISHING")
print("=" * 60)

for position, index in enumerate(top_phishing_indices, start=1):
    print(
        f"{position:2}. "
        f"{feature_names[index]:30} "
        f"{coefficients[index]:+.4f}"
    )


# ============================================================
# TOP CARACTERÍSTICAS → SAFE
# ============================================================

top_safe_indices = coefficients.argsort()[:20]

print("\n" + "=" * 60)
print("TOP 20 CARACTERÍSTICAS ASOCIADAS CON SAFE EMAIL")
print("=" * 60)

for position, index in enumerate(top_safe_indices, start=1):
    print(
        f"{position:2}. "
        f"{feature_names[index]:30} "
        f"{coefficients[index]:+.4f}"
    )