import joblib
import numpy as np


# ============================================================
# CARGAR MODELO Y VECTORIZADOR
# ============================================================

model_path = "models/svm_model.pkl"
vectorizer_path = "models/tfidf_vectorizer_svm.pkl"

print("Cargando modelo SVM...")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

print("✓ Modelo SVM cargado.")
print("✓ Vectorizador cargado.")


# ============================================================
# RECIBIR CORREO
# ============================================================

print("\n" + "=" * 60)
print("EXPLICADOR DE PREDICCIÓN — LINEAR SVM")
print("=" * 60)

print("\nEscribe el contenido del correo.")
print("Cuando termines, presiona ENTER dos veces.\n")

lines = []

while True:
    line = input()

    if line == "":
        break

    lines.append(line)

email_text = "\n".join(lines)


# ============================================================
# VALIDAR ENTRADA
# ============================================================

if not email_text.strip():
    print("\n⚠ No se introdujo ningún correo.")
    exit()


# ============================================================
# TRANSFORMAR TEXTO
# ============================================================

email_vector = vectorizer.transform([email_text])


# ============================================================
# PREDICCIÓN
# ============================================================

prediction = model.predict(email_vector)[0]

decision_score = model.decision_function(email_vector)[0]


# ============================================================
# OBTENER CARACTERÍSTICAS
# ============================================================

feature_names = vectorizer.get_feature_names_out()

# Pesos aprendidos por el SVM
coefficients = model.coef_[0]

# Características que realmente aparecen en este correo
indices = email_vector.nonzero()[1]


# ============================================================
# CALCULAR CONTRIBUCIONES
# ============================================================

contributions = []

for index in indices:

    tfidf_value = email_vector[0, index]

    coefficient = coefficients[index]

    contribution = tfidf_value * coefficient

    contributions.append(
        (
            feature_names[index],
            tfidf_value,
            coefficient,
            contribution
        )
    )


# Ordenar de mayor a menor contribución
contributions.sort(
    key=lambda x: x[3],
    reverse=True
)


# ============================================================
# RESULTADO GENERAL
# ============================================================

print("\n" + "=" * 60)
print("RESULTADO")
print("=" * 60)

print(f"\nDecision Score: {decision_score:.4f}")

if prediction == 1:

    print("\n🚨 PHISHING")
    print("El modelo clasifica este correo como PHISHING.")

else:

    print("\n✅ SAFE")
    print("El modelo clasifica este correo como SEGURO.")


# ============================================================
# CARACTERÍSTICAS QUE EMPUJAN HACIA PHISHING
# ============================================================

print("\n" + "=" * 60)
print("CARACTERÍSTICAS QUE EMPUJAN HACIA PHISHING")
print("=" * 60)

phishing_features = [
    item for item in contributions
    if item[3] > 0
]

phishing_features.sort(
    key=lambda x: x[3],
    reverse=True
)

for word, tfidf, coefficient, contribution in phishing_features[:10]:

    print(
        f"{word:<20} "
        f"contribución: {contribution:+.4f}"
    )


# ============================================================
# CARACTERÍSTICAS QUE EMPUJAN HACIA SAFE
# ============================================================

print("\n" + "=" * 60)
print("CARACTERÍSTICAS QUE EMPUJAN HACIA SAFE")
print("=" * 60)

safe_features = [
    item for item in contributions
    if item[3] < 0
]

safe_features.sort(
    key=lambda x: x[3]
)

for word, tfidf, coefficient, contribution in safe_features[:10]:

    print(
        f"{word:<20} "
        f"contribución: {contribution:+.4f}"
    )