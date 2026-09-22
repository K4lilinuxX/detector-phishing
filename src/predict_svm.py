import joblib


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
print("DETECTOR DE PHISHING — LINEAR SVM")
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
# RESULTADO
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
# INTERPRETACIÓN DEL SCORE
# ============================================================

print("\n" + "-" * 60)
print("Interpretación del Decision Score:")
print("-" * 60)

if decision_score > 0:
    print("Score positivo → el modelo se inclina hacia PHISHING.")

elif decision_score < 0:
    print("Score negativo → el modelo se inclina hacia SAFE.")

else:
    print("Score cercano a 0 → el modelo está cerca de la frontera de decisión.")