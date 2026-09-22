import pandas as pd
import joblib


model_path = "models/phishing_model.pkl"
vectorizer_path = "models/tfidf_vectorizer.pkl"


print("Cargando modelo...")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

print("✓ Modelo cargado.")
print("✓ Vectorizador cargado.")


def predict_email(email_text):

    email_vector = vectorizer.transform([email_text])

    probabilities = model.predict_proba(email_vector)[0]

    safe_probability = probabilities[0]
    phishing_probability = probabilities[1]

    prediction = model.predict(email_vector)[0]

    print("=" * 60)
    print("PHISHING DETECTOR")
    print("=" * 60)

    print(f"\nProbabilidad Safe:     {safe_probability:.2%}")
    print(f"Probabilidad Phishing: {phishing_probability:.2%}")

    print("\nResultado:")

    if prediction == 1:
        print("🚨 PHISHING EMAIL")
    else:
        print("🟢 SAFE EMAIL")

    print("=" * 60)


print("\nPega el contenido del correo.")
print("Cuando termines, escribe una línea vacía y presiona Enter.\n")


lines = []

while True:

    line = input()

    if line == "":
        break

    lines.append(line)


email = "\n".join(lines)


if not email.strip():

    print("\n⚠️ No se ingresó ningún correo.")

else:

    predict_email(email)