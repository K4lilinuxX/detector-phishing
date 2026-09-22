import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. CARGAR DATOS
# ==========================================

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

X_train_text = train_df["Email Text"].fillna("")
X_test_text = test_df["Email Text"].fillna("")

y_train = train_df["label"]
y_test = test_df["label"]

print(f"Registros Train: {len(train_df)}")
print(f"Registros Test:  {len(test_df)}")


# ==========================================
# 2. TF-IDF DE CARACTERES
# ==========================================

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    lowercase=True,
    sublinear_tf=True,
    min_df=2
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print(f"\nTF-IDF Train: {X_train.shape}")
print(f"TF-IDF Test:  {X_test.shape}")
print(f"Características aprendidas: {X_train.shape[1]}")

print("Configuración: caracteres 3-gramas a 5-gramas")


# ==========================================
# 3. ENTRENAR LINEAR SVM
# ==========================================

print("\nEntrenando SVM...")

model = LinearSVC(
    random_state=42,
    max_iter=5000
)

model.fit(X_train, y_train)

print("✓ Entrenamiento terminado.")


# ==========================================
# 4. PREDICCIONES
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 5. MÉTRICAS
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 50)
print("RESULTADOS — CHARACTER N-GRAMS + SVM")
print("=" * 50)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")


# ==========================================
# 6. CLASSIFICATION REPORT
# ==========================================

print("\nCLASSIFICATION REPORT")

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
# 7. MATRIZ DE CONFUSIÓN
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("MATRIZ DE CONFUSIÓN")
print(cm)

tn, fp, fn, tp = cm.ravel()

print(f"\nTrue Negatives : {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives : {tp}")

print(f"\nTotal errores  : {fp + fn}")
print(f"Total correctos: {tn + tp}")


# ==========================================
# 8. GUARDAR MODELO
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/svm_char_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer_svm_char.pkl"
)

print("\n✓ Modelo guardado en:")
print("  models/svm_char_model.pkl")

print("\n✓ Vectorizador guardado en:")
print("  models/tfidf_vectorizer_svm_char.pkl")