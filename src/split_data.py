import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. CARGAR DATASET LIMPIO
# ============================================================

input_path = "data/processed/phishing_clean.csv"

df = pd.read_csv(input_path)

print("=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print(f"Registros totales: {len(df):,}")


# ============================================================
# 2. CREAR DATAFRAME DE TEXTOS ÚNICOS
# ============================================================

# Cada texto aparece una sola vez.
# Como ya comprobamos que no existen etiquetas contradictorias,
# podemos conservar la primera etiqueta de cada texto.

unique_df = (
    df[
        ["Email Text", "Email Type", "label"]
    ]
    .drop_duplicates(
        subset="Email Text"
    )
    .reset_index(drop=True)
)

print(f"Textos únicos: {len(unique_df):,}")


# ============================================================
# 3. DISTRIBUCIÓN DE CLASES EN TEXTOS ÚNICOS
# ============================================================

print("\nDistribución de clases en textos únicos:")

print(
    unique_df["label"]
    .value_counts()
)

print("\nPorcentaje:")

print(
    unique_df["label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 4. SEPARAR TEXTOS ÚNICOS
# ============================================================

train_unique, test_unique = train_test_split(
    unique_df,
    test_size=0.20,
    random_state=42,
    stratify=unique_df["label"]
)

print("\n" + "=" * 60)
print("TEXTOS ÚNICOS SEPARADOS")
print("=" * 60)

print(f"Train: {len(train_unique):,}")
print(f"Test:  {len(test_unique):,}")


# ============================================================
# 5. OBTENER LOS TEXTOS DE CADA CONJUNTO
# ============================================================

train_texts = set(train_unique["Email Text"])
test_texts = set(test_unique["Email Text"])


# ============================================================
# 6. RECUPERAR TODAS LAS COPIAS DE CADA TEXTO
# ============================================================

train_df = df[
    df["Email Text"].isin(train_texts)
].copy()

test_df = df[
    df["Email Text"].isin(test_texts)
].copy()


# ============================================================
# 7. REINICIAR ÍNDICES
# ============================================================

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ============================================================
# 8. VERIFICAR DATA LEAKAGE
# ============================================================

shared_texts = (
    set(train_df["Email Text"])
    &
    set(test_df["Email Text"])
)

print("\n" + "=" * 60)
print("VERIFICACIÓN DE DATA LEAKAGE")
print("=" * 60)

print(
    f"Textos compartidos entre Train y Test: "
    f"{len(shared_texts):,}"
)

if len(shared_texts) == 0:
    print("✓ No existe data leakage por textos duplicados.")
else:
    print("⚠️ ATENCIÓN: existen textos compartidos.")


# ============================================================
# 9. DISTRIBUCIÓN FINAL DE TRAIN
# ============================================================

print("\n" + "=" * 60)
print("TRAIN")
print("=" * 60)

print(f"Registros: {len(train_df):,}")

print("\nClases:")

print(
    train_df["label"]
    .value_counts()
)

print("\nPorcentaje:")

print(
    train_df["label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 10. DISTRIBUCIÓN FINAL DE TEST
# ============================================================

print("\n" + "=" * 60)
print("TEST")
print("=" * 60)

print(f"Registros: {len(test_df):,}")

print("\nClases:")

print(
    test_df["label"]
    .value_counts()
)

print("\nPorcentaje:")

print(
    test_df["label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 11. VERIFICAR QUE NO FALTEN REGISTROS
# ============================================================

total_split = len(train_df) + len(test_df)

print("\n" + "=" * 60)
print("VERIFICACIÓN FINAL")
print("=" * 60)

print(f"Original: {len(df):,}")
print(f"Train + Test: {total_split:,}")

if total_split == len(df):
    print("✓ Todos los registros fueron asignados.")
else:
    print("⚠️ Hay registros sin asignar.")


# ============================================================
# 12. GUARDAR TRAIN Y TEST
# ============================================================

train_path = "data/processed/train.csv"
test_path = "data/processed/test.csv"

train_df.to_csv(
    train_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)

print("\n" + "=" * 60)
print("ARCHIVOS GUARDADOS")
print("=" * 60)

print(f"Train: {train_path}")
print(f"Test:  {test_path}")