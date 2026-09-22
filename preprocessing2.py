from datasets import load_dataset
import pandas as pd

# ============================================================
# 1. CARGAR DATASET
# ============================================================

dataset = load_dataset("zefang-liu/phishing-email-dataset")
df = dataset["train"].to_pandas()

# Crear etiqueta
df["label"] = (df["Email Type"] == "Phishing Email").astype(int)

# ============================================================
# 2. ANALIZAR EL VALOR "empty"
# ============================================================

raw = df["Email Text"].astype("string")

mask_exact = raw == "empty"

mask_normalized = (
    raw
    .fillna("")
    .str.strip()
    .str.lower()
    .eq("empty")
)

print("=" * 60)
print("ANÁLISIS DEL VALOR 'empty'")
print("=" * 60)

print(f'Exactamente "empty": {mask_exact.sum():,}')
print(
    f'Igual a "empty" ignorando espacios/mayúsculas: '
    f'{mask_normalized.sum():,}'
)

# ============================================================
# 3. DISTRIBUCIÓN DE CLASES
# ============================================================

print("\n" + "=" * 60)
print("DISTRIBUCIÓN DE CLASES EN LOS REGISTROS 'empty'")
print("=" * 60)

print(
    df.loc[
        mask_normalized,
        ["Email Type", "label"]
    ].value_counts()
)

# ============================================================
# 4. MOSTRAR EJEMPLOS CRUDOS
# ============================================================

print("\n" + "=" * 60)
print("EJEMPLOS CRUDOS")
print("=" * 60)

sample = df.loc[
    mask_normalized,
    ["Unnamed: 0", "Email Type", "label", "Email Text"]
].head(20).copy()

sample["Email Text"] = sample["Email Text"].map(repr)

print(sample.to_string(index=False))

# ============================================================
# 5. BUSCAR VARIANTES DE "EMPTY"
# ============================================================

print("\n" + "=" * 60)
print("VALORES QUE CONTIENEN 'empty'")
print("=" * 60)

empty_like = df.loc[
    raw.fillna("")
    .str.strip()
    .str.lower()
    .str.contains("empty", regex=False),
    "Email Text"
].value_counts(dropna=False)

for value, count in empty_like.head(20).items():
    print(repr(value), "->", count)

# ============================================================
# 6. ANALIZAR LOS REGISTROS REALMENTE VACÍOS
# ============================================================

blank_mask = raw.fillna("").str.strip().eq("")

print("\n" + "=" * 60)
print("REGISTROS REALMENTE VACÍOS")
print("=" * 60)

print(f"Total: {blank_mask.sum():,}")

print(
    df.loc[
        blank_mask,
        ["Unnamed: 0", "Email Type", "label", "Email Text"]
    ].to_string(index=False)
)

# ============================================================
# 7. REVISAR EL EMAIL MÁS LARGO
# ============================================================

df["email_length"] = raw.fillna("").str.len()

idx = df["email_length"].idxmax()
text = df.loc[idx, "Email Text"]

print("\n" + "=" * 60)
print("EMAIL MÁS LARGO")
print("=" * 60)

print("ID:", df.loc[idx, "Unnamed: 0"])
print("Tipo:", df.loc[idx, "Email Type"])
print("Longitud:", df.loc[idx, "email_length"])

print("\n--- INICIO ---")
print(repr(str(text)[:500]))

print("\n--- FINAL ---")
print(repr(str(text)[-500:]))

# ============================================================
# ANÁLISIS RESUMIDO DE "empty"
# ============================================================

raw = df["Email Text"].astype("string")

mask_empty = (
    raw
    .fillna("")
    .str.strip()
    .str.lower()
    .eq("empty")
)

print("=" * 60)
print("RESUMEN DEL VALOR 'empty'")
print("=" * 60)

print(f"Total de registros: {mask_empty.sum():,}")

print("\nDistribución por tipo:")
print(df.loc[mask_empty, "Email Type"].value_counts())

print("\nDistribución por label:")
print(df.loc[mask_empty, "label"].value_counts())

print("\nPorcentaje por clase:")

class_percent = (
    df.loc[mask_empty, "label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(class_percent)

print("\nValores únicos encontrados:")
unique_values = df.loc[mask_empty, "Email Text"].drop_duplicates()

for value in unique_values:
    print(repr(value))

print("\nPrimeros 5 IDs:")
print(
    df.loc[
        mask_empty,
        ["Unnamed: 0", "Email Type", "label"]
    ].head().to_string(index=False)
)

print("\nÚltimos 5 IDs:")
print(
    df.loc[
        mask_empty,
        ["Unnamed: 0", "Email Type", "label"]
    ].tail().to_string(index=False)
)