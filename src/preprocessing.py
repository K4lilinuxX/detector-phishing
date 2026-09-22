from datasets import load_dataset
import pandas as pd


# ============================================================
# 1. CARGAR DATASET
# ============================================================

print("Cargando dataset...")

dataset = load_dataset("zefang-liu/phishing-email-dataset")
df = dataset["train"].to_pandas()

print(f"Registros originales: {len(df):,}")


# ============================================================
# 2. CREAR LABEL
# ============================================================

df["label"] = (
    df["Email Type"] == "Phishing Email"
).astype(int)


# ============================================================
# 3. LIMPIAR TEXTO
# ============================================================

df["Email Text"] = df["Email Text"].astype("string")


# ============================================================
# 4. ELIMINAR REGISTROS SIN CONTENIDO ÚTIL
# ============================================================

blank_mask = (
    df["Email Text"]
    .fillna("")
    .str.strip()
    .eq("")
)

empty_mask = (
    df["Email Text"]
    .fillna("")
    .str.strip()
    .str.lower()
    .eq("empty")
)


# ============================================================
# 5. IDENTIFICAR EMAIL ANÓMALO
# ============================================================

df["email_length"] = (
    df["Email Text"]
    .fillna("")
    .str.len()
)

outlier_mask = df["email_length"] > 1_000_000


# ============================================================
# 6. MOSTRAR QUÉ VAMOS A ELIMINAR
# ============================================================

print("\n" + "=" * 60)
print("REGISTROS A ELIMINAR")
print("=" * 60)

print(f"Registros vacíos: {blank_mask.sum():,}")
print(f"Registros 'empty': {empty_mask.sum():,}")
print(f"Registros anómalos: {outlier_mask.sum():,}")

total_to_remove = (
    blank_mask | empty_mask | outlier_mask
).sum()

print(f"\nTotal a eliminar: {total_to_remove:,}")


# ============================================================
# 7. CREAR DATASET LIMPIO
# ============================================================

remove_mask = (
    blank_mask
    | empty_mask
    | outlier_mask
)

df_clean = df.loc[~remove_mask].copy()


# ============================================================
# 8. ELIMINAR COLUMNA AUXILIAR
# ============================================================

df_clean = df_clean.drop(
    columns=["email_length"]
)


# ============================================================
# 9. REINICIAR ÍNDICES
# ============================================================

df_clean = df_clean.reset_index(drop=True)


# ============================================================
# 10. RESULTADOS
# ============================================================

print("\n" + "=" * 60)
print("DATASET LIMPIO")
print("=" * 60)

print(f"Registros restantes: {len(df_clean):,}")

print("\nColumnas:")
print(df_clean.columns.tolist())

print("\nValores nulos:")
print(df_clean.isnull().sum())

print("\nDistribución de clases:")
print(df_clean["label"].value_counts())

print("\nPorcentaje de clases:")
print(
    df_clean["label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nPrimeros registros:")
print(
    df_clean[
        ["Email Text", "Email Type", "label"]
    ].head()
)

# ============================================================
# 11. GUARDAR DATASET LIMPIO
# ============================================================

output_path = "data/processed/phishing_clean.csv"

df_clean.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 60)
print("DATASET GUARDADO")
print("=" * 60)

print(f"Archivo: {output_path}")
print(f"Registros: {len(df_clean):,}")