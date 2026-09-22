from datasets import load_dataset
import pandas as pd

# ============================================================
# 1. CARGAR DATASET
# ============================================================

print("Cargando dataset...")

dataset = load_dataset("zefang-liu/phishing-email-dataset")

df = dataset["train"].to_pandas()

print(f"Dataset cargado: {df.shape}")
print()


# ============================================================
# 2. CREAR LABEL
# ============================================================

df["label"] = (
    df["Email Type"] == "Phishing Email"
).astype(int)


# ============================================================
# 3. ANALIZAR TEXTOS VACÍOS O NULOS
# ============================================================

print("=" * 60)
print("1. ANÁLISIS DE TEXTOS VACÍOS")
print("=" * 60)

df["Email Text Clean"] = (
    df["Email Text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

empty_mask = df["Email Text Clean"].eq("")

empty_rows = df[empty_mask]

print(f"Registros sin texto útil: {len(empty_rows)}")
print()

if len(empty_rows) > 0:

    print("Distribución por clase:")
    print(empty_rows["Email Type"].value_counts())
    print()

    print("IDs de los registros:")
    print(
        empty_rows[
            ["Unnamed: 0", "Email Type", "label"]
        ].to_string(index=False)
    )

print()


# ============================================================
# 4. LONGITUD DE LOS CORREOS
# ============================================================

print("=" * 60)
print("2. ANÁLISIS DE LONGITUD")
print("=" * 60)

df["email_length"] = df["Email Text Clean"].str.len()

print("Estadísticas generales:")
print(df["email_length"].describe())

print()

percentiles = df["email_length"].quantile(
    [0.90, 0.95, 0.99, 0.995, 0.999]
)

print("Percentiles:")
print(percentiles)

print()


# ============================================================
# 5. CORREOS MÁS LARGOS
# ============================================================

print("=" * 60)
print("3. CORREOS MÁS LARGOS")
print("=" * 60)

longest = (
    df.sort_values(
        "email_length",
        ascending=False
    )
    .head(10)
)

print(
    longest[
        [
            "Unnamed: 0",
            "Email Type",
            "label",
            "email_length"
        ]
    ].to_string(index=False)
)

print()


# ============================================================
# 6. PREVIEW DEL CORREO MÁS LARGO
# ============================================================

print("=" * 60)
print("4. PREVIEW DEL CORREO MÁS LARGO")
print("=" * 60)

longest_index = df["email_length"].idxmax()

longest_email = df.loc[
    longest_index,
    "Email Text Clean"
]

print(f"ID: {df.loc[longest_index, 'Unnamed: 0']}")
print(f"Tipo: {df.loc[longest_index, 'Email Type']}")
print(f"Label: {df.loc[longest_index, 'label']}")
print(f"Longitud: {len(longest_email):,} caracteres")
print()

print("--- PRIMEROS 500 CARACTERES ---")
print(longest_email[:500])

print()
print("--- ÚLTIMOS 500 CARACTERES ---")
print(longest_email[-500:])

print()

# ============================================================
# 7. ANALIZAR TEXTOS DUPLICADOS
# ============================================================

print("=" * 60)
print("5. ANÁLISIS DE DUPLICADOS")
print("=" * 60)

text_counts = df["Email Text Clean"].value_counts()

duplicated_texts = text_counts[
    text_counts > 1
]

print(
    f"Textos diferentes que aparecen más de una vez: "
    f"{len(duplicated_texts):,}"
)

print(
    f"Registros pertenecientes a esos grupos duplicados: "
    f"{duplicated_texts.sum():,}"
)

print()

print("Frecuencia de los textos duplicados:")
print(duplicated_texts.describe())

print()


# ============================================================
# 8. CONSISTENCIA DE LABEL
# ============================================================

print("=" * 60)
print("6. CONSISTENCIA DE LABEL EN DUPLICADOS")
print("=" * 60)

duplicate_groups = (
    df[
        df["Email Text Clean"].isin(
            duplicated_texts.index
        )
    ]
    .groupby("Email Text Clean")["label"]
    .nunique()
)

consistent_groups = (
    duplicate_groups == 1
).sum()

conflicting_groups = (
    duplicate_groups > 1
).sum()

print(
    f"Grupos duplicados con la MISMA label: "
    f"{consistent_groups:,}"
)

print(
    f"Grupos duplicados con labels DIFERENTES: "
    f"{conflicting_groups:,}"
)

print()


# ============================================================
# 9. ENCONTRAR EL ÚNICO DUPLICADO CONFLICTIVO
# ============================================================

print("=" * 60)
print("7. DUPLICADO CONFLICTIVO")
print("=" * 60)

conflicting_texts = duplicate_groups[
    duplicate_groups > 1
].index

print(
    f"Textos conflictivos encontrados: "
    f"{len(conflicting_texts)}"
)

print()

for text in conflicting_texts:

    conflict_records = df[
        df["Email Text Clean"] == text
    ][
        [
            "Unnamed: 0",
            "Email Type",
            "label",
            "email_length"
        ]
    ]

    print("--- REGISTROS CONFLICTIVOS ---")
    print(conflict_records.to_string(index=False))

    print()
    print("--- TEXTO ---")
    print(text[:2000])

    print()
    print("-" * 60)


# ============================================================
# 10. RESUMEN
# ============================================================

print()
print("=" * 60)
print("RESUMEN DEL ANÁLISIS")
print("=" * 60)

print(f"Total de registros: {len(df):,}")
print(f"Textos vacíos: {empty_mask.sum():,}")
print(f"Textos duplicados: {duplicated_texts.sum():,}")
print(f"Grupos duplicados: {len(duplicated_texts):,}")
print(f"Duplicados consistentes: {consistent_groups:,}")
print(f"Duplicados conflictivos: {conflicting_groups:,}")
print(
    f"Correo más largo: "
    f"{df['email_length'].max():,} caracteres"
)

print()
print("Análisis terminado.")
print("NO se eliminó ni modificó ningún registro.")