import pandas as pd


# ============================================================
# 1. CARGAR DATASET LIMPIO
# ============================================================

input_path = "data/processed/phishing_clean.csv"

df = pd.read_csv(input_path)

print("=" * 60)
print("AUDITORÍA DE DUPLICADOS")
print("=" * 60)

print(f"Registros totales: {len(df):,}")


# ============================================================
# 2. CONTAR TEXTOS ÚNICOS
# ============================================================

unique_texts = df["Email Text"].nunique()

print(f"Textos únicos: {unique_texts:,}")
print(f"Textos repetidos: {len(df) - unique_texts:,}")


# ============================================================
# 3. FRECUENCIA DE CADA TEXTO
# ============================================================

text_counts = (
    df["Email Text"]
    .value_counts()
)

duplicated_groups = text_counts[text_counts > 1]

print("\n" + "=" * 60)
print("GRUPOS DUPLICADOS")
print("=" * 60)

print(
    f"Grupos de textos duplicados: "
    f"{len(duplicated_groups):,}"
)

print(
    f"Registros pertenecientes a grupos duplicados: "
    f"{duplicated_groups.sum():,}"
)

print(
    f"Mayor cantidad de repeticiones de un mismo texto: "
    f"{text_counts.max():,}"
)


# ============================================================
# 4. BUSCAR ETIQUETAS CONTRADICTORIAS
# ============================================================

label_counts = (
    df.groupby("Email Text")["label"]
    .nunique()
)

conflicting_texts = label_counts[label_counts > 1]

print("\n" + "=" * 60)
print("TEXTOS CON ETIQUETAS CONTRADICTORIAS")
print("=" * 60)

print(
    f"Textos con más de una etiqueta: "
    f"{len(conflicting_texts):,}"
)


# ============================================================
# 5. MOSTRAR CONFLICTOS
# ============================================================

if len(conflicting_texts) > 0:

    print("\nEjemplos de conflictos:")

    conflict_texts = conflicting_texts.index[:10]

    for text in conflict_texts:

        rows = df[
            df["Email Text"] == text
        ]

        print("\n" + "-" * 60)

        print("Texto:")
        print(repr(text[:300]))

        print("\nEtiquetas:")
        print(
            rows[
                ["Email Type", "label"]
            ]
            .value_counts()
        )

else:

    print("\nNo existen textos con etiquetas contradictorias.")


# ============================================================
# 6. DISTRIBUCIÓN DE REPETICIONES
# ============================================================

print("\n" + "=" * 60)
print("DISTRIBUCIÓN DE REPETICIONES")
print("=" * 60)

print(
    text_counts[
        text_counts > 1
    ].describe()
)


# ============================================================
# 7. TOP 10 TEXTOS MÁS REPETIDOS
# ============================================================

print("\n" + "=" * 60)
print("TOP 10 TEXTOS MÁS REPETIDOS")
print("=" * 60)

for text, count in text_counts.head(10).items():

    label_distribution = (
        df.loc[
            df["Email Text"] == text,
            "label"
        ]
        .value_counts()
        .to_dict()
    )

    print("\n" + "-" * 60)
    print(f"Repeticiones: {count}")
    print(f"Labels: {label_distribution}")
    print(f"Texto: {repr(text[:200])}")