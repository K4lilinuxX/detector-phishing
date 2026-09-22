# ============================================================
# PHISHING DETECTOR
# Etapa 1 - Carga y exploración del dataset
# ============================================================

# -------------------------
# 1. Importar dependencias
# -------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset


# -------------------------
# 2. Cargar dataset
# -------------------------

print("=" * 60)
print("CARGANDO DATASET")
print("=" * 60)

dataset = load_dataset("zefang-liu/phishing-email-dataset")

print("\nDataset cargado correctamente.")
print(dataset)


# -------------------------
# 3. Revisar los splits
# -------------------------

print("\n" + "=" * 60)
print("SPLITS DISPONIBLES")
print("=" * 60)

print(dataset)


# ------------------------------------------------
# 4. Seleccionar el split de entrenamiento
# ------------------------------------------------

# El dataset normalmente contiene un split llamado "train".
# No asumimos todavía que sea el único split.
# Primero comprobamos cuáles existen.

print("\nSplits disponibles:")

for split_name in dataset.keys():
    print(f"- {split_name}")


# Utilizamos train si está disponible.
if "train" in dataset:
    df = dataset["train"].to_pandas()
else:
    # Si el nombre fuera diferente, utilizamos el primer split
    # disponible para evitar asumir una estructura que no existe.
    first_split = list(dataset.keys())[0]
    df = dataset[first_split].to_pandas()

    print(
        f"\nNo se encontró un split llamado 'train'. "
        f"Se utilizará: '{first_split}'"
    )


# -------------------------
# 5. Información general
# -------------------------

print("\n" + "=" * 60)
print("INFORMACIÓN GENERAL")
print("=" * 60)

print(f"\nNúmero de registros: {df.shape[0]}")
print(f"Número de columnas: {df.shape[1]}")

print("\nShape:")
print(df.shape)

print("\nColumnas:")
for column in df.columns:
    print(f"- {column}")


# -------------------------
# 6. Tipos de datos
# -------------------------

print("\n" + "=" * 60)
print("TIPOS DE DATOS")
print("=" * 60)

print(df.dtypes)


# -------------------------
# 7. Valores nulos
# -------------------------

print("\n" + "=" * 60)
print("VALORES NULOS")
print("=" * 60)

null_values = df.isnull().sum()

print(null_values)

print("\nTotal de valores nulos:")
print(null_values.sum())


# -------------------------
# 8. Primeros registros
# -------------------------

print("\n" + "=" * 60)
print("PRIMEROS REGISTROS")
print("=" * 60)

print(df.head())


# -------------------------
# 9. Distribución de clases
# -------------------------

print("\n" + "=" * 60)
print("DISTRIBUCIÓN DE CLASES")
print("=" * 60)

# Comprobamos primero si las columnas existen.
if "label" in df.columns:

    label_counts = df["label"].value_counts().sort_index()

    print("\nCantidad de registros por label:")
    print(label_counts)

    print("\nPorcentaje por clase:")
    print(
        df["label"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    # Gráfica
    plt.figure(figsize=(7, 5))

    label_counts.plot(kind="bar")

    plt.title("Distribución de etiquetas")
    plt.xlabel("Label")
    plt.ylabel("Número de correos")
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.show()

else:
    print(
        "\nLa columna 'label' no está presente en el dataset."
    )


# -------------------------
# 10. Tipos de correo
# -------------------------

print("\n" + "=" * 60)
print("DISTRIBUCIÓN DE EMAIL TYPE")
print("=" * 60)

if "Email Type" in df.columns:

    print(df["Email Type"].value_counts())

else:
    print(
        "\nLa columna 'Email Type' no está presente."
    )


# -------------------------
# 11. Ejemplos de correos
# -------------------------

print("\n" + "=" * 60)
print("EJEMPLOS DE CORREOS")
print("=" * 60)

if "Email Text" in df.columns:

    examples = df.sample(
        n=min(5, len(df)),
        random_state=42
    )

    for index, row in examples.iterrows():

        print("\n" + "-" * 60)

        print(f"Índice: {index}")

        if "label" in df.columns:
            print(f"Label: {row['label']}")

        if "Email Type" in df.columns:
            print(f"Email Type: {row['Email Type']}")

        print("\nContenido del correo:")

        print(row["Email Text"])

else:
    print(
        "\nLa columna 'Email Text' no está presente."
    )


# -------------------------
# 12. Información final
# -------------------------

print("\n" + "=" * 60)
print("EXPLORACIÓN TERMINADA")
print("=" * 60)

print(
    """
La primera etapa de exploración ha terminado.

En esta etapa solamente analizamos la estructura
del dataset y no modificamos los datos.

Los siguientes pasos serán:

1. Analizar la calidad del texto.
2. Revisar valores faltantes.
3. Analizar posibles duplicados.
4. Preparar el texto.
5. Dividir entrenamiento/prueba.
6. Aplicar TF-IDF.
7. Entrenar Logistic Regression.
"""
)