# Librerias necesarias para el analisis
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests 
import zipfile
import seaborn as sns
sns.set_style("whitegrid")

# Cargar datos Locales
base_dir = os.path.dirname(__file__)
majors = pd.read_csv(os.path.join(base_dir,"data","data","majors.csv"))
names = pd.read_csv(os.path.join(base_dir,"data","data","names.csv"))

# Cargar datos externos BABYNAMES
data_url = "https://www.ssa.gov/oact/babynames/names.zip"
local_filename = "babynames.zip"

if not os.path.exists(local_filename):
    headers = {
        "User-Agent": "Mozila/5.0"
    }
    r = requests.get(data_url, headers=headers)
    print(r.status_code)  # debug
    r.raise_for_status()
    with open(local_filename, "wb") as f:
        f.write(r.content)


babynames = []
with zipfile.ZipFile(local_filename, "r") as zf:
    data_files = [f for f in zf.filelist if f.filename[-3:] == "txt"]
    def extract_year_from_filename(fn):
        return int (fn[3:7])
    for f in data_files:
        year = extract_year_from_filename(f.filename)
        with zf.open(f) as fp:
            df = pd.read_csv(fp, names=["Name", "Sex", "Count"])
            df["Year"]  = year
            babynames.append(df)
babynames = pd.concat(babynames)


# Verificación inicial
print(f"✓ majors: {len(majors)} registros")
print(f"✓ names: {len(names)} registros")
print(f"✓ babynames: {len(babynames)} registros")

# Estandarizamos los datos
names['Name'] = names['Name'].str.lower()
babynames['Name'] = babynames['Name'].str.lower()   

# Agrupamos por nombre y sexo, sumando conteos históricos
name_sex = babynames.groupby(['Name', 'Sex'])['Count'].sum().unstack(fill_value=0)

# Calcular probabilidad de ser mujer
name_sex['Total'] = name_sex['F'] + name_sex['M']
name_sex['P_Female'] = name_sex['F'] / name_sex['Total']
prob_gender = name_sex[['P_Female']].reset_index()

print(name_sex)
print(prob_gender)
print(f"✓ Probabilidades calculadas para {len(prob_gender)} nombres únicos")

# Unir lo nombres con la probabilidad de genero
students_with_gender = pd.merge(names, prob_gender, on='Name',how='left')

# Unir con majors para tener especialidad + genero estimado
full_data = pd.merge(students_with_gender, majors, left_index=True, right_index=True, how='inner')

print(full_data.head())

missing = full_data['P_Female'].isna().sum()
print(f"Nombres sin datos en babynames: {missing} ({missing/len(full_data)*100:.1f}%)")

full_data_clean = full_data.dropna(subset=['P_Female']).copy()
print(f"✓ Registros válidos para análisis: {len(full_data_clean)}")

# (umbral ajustable: 0.5 = neutro, 0.7 = más conservador, 0.9 = muy estricto)
UMBRAL_FEMALE = 0.7
full_data_clean['Estimated_Gender'] = full_data_clean['P_Female'].apply(
    lambda x: 'Female' if x >= UMBRAL_FEMALE else ('Male' if x <= (1-UMBRAL_FEMALE) else 'Unknown')
)

# Filtrar solo casos con estimación clara
print(full_data_clean[full_data_clean['Estimated_Gender']=='Unknown'])
data_filtered = full_data_clean[full_data_clean['Estimated_Gender'] != 'Unknown'].copy()
print(f"✓ Estudiantes con género estimado claro: {len(data_filtered)}")

# Contar especialidades por género estimado
major_by_gender = data_filtered.groupby(['Majors', 'Estimated_Gender']).size().unstack(fill_value=0)
print(major_by_gender)

# Top 10 especialidades más populares para mujeres
top_female = major_by_gender.nlargest(10, 'Female')
print("\n Top 10 especialidades para mujeres (estimado)")
print(top_female['Female'])

# Top 10 especialidades más populares para hombres
top_male = major_by_gender.nlargest(10, 'Male')
print("\n Top 10 especialidades para hombres (estimado)")
print(top_male['Male'])

# Grafico Comparativo
fig, axes = plt.subplots(1,2 , figsize=(14,6))
top5_female = major_by_gender.nlargest(5,'Female')['Female']
axes[0].barh(top5_female.index, top5_female.values, color='#FF69B4',edgecolor='black')
axes[0].set_xlabel('Cantidad de estudiantes')
axes[0].set_title('Especialidades mas populares - Mujeres')
axes[0].invert_yaxis()

top5_male = major_by_gender.nlargest(5, 'Male')['Male']
axes[1].barh(top5_male.index, top5_male.values, color='#4169E1', edgecolor='black')
axes[1].set_xlabel('Cantidad de estudiantes')
axes[1].set_title('Especialidades más populares - Hombres')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()



