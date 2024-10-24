# -*- coding: latin-1 -*-
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from tqdm import tqdm
import re
import unicodedata
import plotly.express as px
from sklearn.decomposition import PCA

file_path_excel = r'HAB\Diccionari HAB.xlsx' 
file_path_xlsx = r'HAB_columnas_y_labels_2024.xlsx'
sheet_name = 'HAB'
output_file = 'arxiu_final_2.xlsx'

# Cargar el modelo preentrenado de Sentence-BERT
print("Cargando el modelo preentrenado de Sentence-BERT...")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Función para limpiar y normalizar las etiquetas
def clean_label(label):
    if not isinstance(label, str):
        label = str(label)  # Convertir a cadena si no lo es
    label = label.lower().strip()  # Convertir a minúsculas y eliminar espacios en blanco extra

    # Eliminar números en los primeros 5 caracteres
    first_five = label[:5]
    label = re.sub(r'\d', '', first_five) + label[5:]

    # Eliminar números que empiezan con "20"
    label = re.sub(r'\b20\d*\b', '', label)

    label = re.sub(r'\s+', ' ', label)  # Reemplazar múltiples espacios por un solo espacio
    label = re.sub(r'[^\w\s]', '', label)  # Eliminar caracteres especiales
    label = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode('utf-8', 'ignore')  # Eliminar acentos
    return label

# Cargar el archivo Excel original
print("Cargando el archivo Excel original...")

excel_data = pd.ExcelFile(file_path_excel)

# Cargar la hoja de datos específica, limitando el rango a A2:K3281
df_excel = excel_data.parse(sheet_name)
print("Archivo Excel original cargado. Filas y columnas:", df_excel.shape)

# Cargar el segundo archivo .xlsx
print("Cargando el archivo .xlsx para comparar...")

df_xlsx = pd.read_excel(file_path_xlsx)
print("Archivo .xlsx cargado. Filas y columnas:", df_xlsx.shape)

# Seleccionar las columnas de etiquetas en el Excel original
label_columns = [col for col in df_excel.columns if 'label' in col][::-1]
print(f"Columnas de etiquetas seleccionadas en el archivo Excel original: {label_columns}")

# Limpiar y normalizar las etiquetas
print("Limpiando las etiquetas de ambos archivos...")
df_labels_excel = df_excel[label_columns].apply(lambda col: col.map(clean_label)).fillna('')
df_xlsx['label'] = df_xlsx['label'].apply(clean_label)

# Generar embeddings para las etiquetas del Excel original
print("Generando los embeddings para las etiquetas del Excel original...")
embeddings_excel = {}
for col in label_columns:
    embeddings_excel[col] = model.encode(df_labels_excel[col].tolist(), convert_to_tensor=True)

# Generar embeddings para las etiquetas del archivo .xlsx
print("Generando los embeddings para las etiquetas del archivo .xlsx...")
embeddings_xlsx = model.encode(df_xlsx['label'].tolist(), convert_to_tensor=True)

# Función para visualizar los embeddings del Excel original y el archivo .xlsx juntos en 3D, con etiquetas
def plot_combined_embeddings_3d(embeddings_excel, embeddings_xlsx, df_labels_excel, df_xlsx, title):
    # Combinar todos los embeddings en una sola matriz
    all_embeddings_excel = np.vstack([emb for emb in embeddings_excel.values()])
    all_embeddings_xlsx = embeddings_xlsx.numpy()

    # Concatenar los embeddings para visualizarlos juntos
    combined_embeddings = np.vstack([all_embeddings_excel, all_embeddings_xlsx])

    # Reducir la dimensionalidad a 3D usando PCA
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(combined_embeddings)

    # Crear un DataFrame para los embeddings reducidos
    df_embeddings = pd.DataFrame(reduced_embeddings, columns=['PC1', 'PC2', 'PC3'])
    
    # Añadir información de origen de los embeddings
    num_excel_embeddings = all_embeddings_excel.shape[0]
    df_embeddings['source'] = ['Excel'] * num_excel_embeddings + ['XLSX'] * all_embeddings_xlsx.shape[0]

    # Añadir las etiquetas correspondientes
    labels_excel = np.concatenate([df_labels_excel[col].values for col in embeddings_excel.keys()])
    labels_xlsx = df_xlsx['label'].values
    df_embeddings['label'] = np.concatenate([labels_excel, labels_xlsx])
    
    # Visualización interactiva usando plotly, con etiquetas mostradas en el hover
    fig = px.scatter_3d(df_embeddings, x='PC1', y='PC2', z='PC3', color='source', hover_data=['label'], title=title)
    fig.show()

# Visualizar los embeddings del Excel original y el archivo .xlsx juntos en 3D, con etiquetas
plot_combined_embeddings_3d(embeddings_excel, embeddings_xlsx, df_labels_excel, df_xlsx, "Comparación de Embeddings entre Excel Original y XLSX (3D)")

# Comparar y encontrar las etiquetas más similares con penalización por distancia
def find_and_update_labels_xlsx(embeddings_excel, embeddings_xlsx, threshold, max_distance=10):
    print("Buscando etiquetas similares y actualizando el archivo .xlsx...")
    similar_pairs = {}
    num_labels_excel = embeddings_excel[next(iter(embeddings_excel))].shape[0]
    num_labels_xlsx = embeddings_xlsx.shape[0]
    total_vinculaciones = 0
    duplicados_codi = 0

    # Barra de progreso para las columnas del Excel original
    for col, emb_excel in tqdm(embeddings_excel.items(), desc="Comparando columnas del Excel original"):
        cosine_scores = util.pytorch_cos_sim(emb_excel, embeddings_xlsx)
        
        # Barra de progreso para las etiquetas del Excel original
        for index_excel in tqdm(range(num_labels_excel), desc="Comparando etiquetas del Excel original", leave=False):
            
            # Barra de progreso para las etiquetas del archivo .xlsx
            for index_xlsx in tqdm(range(num_labels_xlsx), desc="Comparando etiquetas del archivo .xlsx", leave=False):
                # Calcular la penalización por distancia
                distance_penalty = abs(index_excel - index_xlsx) / max(num_labels_excel, num_labels_xlsx)
                adjusted_score = cosine_scores[index_excel, index_xlsx].item() #- distance_penalty

                if adjusted_score >= threshold:
                    label_excel = df_labels_excel[col].iloc[index_excel]
                    label_xlsx = df_xlsx['label'].iloc[index_xlsx]
                    variable_xlsx = df_xlsx.iloc[index_xlsx]['variable']
                    codi_excel = df_excel['codi'].iloc[index_excel]

                    # Verificar si el codi ya existe en el similar_pairs
                    if codi_excel in similar_pairs:
                        duplicados_codi += 1
                        # Si ya existe, reemplazar solo si la nueva similitud es mayor
                        if adjusted_score > similar_pairs[codi_excel]['similarity']:
                            similar_pairs[codi_excel] = {
                                'col': col,
                                'label_excel': label_excel,
                                'variable_xlsx': variable_xlsx,
                                'codi_excel': codi_excel,
                                'similarity': adjusted_score,
                                'index_xlsx': index_xlsx
                            }
                            
                    else:
                        # Si no existe, añadirlo al similar_pairs
                        similar_pairs[codi_excel] = {
                            'col': col,
                            'label_excel': label_excel,
                            'variable_xlsx': variable_xlsx,
                            'codi_excel': codi_excel,
                            'similarity': adjusted_score,
                            'index_xlsx': index_xlsx
                        }
                        total_vinculaciones += 1
                        

    # Actualizar el DataFrame con los resultados
    for codi_excel, data in (similar_pairs.items()):
        df_xlsx.at[data['index_xlsx'], 'matched_label_excel'] = data['label_excel']
        df_xlsx.at[data['index_xlsx'], 'similarity'] = data['similarity']
        df_xlsx.at[data['index_xlsx'], 'codi'] = data['codi_excel']

    stats_df = pd.DataFrame({
       'Estadística': ['Total de vinculaciones', 'Número de duplicados de codi'],
       'Valor': [total_vinculaciones, duplicados_codi]
    })
    print(stats_df)
    print("Actualización del archivo .xlsx completada.")
    return similar_pairs

# Buscar etiquetas similares y actualizar el archivo .xlsx
similar_labels_xlsx = find_and_update_labels_xlsx(embeddings_excel, embeddings_xlsx, threshold=0.7, max_distance=10)

# Guardar el DataFrame actualizado en un nuevo archivo .xlsx
print(f"Guardando el archivo actualizado en {output_file}...")
df_xlsx.to_excel(output_file, index=False)
print("Archivo .xlsx guardado.")
