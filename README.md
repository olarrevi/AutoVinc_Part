# AutoVinc_Part
Este proyecto permite buscar coincidencias entre etiquetas a partir de dos archivos .xlsx usando embeddings generados por el modelo `Sentence-BERT`. El script se encarga de comparar y vincular etiquetas entre un archivo de diccionario y un archivo con datos etiquetados, mostrando los resultados más similares.

## Requisitos

Para ejecutar este proyecto, necesitas tener instalado lo siguiente:

- Python 3.x
- Paquetes de Python:
  - `pandas`
  - `sentence-transformers`
  - `numpy`
  - `tqdm`
  - `re`
  - `unicodedata`
  - `plotly`
  - `scikit-learn`

Puedes instalar las dependencias ejecutando:
```bash
pip install pandas sentence-transformers numpy tqdm plotly scikit-learn
```

## Descripción del script

El script realiza las siguientes acciones:

1. **Carga de datos**: 
   - Carga un archivo de Excel con etiquetas (`Diccionari HAB.xlsx`) y un archivo adicional para comparar (`HAB_columnas_y_labels_2024.xlsx`).
2. **Limpieza y normalización de etiquetas**:
   - Convierte las etiquetas a minúsculas, elimina caracteres especiales, números y acentos.
3. **Generación de embeddings**:
   - Utiliza el modelo `Sentence-BERT` (`paraphrase-MiniLM-L6-v2`) para generar representaciones vectoriales (embeddings) de las etiquetas.
4. **Comparación de similitud**:
   - Encuentra las etiquetas más similares entre los dos conjuntos utilizando la similitud coseno.
5. **Visualización 3D**:
   - Reduce la dimensionalidad de los embeddings y los visualiza en un gráfico 3D utilizando `plotly`.
6. **Actualización del archivo .xlsx**:
   - Agrega las coincidencias más similares al archivo de comparación y guarda el resultado en un nuevo archivo (`arxiu_final_2.xlsx`).

## Uso

Para ejecutar el script, simplemente ejecuta:

```bash
python match_embeddings.py
```

El script realizará las operaciones y mostrará la visualización interactiva. Los resultados actualizados se guardarán en `arxiu_final_2.xlsx`.

## Personalización

Puedes ajustar los siguientes parámetros en el script:

- `file_path_excel`: Ruta del archivo de etiquetas original.
- `file_path_xlsx`: Ruta del archivo con datos para comparar.
- `sheet_name`: Nombre de la hoja del archivo de etiquetas original.
- `output_file`: Nombre del archivo de salida con las coincidencias encontradas.
- `threshold`: Umbral de similitud coseno para considerar dos etiquetas como coincidencias.
- `max_distance`: Penalización por distancia en los índices de las etiquetas.

## Licencia

Este proyecto está licenciado bajo la [Licencia MIT](LICENSE).
