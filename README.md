# Análisis y preparación de datos sobre las maratones de 2020 en EEUU usando Python (Pandas, Matplotlib y Seaborn)
Desarrollo de un proyecto en Google Colab utilizando Python y librerías especializadas en análisis y limpieza de datos, como Pandas, Matplotlib y Seaborn. El proceso incluye la limpieza y transformación de los datos para garantizar su calidad, trabajando exclusivamente con los campos necesarios para obtener información relevante. Asimismo, se preparan los datasets finales para su uso en el entrenamiento de modelos de Machine Learning. Durante el análisis, se emplean herramientas visuales como histogramas, mapas de calor y violin plots, entre otras, para extraer insights significativos.

![image](https://github.com/user-attachments/assets/d9163db1-d7c8-482f-9ebd-0241d348beb5)


## Resumen
Se hace un procesamiento exhaustivo de datos usando Python para poder responder a las siguientes preguntas:

- Diferencia de velocidad en las carreras "50km" y "50mi" entre mujeres y hombres.
- ¿Qué edades son las mejores y peores en la carrera "50mi"? (Solo personas mayores de 19 años).
- ¿Son los corredores más lentos en verano que en invierno? Ver diferencia entre estaciones.

Además, se realizan diversas observaciones con el fin de elaborar un informe analítico sobre las variables, estableciendo así una base de conocimiento y obteniendo datos procesados que servirán para la construcción de modelos predictivos en Machine Learning.

## Dataset usado

El dataset usado en el proyecto se encuentra público en la página kaggle: https://www.kaggle.com/datasets/fatihyavuzz/two-centuries-of-um-races

## Conclusiones generales

## Trabajo realizado

Tras cargar los datos mediante la librería Pandas, se genera un reporte del perfil del dataset para comprobar distinta información descriptiva de la misma. Con dtypes vemos el tipo de cada campo.

```python
pp.ProfileReport(df, explorative=True)

df.dtypes
```


<p align="center">
  <img src="https://github.com/user-attachments/assets/03a74995-9885-48c9-b5ab-e20e5f608498" width="750">
  <img src="https://github.com/user-attachments/assets/d063286d-9c0e-4f8e-9976-9e3622a5db55" width="200">
</p>


