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

### Carga de datos y perfil de datos

Tras cargar los datos mediante la librería Pandas, se genera un reporte del perfil del dataset para comprobar distinta información descriptiva de la misma. Con dtypes vemos el tipo de cada campo.

```python
pp.ProfileReport(df, explorative=True)

df.dtypes
```


<p align="center">
  <img src="https://github.com/user-attachments/assets/03a74995-9885-48c9-b5ab-e20e5f608498" width="750">
  <img src="https://github.com/user-attachments/assets/d063286d-9c0e-4f8e-9976-9e3622a5db55" width="200">
</p>

En primera instancia, podemos observar que tenemos 13 campos de los cuales 4 de ellos son totalmente innecesarios para cumplir nuestro objetivo ('Year of event', 'Athlete club', 'Athlete country' y 'Athlete age category') y 1 de ellos será usado para crear una columna nueva de mayor utlidad ('Athlete year of birth'). Entre los tipos de los campos, 'Athelete average speed' será cambiado para poder usarlo como variable numérica más adelante.

- Year of event: Nos será redundante, ya sabemos que todo el dataset es de maratones llevadas a cabo durante el año 2020.
- Athlete club: Es irrelevante para este estudio. Nos queremos fijar más en los aspectos de rendimiento y biológicos de los atletas en cada carrera.
- Athlete country: Aunque también podríamos tener en cuenta el país de procedencia, para este primer análisis no queremos expandir por esta dirección.
- Athlete age category: Sí queremos saber la edad de los atletas, pero como aquí sólo nos dan un rango de edad, calcularemos la edad real del atleta para ser más precisos.

En el reporte también vemos que sólo un 4.1% de los valores son nulos. Este porcentaje es bastante reducido, pero realmente queremos saber cuáles son nulos dentro de nuestro sector de datos que cumpla ser carreras de 2020 y en EEUU. Los campos que tienen mayor volumen de valores nulos son 'Athlete club', el cual no será relevante y 'Athlete year of birth' con un 7.9%. 

### Limpieza de datos

Empezamos reduciendo el número de registros filtrando sólo aquellos que nos interesan mediante condicionales:

```python
df_filtered = df[(df['Event distance/length'].isin(['50mi', '50km'])) & (df['Year of event'] == 2020) & (df['Event name'].str.split('(').str.get(1).str.split(')').str.get(0) == 'USA')]

num_filas, num_columnas = df_filtered.shape
print(f"Número de columnas: {num_columnas}. Número de registros: {num_filas}")

> Número de columnas: 13. Número de registros: 26090
```
De esta forma, hemos reducido el dataset a un 0.35% de los registros. Haciendo el conjunto más razonable para no hacer entrenamientos excesivamente largos. Como nos quedamos con todas las maratones de EEUU, entonces podemos quitar la cadena "(USA)" de 'Event name':

```python
df_filtered['Event name'] = df_filtered['Event name'].str.split('(').str.get(0)
```

El año de los atletas nos aportará mayor información en comparación a las columnas 'Athlete year of birth' y 'Athlete age category'. Por lo tanto, calculamos la edad aproximada de cada atleta sabiendo que las maratones son de 2020 y teniendo el año de nacimiento:

```python
df_filtered['Athlete age'] = 2020 - df_filtered['Athlete year of birth']
```

Al igual que eliminamos la cadena "(USA)" de 'Event name', ahora quitamos la cadena "h" de 'Athlete average speed' para transformarla a una variable numérica. Por último, acabamos eliminando las columnas innecesarias previamente mencionadas:

```python
df_filtered['Athlete performance'] = df_filtered['Athlete performance'].str.split(' ').str.get(0)

df_filtered = df_filtered.drop(['Year of event', 'Athlete club', 'Athlete country', 'Athlete age category', 'Athlete year of birth'], axis=1)
```

![image](https://github.com/user-attachments/assets/946a1279-21f4-4255-8668-d2b5c1eafb86)

Comprobamos de nuevo los tipos de cada campo que ha quedado en el nuevo dataframe. 'Athlete age' se muestra como float y 'Athlete average speed' como objeto, así que cambiamos los tipos. Se pone el parámetro de "errors" con "ignore", ya que, aún quedan valores nulos en estos campos:

```python
df_filtered['Athlete age'] = df_filtered['Athlete age'].astype(int, errors="ignore")
df_filtered['Athlete average speed'] = df_filtered['Athlete average speed'].astype(float, errors="ignore")
```

Tras haber hecho la reducción de registros y campos a los que nos interesan, volvemos a comprobar el número de valores nulos que quedan en el dataframe y en qué columnas se encuentran:

```python
df_filtered.isna().sum()
```
<p align="left">
  <img src="https://github.com/user-attachments/assets/1a70197c-f64c-4ebd-8dff-97f2c57df9f3" width="200">
</p>

Como el número de nulos restantes sólo representa un 0.9% de nuestros registros, simplemente se eliminan estos:

```python
df_clean = df_filtered.dropna()

num_filas, num_columnas = df_clean.shape
print(f"Número de columnas: {num_columnas}. Número de registros: {num_filas}")

> Número de columnas: 9. Número de registros: 25857
```

Se comprueba si hay registros duplicados con la siguiente llamada, pero no nos devuelve nada, así que nuestro dataframe está libre de duplicados:

```python
df_clean[df_clean.duplicated()]
```

![image](https://github.com/user-attachments/assets/2e076895-64ba-494b-8bd9-0a167b256c9e)

Ahora que contamos con nuestro dataframe final, realizaremos los últimos ajustes modificando los nombres de los campos para cumplir con un convenio, reorganizando el orden de las columnas y restableciendo el índice de los registros para desvincularlo del dataset original:

```python
df_renamed = df_clean.rename(columns = {
    'Event dates' : 'event_date',
    'Event name' : 'event_name',
    'Event distance/length' : 'race_distance',
    'Event number of finishers' : 'race_number_finishers',
    'Athlete performance' : 'athlete_performance',
    'Athlete gender' : 'athlete_gender',
    'Athlete average speed' : 'athlete_avg_speed',
    'Athlete ID' : 'athlete_ID',
    'Athlete age' : 'athlete_age'

})

df_renamed.reset_index()

df_renamed = df_renamed[['event_date', 'event_name', 'race_distance', 'race_number_finishers', 'athlete_ID', 'athlete_gender', 'athlete_age', 'athlete_performance', 'athlete_avg_speed']]

df_renamed.head(5)
```

![image](https://github.com/user-attachments/assets/75a56d7b-2f53-4eef-88fa-8c462cce155c)

De esta forma, ya tendríamos finalizado el primer dataset limpio y transformado llamada 'UM_RACES_CLEANED.csv'.

## Exploración de los datos




