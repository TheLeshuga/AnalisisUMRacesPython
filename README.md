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

df_filtered.head(5)
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

## Fase de análisis

### Primer insight

Para comparar las velocidades medias entre hombres y mujeres en los dos tipos de maratón, primero analizamos la distribución de participantes por género y por tipo de maratón. En el conjunto de datos, el 67% de los corredores son hombres y el 33% mujeres. Además, el 18% de las mujeres y el 23% de los hombres han competido en la maratón de 50 millas (50mi). En términos generales, el 78% de los registros corresponden a la maratón de 50km, mientras que el 22% restante pertenecen a la de 50mi.

Esta distribución revela una marcada disparidad tanto en la representación por género como en la cantidad de datos recogidos por tipo de maratón. Aunque esto puede ser esperable, dado que menos personas participan en la maratón de 50 millas y menos mujeres se inscriben en maratones en general, es importante minimizar posibles sesgos de cobertura. En caso de utilizar estos datos para entrenar un modelo predictivo con Machine Learning, sería recomendable aplicar una ponderación que equilibre la representación de género y aumente el peso de los datos de la maratón de 50 millas frente a los de la de 50 km.


<p align="center">
  <img src="https://github.com/user-attachments/assets/c982d5d0-06d2-440f-ba08-0e9e887b8771" width="450">
  <img src="https://github.com/user-attachments/assets/497714a4-22ca-4b8a-aa5a-345b065715d2" width="450">
</p>

Usando un modelo lineal, analizamos ahora la diferencia en las velocidades medias considerando la edad y el género. Las líneas que representan las velocidades medias para cada género siguen una pendiente descendente: comienzan con valores altos y disminuyen progresivamente. Esto es coherente, ya que a menor edad, la estamina y la resistencia física suelen ser mayores, mientras que en edades avanzadas estas capacidades se van empeorando.  

A pesar de esta tendencia, la diferencia entre los valores extremos de edad es de apenas 2 km/h, lo que indica que el impacto en la media no es lo suficientemente significativo como para sesgarla hacia valores extremadamente altos o bajos.

<p align="center">
  <img src="https://github.com/user-attachments/assets/51d74816-ab4d-4c3d-b388-dc003f82345d" width="450">
</p>

Agrupando los datos por distancia de maratón y género, calculamos la media y determinamos los valores mínimo y máximo para analizar los extremos y verificar que no introducen sesgos en los datos. Además, generamos otra tabla con la mediana para comparar su diferencia con la media. En todos los casos, la discrepancia entre media y mediana es mínima, de apenas 0.2 km/h, lo que indica que los valores extremos no están influyendo significativamente en los resultados.  

Asimismo, observamos que la diferencia en la velocidad media entre hombres y mujeres es del 9% en la maratón de 50 km, mientras que en la de 50mi es ligeramente menor, con un 6%.  

Esta diferencia es comprensible, ya que participar en una maratón requiere un alto nivel de entrenamiento y capacidad física para completar la carrera. En el caso de la maratón de 50 millas, la exigencia es aún mayor, lo que reduce la brecha de velocidad entre géneros, ya que solo quienes cuentan con la preparación necesaria logran finalizarla, independientemente de su género.

<p align="center">
  <img src="https://github.com/user-attachments/assets/48e913b2-8f43-49fb-be26-02b333504d08" width="400">
  <img src="https://github.com/user-attachments/assets/5689a704-4ddb-4772-b7c2-f3e15c5db5f8" width="280">
</p>

### Segundo insight




