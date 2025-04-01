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
  <img src="https://github.com/user-attachments/assets/51d74816-ab4d-4c3d-b388-dc003f82345d" width="500">
</p>

Agrupando los datos por distancia de maratón y género, calculamos la media y determinamos los valores mínimo y máximo para analizar los extremos y verificar que no introducen sesgos en los datos. Además, generamos otra tabla con la mediana para comparar su diferencia con la media. En todos los casos, la discrepancia entre media y mediana es mínima, de apenas 0.2 km/h, lo que indica que los valores extremos no están influyendo significativamente en los resultados.  

Asimismo, observamos que la diferencia en la velocidad media entre hombres y mujeres es del 9% en la maratón de 50 km, mientras que en la de 50mi es ligeramente menor, con un 6%.  

Esta diferencia es comprensible, ya que participar en una maratón requiere un alto nivel de entrenamiento y capacidad física para completar la carrera. En el caso de la maratón de 50 millas, la exigencia es aún mayor, lo que reduce la brecha de velocidad entre géneros, ya que solo quienes cuentan con la preparación necesaria logran finalizarla, independientemente de su género.

<p align="center">
  <img src="https://github.com/user-attachments/assets/48e913b2-8f43-49fb-be26-02b333504d08" width="400">
  <img src="https://github.com/user-attachments/assets/5689a704-4ddb-4772-b7c2-f3e15c5db5f8" width="280">
</p>

Para completar el análisis de las relaciones y diferencias entre los atletas y el tipo de maratón, generamos un gráfico de violín para visualizar la distribución de los cuartiles. En la maratón de 50km, la diferencia entre géneros es más pronunciada: el 50% de los hombres (Q25 - Q75) mantiene velocidades entre 7 km/h y 8.5 km/h, mientras que en las mujeres este rango se sitúa aproximadamente entre 6.5 km/h y 8 km/h.  

En la maratón de 50 millas, el 50% de los hombres registra velocidades entre 7 km/h y 8 km/h, mientras que en las mujeres el rango es de 6.2 km/h a 7.5 km/h. En ambas distancias, la diferencia de velocidad en el 50% central de los atletas es similar, con una variación de entre 0.5 y 0.8 km/h. Este patrón sugiere que en la maratón de 50km los corredores pueden mantener una velocidad media superior en comparación con la de 50mi, probablemente debido a la menor distancia. En consecuencia, tanto hombres como mujeres muestran velocidades más altas en la maratón de 50km, aunque en el caso de las mujeres la diferencia es menos marcada.

<p align="center">
  <img src="https://github.com/user-attachments/assets/97debf1b-92e5-48ef-a840-ece083397732" width="500">
</p>

### Segundo insight



### Tercer insight

Para conocer la diferencia de velocidades de los atletas según las estaciones, primero deberemos crear una variable categórica de estación del año. Al tener la fecha del evento, podemos sacar el mes y después asignarle la variable categórica de la estación:

```python
df_renamed['race_month'] = df_renamed['event_date'].astype(str).str.split('.').str[1].astype(int)

df_renamed['race_season'] = df_renamed['race_month'].apply(lambda x : "Winter" if x > 11 else "Fall" if x > 8 else "Summer" if x > 5 else "Spring" if x > 2 else "Winter")

df_renamed.head(8)
```

![image](https://github.com/user-attachments/assets/988bfbcd-15c5-4593-8371-f7c25c3c309b)

Generamos un gráfico de violín para visualizar la distribución de las velocidades medias por estación, analizar los cuartiles y comparar de forma clara las estaciones de invierno con verano y primavera con otoño. A simple vista, se observa que otoño e invierno presentan una distribución muy similar, mientras que en verano y primavera el pico de la distribución se inclina hacia valores más bajos. En cuanto a los cuartiles (Q25 - Q75), en la mayoría de las estaciones las velocidades oscilan entre aproximadamente 6 km/h y 8 km/h. Sin embargo, en verano, el rango se desplaza ligeramente hacia valores más bajos, situándose entre 5.4 km/h y 7.7 km/h.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8f68c2e7-2650-4321-b621-97f29ac5ffdb" width="500">
</p>

Si analizamos la moda, es decir, el valor de velocidad media más frecuente en cada estación, observamos que los valores más altos corresponden a otoño e invierno, seguidos de primavera y verano. La diferencia entre otoño e invierno es mínima, de apenas 0.4 km/h, mientras que de invierno a primavera aumenta casi a 1 km/h. Estos datos confirman que existe una diferencia en las velocidades medias según la estación, especialmente entre las estaciones cálidas y frías.  

Al observar la media, primavera se posiciona como la estación con la velocidad media más alta, seguida de invierno. De nuevo, se evidencia una notable diferencia entre verano y el resto de estaciones. Aunque la diferencia entre la media de invierno y la de otoño es de apenas 0.1 km/h, entre otoño e verano asciende a 0.6 km/h.  

Además, verano y primavera muestran la mayor discrepancia entre la media y la moda, lo que podría explicarse por la menor cantidad de registros en estas estaciones y una distribución más dispersa con valores más altos que la moda.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c965d6e7-2696-4497-9bec-12126a445ff6" width="300">
  <img src="https://github.com/user-attachments/assets/95fddf4a-aab8-4d71-9afe-00b2197ff077" width="270">
</p>

Para concluir el análisis, examinamos la mediana, diferenciando también por tipo de maratón. En general, las maratones de 50 millas tienden a presentar valores más bajos, al igual que la estación de verano. En cuanto al orden de las estaciones, la mediana y la media siguen el mismo patrón, aunque con valores distintos.  

Uno de los resultados más llamativos es que, en otoño, la mediana de las carreras de 50 millas supera a la de 50 km en la misma estación. Además, la media de la maratón de 50 millas en primavera se sitúa por debajo de la de invierno, lo que resulta inusual.  

Con estos datos, podemos afirmar con seguridad que existe una diferencia en las velocidades medias de los atletas entre verano e invierno, aunque la variación es relativamente pequeña, con una diferencia promedio del 8%. Asimismo, se observa una caída significativa en la participación durante el verano en comparación con el invierno. Estos resultados son coherentes, ya que las altas temperaturas y la exposición al sol pueden generar incomodidad y fatiga, reduciendo tanto el número de corredores como sus velocidades medias. En primavera también se registra una menor participación, pero al tratarse de un clima más templado, el rendimiento de los atletas no se ve tan afectado.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c32f2a82-be38-4638-8b5f-8ac445a0ef76" width="300">
</p>
