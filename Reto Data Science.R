# DESAFÍO: Proyecto de análisis y modelado de vuelos

# 1. Limpieza y transformación de datos
# - ¿Cuántos valores faltantes hay en cada columna?
# - Elimina las filas con más del 50% de valores faltantes.
# - Imputa AIR_TIME y DEPARTURE_DELAY con la media por AIRLINE.

# 2. Ingeniería de variables
# - Crea una nueva variable binaria: LATE_ARRIVAL (1 si ARRIVAL_DELAY > 15, 0 si no).
# - Crea una variable HORA_DIA a partir de SCHEDULED_DEPARTURE (mañana, tarde, noche).
# - Convierte las variables categóricas relevantes (AIRLINE, ORIGIN_AIRPORT, etc.) en factores.

# 3. Normalización y escalado
# - Aplica normalización [0,1] a DISTANCE y AIR_TIME.
# - Estándariza DEPARTURE_DELAY y ARRIVAL_DELAY.

# 4. Análisis exploratorio
# - ¿Cuál es la proporción de vuelos con retraso por aerolínea?
# - ¿Hay alguna relación entre DISTANCE y ARRIVAL_DELAY? Usa un gráfico y un modelo lineal simple.

# 5. Clasificación: KNN
# - Ajusta un modelo KNN para predecir LATE_ARRIVAL.
# - Evalúa su rendimiento con accuracy y matriz de confusión.

# 6. Regresión: modelos penalizados
# - Ajusta un modelo de regresión Lasso para predecir ARRIVAL_DELAY.
# - Compara los coeficientes con un modelo de regresión lineal sin regularización.

# 7. Árboles de decisión
# - Entrena un árbol de decisión para predecir LATE_ARRIVAL.
# - ¿Qué variables resultan más importantes según el modelo?

# 8. Clustering no supervisado
# - Aplica K-Means sobre AIR_TIME y DISTANCE. Usa 3 clusters.
# - ¿Qué caracteriza a cada grupo? ¿Hay relación con la variable LATE_ARRIVAL?

# 9. Serie temporal
# - Agrupa los datos por día y calcula el número total de vuelos diarios.
# - Ajusta un modelo ARIMA para predecir la serie. ¿Detectas tendencia o estacionalidad?

# 10. Evaluación y validación
# - Compara el rendimiento de los modelos de regresión (lineal, lasso, árbol) usando RMSE y R².
# - ¿Qué modelo predice mejor los retrasos? ¿Por qué?

