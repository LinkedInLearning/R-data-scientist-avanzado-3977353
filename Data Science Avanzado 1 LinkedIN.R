ruta <- "C:/Users/linkedin/Dropbox/LinkedIn/"

df <- read.csv(paste0(ruta,"flights_small.csv"), sep=";")
df

#####################################
# Transformación y preprocesamiento #
#####################################

# Transformación y preprocesamiento de variables con el dataset df

# ------------------------
# 1. Conversión de tipos
# ------------------------

# Convertir algunas columnas a factor
df$AIRLINE <- as.factor(df$AIRLINE)
df$ORIGIN_AIRPORT <- as.factor(df$ORIGIN_AIRPORT)
df$DESTINATION_AIRPORT <- as.factor(df$DESTINATION_AIRPORT)

# Convertir el día de la semana a factor ordenado
df$DAY_OF_WEEK <- factor(df$DAY_OF_WEEK, 
                         levels = 1:7, 
                         labels = c("domingo", "lunes", "martes", "miércoles", "jueves", "viernes", "sábado"),
                         ordered = TRUE)

# ------------------------
# 2. Normalización de variables numéricas
# ------------------------

# Normalizar la variable DISTANCE con z-score
media_dist <- mean(df$DISTANCE, na.rm = TRUE)
sd_dist <- sd(df$DISTANCE, na.rm = TRUE)
df$DISTANCE_NORM <- (df$DISTANCE - media_dist) / sd_dist

# Normalización Min-Max de AIR_TIME
min_air <- min(df$AIR_TIME, na.rm = TRUE)
max_air <- max(df$AIR_TIME, na.rm = TRUE)
df$AIR_TIME_NORM <- (df$AIR_TIME - min_air) / (max_air - min_air)

# ------------------------
# 3. Transformaciones de fecha y hora
# ------------------------

# Crear columna de fecha (tipo Date)
df$fecha <- as.Date(paste(df$YEAR, df$MONTH, df$DAY, sep = "-"))

# Extraer hora de salida programada
df$HORA_SALIDA <- floor(df$SCHEDULED_DEPARTURE / 100)

# ------------------------
# 4. Codificación simple de variables categóricas
# ------------------------

# Codificar aerolínea como variable numérica (sin usar dplyr)
df$AIRLINE_CODED <- as.numeric(df$AIRLINE)

# Codificar origen y destino con números
df$ORIGIN_CODED <- as.numeric(df$ORIGIN_AIRPORT)
df$DEST_CODED <- as.numeric(df$DESTINATION_AIRPORT)

# ------------------------
# 5. Visualización rápida para comprobar transformaciones
# ------------------------

library(ggplot2)

# Comparar distribución original vs normalizada de distancia
ggplot(df, aes(x = DISTANCE)) +
  geom_histogram(bins = 50, fill = "skyblue", alpha = 0.6) +
  labs(title = "Distribución original de la distancia")

ggplot(df, aes(x = DISTANCE_NORM)) +
  geom_histogram(bins = 50, fill = "darkgreen", alpha = 0.6) +
  labs(title = "Distancia normalizada (z-score)")

# Ver número de vuelos por día de la semana
ggplot(df, aes(x = DAY_OF_WEEK)) +
  geom_bar(fill = "orange") +
  labs(title = "Número de vuelos por día de la semana", x = "Día", y = "Cantidad")

# ------------------------
# 6. Preparación final para modelado
# ------------------------

# Subset de variables ya transformadas
df_modelo <- df[, c("DISTANCE_NORM", "AIR_TIME_NORM", "AIRLINE_CODED", 
                    "ORIGIN_CODED", "DEST_CODED", "DAY_OF_WEEK")]
df_modelo <- na.omit(df_modelo)

####################
# Valores Ausentes #
####################

# Manejo de valores ausentes en el dataset df

# ------------------------
# 1. Detección de valores ausentes
# ------------------------

# Contar cuántos NAs hay por columna
colSums(is.na(df))

# Porcentaje de valores ausentes por variable
porcentaje_na <- colMeans(is.na(df)) * 100
print(round(porcentaje_na, 2))

# Ver algunas filas con NAs para hacerse una idea
head(df[!complete.cases(df), ])

# ------------------------
# 2. Eliminación de filas con NAs
# ------------------------

# Eliminar filas con al menos un NA
df_sin_na <- na.omit(df)

# También se puede eliminar solo si hay NA en columnas concretas
df_sin_na2 <- df[!is.na(df$AIR_TIME) & !is.na(df$DISTANCE), ]

# ------------------------
# 3. Imputación simple de valores numéricos
# ------------------------

# Reemplazar NAs en AIR_TIME por la media
media_airtime <- mean(df$AIR_TIME, na.rm = TRUE)
df$AIR_TIME_IMPUTADA <- ifelse(is.na(df$AIR_TIME), media_airtime, df$AIR_TIME)

# Imputar la distancia con la mediana
mediana_dist <- median(df$DISTANCE, na.rm = TRUE)
df$DISTANCE_IMPUTADA <- ifelse(is.na(df$DISTANCE), mediana_dist, df$DISTANCE)

# ------------------------
# 4. Imputación por valor fijo o 0
# ------------------------

# Imputar retraso al aterrizar con 0 si está vacío (posible cancelación o dato perdido)
df$ARRIVAL_DELAY_IMPUTADA <- ifelse(is.na(df$ARRIVAL_DELAY), 0, df$ARRIVAL_DELAY)

# ------------------------
# 5. Imputación de variables categóricas
# ------------------------

# Reemplazar NAs con una categoría "Desconocido"
df$TAIL_NUMBER <- as.character(df$TAIL_NUMBER)
df$TAIL_NUMBER[is.na(df$TAIL_NUMBER)] <- "Desconocido"
df$TAIL_NUMBER <- as.factor(df$TAIL_NUMBER)

# ------------------------
# 6. Visualización antes y después de imputar
# ------------------------

library(ggplot2)

# Comparar histogramas de la variable imputada y original
ggplot(df, aes(x = AIR_TIME_IMPUTADA)) +
  geom_histogram(bins = 40, fill = "steelblue", alpha = 0.7) +
  labs(title = "AIR_TIME tras imputación con la media")

# Comparar cantidad de NAs antes y después
sum(is.na(df$AIR_TIME))      # Original
sum(is.na(df$AIR_TIME_IMPUTADA))  # Ya imputada

# ------------------------
# 7. Crear copia final limpia
# ------------------------

# Crear dataframe con variables imputadas y sin NAs
df_limpio <- df[, c("AIR_TIME_IMPUTADA", "DISTANCE_IMPUTADA", "ARRIVAL_DELAY_IMPUTADA",
                    "TAIL_NUMBER", "AIRLINE", "DAY_OF_WEEK")]
df_limpio <- na.omit(df_limpio)  # Por si queda algún NA residual


##### Iterative Imputer

library(mice)

# Seleccionar columnas numéricas con posibles NA
datos <- df[, c("DEPARTURE_DELAY", "ARRIVAL_DELAY", "AIR_TIME", "DISTANCE")]


# Imputación iterativa con modelo predictivo (regresión)
imp <- mice(datos, method = "norm.predict", m = 1, maxit = 50, seed = 42)

# Obtener dataset completo
df_imputado <- complete(imp)

df_imputado


###########################
# Ingeniería de variables #
###########################

# Ingeniería de características con el dataset df

# ------------------------
# 1. Ejemplos sencillos: combinación de columnas
# ------------------------

# Tiempo total en tierra: taxi out + taxi in
df$TIEMPO_TIERRA <- df$TAXI_OUT + df$TAXI_IN

# Diferencia entre tiempo programado y real
df$DIFERENCIA_TIEMPO <- df$ELAPSED_TIME - df$SCHEDULED_TIME

# ------------------------
# 2. Variables binarias
# ------------------------


# ¿Retraso en llegada mayor a 15 minutos?
df$RETRASO_LLEGADA <- ifelse(df$ARRIVAL_DELAY > 15, 1, 0)

# ------------------------
# 3. Variables horarias a categorías
# ------------------------

# Crear variable de parte del día a partir de la hora de salida programada
df$HORA <- floor(df$SCHEDULED_DEPARTURE / 100)

df$PERIODO_DIA <- ifelse(df$HORA < 6, "madrugada",
                         ifelse(df$HORA < 12, "mañana",
                                ifelse(df$HORA < 18, "tarde", "noche")))
df$PERIODO_DIA <- factor(df$PERIODO_DIA,
                         levels = c("madrugada", "mañana", "tarde", "noche"))

# ------------------------
# 4. Extracción de características temporales
# ------------------------

# Fecha completa
df$FECHA <- as.Date(paste(df$YEAR, df$MONTH, df$DAY, sep = "-"))

# Día del mes como variable
df$DIA_DEL_MES <- df$DAY

# ¿Es fin de semana?
df$FIN_DE_SEMANA <- ifelse(df$DAY_OF_WEEK %in% c(1, 7), 1, 0)

# ------------------------
# 5. Agrupación de aeropuertos
# ------------------------

# Aeropuertos con más vuelos de origen
tabla_origen <- sort(table(df$ORIGIN_AIRPORT), decreasing = TRUE)
principales <- names(tabla_origen)[1:5]

# Nueva variable: aeropuerto de origen popular o no
df$ORIGEN_POPULAR <- ifelse(df$ORIGIN_AIRPORT %in% principales, "sí", "no")
df$ORIGEN_POPULAR <- factor(df$ORIGEN_POPULAR)

# ------------------------
# 6. Combinaciones útiles para predicción
# ------------------------

# Tiempo total de vuelo (aire + tierra)
df$TIEMPO_TOTAL_VUELO <- df$AIR_TIME + df$TAXI_OUT + df$TAXI_IN

# Retraso total (salida + llegada)
df$RETRASO_TOTAL <- df$DEPARTURE_DELAY + df$ARRIVAL_DELAY

# Porcentaje de tiempo en tierra sobre el total
df$PORC_TIERRA <- df$TIEMPO_TIERRA / df$TIEMPO_TOTAL_VUELO

# ------------------------
# 7. Visualización de variables nuevas
# ------------------------

library(ggplot2)

# Distribución del tiempo total de vuelo
ggplot(df, aes(x = TIEMPO_TOTAL_VUELO)) +
  geom_histogram(bins = 50, fill = "steelblue", alpha = 0.7) +
  labs(title = "Tiempo total de vuelo (aire + tierra)")

# Comparar retraso total entre días laborables y fines de semana
ggplot(df, aes(x = as.factor(FIN_DE_SEMANA), y = RETRASO_TOTAL)) +
  geom_boxplot(fill = "orange") +
  labs(title = "Retraso total entre semana y fin de semana",
       x = "¿Fin de semana?", y = "Retraso total (min)")

# ------------------------
# 8. Dataset listo para modelar
# ------------------------

# Seleccionar algunas variables nuevas junto a otras básicas
df_modelo <- df[, c("RETRASO_LLEGADA", "TIEMPO_TOTAL_VUELO", "PORC_TIERRA",
                    "CANCELADO_BIN", "DESVIADO_BIN", "FIN_DE_SEMANA", 
                    "ORIGEN_POPULAR", "PERIODO_DIA")]

# Eliminar filas con NAs
df_modelo <- na.omit(df_modelo)

########################
# Datos desbalanceados #
########################

# Estrategias para equilibrar datos desbalanceados con el dataset df

# ------------------------
# 1. Crear variable objetivo binaria
# ------------------------

# Se considera retraso en llegada si supera los 15 minutos
df$RETRASO_LLEGADA <- ifelse(df$ARRIVAL_DELAY > 15, 1, 0)

# Tabla de clases
table(df$RETRASO_LLEGADA)

# Proporción de clases
prop.table(table(df$RETRASO_LLEGADA))

# ------------------------
# 2. Submuestreo de la clase mayoritaria
# ------------------------

# Índices de cada clase
indices_0 <- which(df$RETRASO_LLEGADA == 0)
indices_1 <- which(df$RETRASO_LLEGADA == 1)

# Número de elementos de la clase minoritaria
n_minoria <- length(indices_1)

# Seleccionar aleatoriamente el mismo número de clase mayoritaria
set.seed(123)
indices_0_sub <- sample(indices_0, n_minoria)

# Combinar ambos
df_balanceado_sub <- df[c(indices_0_sub, indices_1), ]

# Comprobamos proporciones
prop.table(table(df_balanceado_sub$RETRASO_LLEGADA))

# ------------------------
# 3. Sobremuestreo de la clase minoritaria
# ------------------------

# Crear réplicas de la clase minoritaria
indices_1_rep <- sample(indices_1, length(indices_0), replace = TRUE)

# Combinar con clase mayoritaria
df_balanceado_over <- df[c(indices_0, indices_1_rep), ]

# Comprobamos proporciones
prop.table(table(df_balanceado_over$RETRASO_LLEGADA))

# ------------------------
# 4. Dataset original vs balanceado
# ------------------------

library(ggplot2)

# Original
ggplot(df, aes(x = as.factor(RETRASO_LLEGADA))) +
  geom_bar(fill = "grey") +
  labs(title = "Distribución original de la variable objetivo",
       x = "Retraso en llegada", y = "Frecuencia")

# Tras submuestreo
ggplot(df_balanceado_sub, aes(x = as.factor(RETRASO_LLEGADA))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribución tras submuestreo",
       x = "Retraso en llegada", y = "Frecuencia")

# Tras sobremuestreo
ggplot(df_balanceado_over, aes(x = as.factor(RETRASO_LLEGADA))) +
  geom_bar(fill = "darkgreen") +
  labs(title = "Distribución tras sobremuestreo",
       x = "Retraso en llegada", y = "Frecuencia")

# ------------------------
# 5. Preparar dataset final para modelado
# ------------------------

# Selección de algunas variables predictoras
variables <- c("RETRASO_LLEGADA", "DISTANCE", "AIR_TIME", "DEPARTURE_DELAY")

# Dataset balanceado por submuestreo
df_modelo_sub <- df_balanceado_sub[, variables]
df_modelo_sub <- na.omit(df_modelo_sub)

# Dataset balanceado por sobremuestreo
df_modelo_over <- df_balanceado_over[, variables]
df_modelo_over <- na.omit(df_modelo_over)

####### SMOTE

# Instalar el paquete si no está instalado
# install.packages("smotefamily")
library(smotefamily)


# Convertir la variable objetivo a factor
df$RETRASO_LLEGADA <- ifelse(df$ARRIVAL_DELAY > 15, 1, 0)
df$RETRASO_LLEGADA <- as.factor(df$RETRASO_LLEGADA)

# Subconjunto con variables numéricas
df_sub <- df[, c("RETRASO_LLEGADA", "DISTANCE", "DEPARTURE_DELAY")]
df_sub <- na.omit(df_sub)

# Separar en variables predictoras y objetivo
x <- df_sub[, c("DISTANCE", "DEPARTURE_DELAY")]
y <- df_sub$RETRASO_LLEGADA

# Aplicar SMOTE
set.seed(123)
smote_result <- SMOTE(X = x, target = y, K = 5)

# Dataset balanceado
df_smote <- smote_result$data
df_smote$RETRASO_LLEGADA <- as.factor(df_smote$class)
df_smote$class <- NULL

# Verificar proporciones
prop.table(table(df_smote$RETRASO_LLEGADA))
prop.table(table(y))

