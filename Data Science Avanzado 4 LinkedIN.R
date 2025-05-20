ruta <- "C:/Users/linkedin/Dropbox/LinkedIn/"

df <- read.csv(paste0(ruta,"flights_small.csv"), sep=";")
df


#####################
# Intro Time Series #
#####################

# Fundamentos de análisis de series temporales en R

# ------------------------
# 1. Crear variable de fecha
# ------------------------

df$FECHA <- as.Date(paste(df$YEAR, df$MONTH, df$DAY, sep = "-"))

# Verificar rango temporal
range(df$FECHA)

# ------------------------
# 2. Agregar retraso medio diario
# ------------------------

# Agregamos la media del ARRIVAL_DELAY por fecha
df_ts <- aggregate(ARRIVAL_DELAY ~ FECHA, data = df, FUN = mean, na.rm = TRUE)

# Renombrar
colnames(df_ts) <- c("FECHA", "RETRASO_MEDIO")

# ------------------------
# 3. Visualización básica de la serie
# ------------------------

library(ggplot2)

ggplot(df_ts, aes(x = FECHA, y = RETRASO_MEDIO)) +
  geom_line(color = "steelblue") +
  labs(title = "Retraso medio diario en llegada",
       x = "Fecha", y = "Minutos de retraso")

# ------------------------
# 4. Convertir a objeto ts
# ------------------------

# Hay que asegurarse de que sea regular (por ejemplo, diaria sin huecos)
# Usamos zoo para crear una serie temporal flexible
# install.packages("zoo")
library(zoo)

serie_zoo <- zoo(df_ts$RETRASO_MEDIO, order.by = df_ts$FECHA)
plot(serie_zoo, main = "Serie temporal: retraso medio diario", ylab = "Minutos")

# ------------------------
# 5. Comprobar estacionalidad o tendencia visualmente
# ------------------------

# Promedios móviles para suavizar
media_movil <- rollmean(serie_zoo, k = 7, fill = NA)

plot(serie_zoo, main = "Serie original y media móvil (7 días)", col = "grey")
lines(media_movil, col = "red", lwd = 2)


# ------------------------
# 8. Preparación para análisis futuro
# ------------------------

# Crear serie ts (ejemplo: frecuencia semanal)
serie_ts <- ts(df_ts$RETRASO_MEDIO, frequency = 7)


##################
# Descomposición #
##################

# Componentes de series temporales: tendencia, estacionalidad y ruido

# ------------------------
# 1. Crear la serie temporal
# ------------------------

df$FECHA <- as.Date(paste(df$YEAR, df$MONTH, df$DAY, sep = "-"))
df_ts <- aggregate(ARRIVAL_DELAY ~ FECHA, data = df, FUN = mean, na.rm = TRUE)
colnames(df_ts) <- c("FECHA", "RETRASO_MEDIO")

# Para descomponer, necesitamos una serie regular (sin días faltantes)
# Creamos una secuencia diaria completa
fechas_completas <- seq(min(df_ts$FECHA), max(df_ts$FECHA), by = "day")

# Rellenamos días faltantes con NA
ts_completa <- merge(data.frame(FECHA = fechas_completas), df_ts, all.x = TRUE)

# Imputar valores NA por interpolación lineal (simple)
ts_completa$RETRASO_MEDIO <- zoo::na.approx(ts_completa$RETRASO_MEDIO, na.rm = FALSE)

# Crear objeto ts (diario, sin frecuencia clara, asumimos semanal para empezar)
serie_ts <- ts(ts_completa$RETRASO_MEDIO, frequency = 7)

# ------------------------
# 2. Descomposición clásica
# ------------------------

descomp <- decompose(serie_ts)

# Visualizar los componentes
plot(descomp)

# ------------------------
# 3. Interpretar cada componente
# ------------------------

# Tendencia: evolución a largo plazo
# Estacionalidad: patrón que se repite cada semana
# Ruido: variación no explicada

# ------------------------
# 4. Extraer y graficar individualmente
# ------------------------

descomp$seasonal[1:7]

library(ggplot2)

componentes <- data.frame(
  Fecha = fechas_completas,
  Observado = descomp$x,
  Tendencia = descomp$trend,
  Estacionalidad = descomp$seasonal,
  Ruido = descomp$random
)

# Tendencia
ggplot(componentes, aes(x = Fecha, y = Tendencia)) +
  geom_line(na.rm = TRUE, color = "red") +
  labs(title = "Tendencia del retraso medio diario", y = "Tendencia", x = "Fecha")

# Estacionalidad semanal
ggplot(componentes, aes(x = Fecha, y = Estacionalidad)) +
  geom_line(color = "blue") +
  labs(title = "Estacionalidad semanal", y = "Estacionalidad", x = "Fecha")

# Ruido
ggplot(componentes, aes(x = Fecha, y = Ruido)) +
  geom_line(color = "grey") +
  labs(title = "Componente aleatoria (residuos)", y = "Ruido", x = "Fecha")

#########
# ARIMA #
#########

# Modelos ARIMA: autoregresión y medias móviles

# ------------------------
# 1. Preparar la serie temporal
# ------------------------

df$FECHA <- as.Date(paste(df$YEAR, df$MONTH, df$DAY, sep = "-"))
df_ts <- aggregate(ARRIVAL_DELAY ~ FECHA, data = df, FUN = mean, na.rm = TRUE)
colnames(df_ts) <- c("FECHA", "RETRASO_MEDIO")

# Completar fechas faltantes y rellenar
fechas_completas <- seq(min(df_ts$FECHA), max(df_ts$FECHA), by = "day")
ts_completa <- merge(data.frame(FECHA = fechas_completas), df_ts, all.x = TRUE)
ts_completa$RETRASO_MEDIO <- zoo::na.approx(ts_completa$RETRASO_MEDIO, na.rm = FALSE)

# Crear objeto ts
serie_ts <- ts(ts_completa$RETRASO_MEDIO)

# ------------------------
# 2. Visualización y análisis preliminar
# ------------------------

plot(serie_ts, main = "Retraso medio diario", ylab = "Minutos")

# ------------------------
# 3. Comprobar estacionariedad
# ------------------------

# install.packages("tseries")
library(tseries)

# Test de Dickey-Fuller aumentado
adf.test(serie_ts, alternative = "stationary")

# Si p > 0.05 → no es estacionaria → aplicar diferenciación
serie_diff <- diff(serie_ts)

# Repetir test
adf.test(serie_diff, alternative = "stationary")

# ------------------------
# 4. Identificación automática del modelo ARIMA
# ------------------------

# install.packages("forecast")
library(forecast)

modelo_auto <- auto.arima(serie_ts)
summary(modelo_auto)

# ------------------------
# 5. Diagnóstico del modelo
# ------------------------

checkresiduals(modelo_auto)

# ------------------------
# 6. Predicción
# ------------------------

# Predecir los próximos 14 días
pred <- forecast(modelo_auto, h = 14)

# Mostrar predicción
plot(pred, main = "Pronóstico de retraso medio (ARIMA)")

# ------------------------
# 7. Guardar predicciones
# ------------------------

pred_df <- data.frame(
  Fecha = seq(max(ts_completa$FECHA) + 1, by = "day", length.out = 14),
  Prediccion = as.numeric(pred$mean)
)

pred_df


####################
# SARIMA y SARIMAX #
####################

# Variaciones avanzadas de ARIMA: SARIMA y ARIMAX

# ------------------------
# 1. Preparar la serie temporal
# ------------------------

df$FECHA <- as.Date(paste(df$YEAR, df$MONTH, df$DAY, sep = "-"))
df_ts <- aggregate(ARRIVAL_DELAY ~ FECHA, data = df, FUN = mean, na.rm = TRUE)
colnames(df_ts) <- c("FECHA", "RETRASO_MEDIO")

# Completar fechas faltantes
fechas_completas <- seq(min(df_ts$FECHA), max(df_ts$FECHA), by = "day")
ts_completa <- merge(data.frame(FECHA = fechas_completas), df_ts, all.x = TRUE)
ts_completa$RETRASO_MEDIO <- zoo::na.approx(ts_completa$RETRASO_MEDIO, na.rm = FALSE)

# Crear objeto ts con frecuencia semanal (7 días)
serie_ts <- ts(ts_completa$RETRASO_MEDIO, frequency = 7)

# ------------------------
# 2. SARIMA: ARIMA con componente estacional
# ------------------------

# install.packages("forecast")
library(forecast)

modelo_auto <- auto.arima(serie_ts, seasonal = TRUE)
summary(modelo_auto)

# Ajustar modelo SARIMA manual
modelo_sarima <- Arima(serie_ts, order = c(3,1,3), seasonal = c(3,0,3))

summary(modelo_sarima)

# Pronóstico
pred_sarima <- forecast(modelo_sarima, h = 14)
plot(pred_sarima, main = "SARIMA - Retraso medio diario")

# ------------------------
# 3. ARIMAX: ARIMA con variables externas
# ------------------------

# Agregamos DISTANCE medio diario como regresor externo
df_dist <- aggregate(DISTANCE ~ FECHA, data = df, FUN = mean, na.rm = TRUE)
ts_externa <- merge(data.frame(FECHA = fechas_completas), df_dist, all.x = TRUE)
ts_externa$DISTANCE <- zoo::na.approx(ts_externa$DISTANCE, na.rm = FALSE)

# Convertimos a ts
serie_externa <- ts(ts_externa$DISTANCE, frequency = 7)

# Ajustamos ARIMAX con variable externa
modelo_arimax <- auto.arima(serie_ts, xreg = serie_externa)
summary(modelo_arimax)

# Predecimos 14 días, generando también DISTANCE futura (ejemplo ficticio: mantener constante)
n_futuro <- 14
xreg_futuro <- rep(mean(ts_externa$DISTANCE, na.rm = TRUE), n_futuro)

pred_arimax <- forecast(modelo_arimax, xreg = xreg_futuro, h = n_futuro)

plot(pred_arimax, main = "ARIMAX - Retraso medio con DISTANCE como regresor")

# ------------------------
# 4. Comparar resultados
# ------------------------

data.frame(
  SARIMA = round(pred_sarima$mean, 2),
  ARIMAX = round(pred_arimax$mean, 2)
)


############
# Forecast #
############

# Pronóstico de ARRIVAL_DELAY con zoom en el tramo final de la serie

# ------------------------
# 1. Preparar la serie temporal
# ------------------------

df$FECHA <- as.Date(paste(df$YEAR, df$MONTH, df$DAY, sep = "-"))
df_ts <- aggregate(ARRIVAL_DELAY ~ FECHA, data = df, FUN = mean, na.rm = TRUE)
colnames(df_ts) <- c("FECHA", "RETRASO_MEDIO")

fechas_completas <- seq(min(df_ts$FECHA), max(df_ts$FECHA), by = "day")
ts_completa <- merge(data.frame(FECHA = fechas_completas), df_ts, all.x = TRUE)
ts_completa$RETRASO_MEDIO <- zoo::na.approx(ts_completa$RETRASO_MEDIO, na.rm = FALSE)

# Crear serie ts con frecuencia semanal
serie_ts <- ts(ts_completa$RETRASO_MEDIO, frequency = 7)

# ------------------------
# 2. Ajustar modelo ARIMA y predecir
# ------------------------

library(forecast)

modelo <- auto.arima(serie_ts,seasonal = TRUE)
modelo <- Arima(serie_ts, order = c(3,1,3), seasonal = c(3,1,3))


forecast_result <- forecast(modelo, h = 30)

# ------------------------
# 3. Extraer los últimos 60 días + predicción
# ------------------------

# Extraer fechas reales finales
fechas_finales <- tail(fechas_completas, 60)

# Valores reales de los últimos 60 días
reales_final <- tail(ts_completa$RETRASO_MEDIO, 60)

# Fechas de predicción
fechas_futuras <- seq(max(fechas_finales) + 1, by = "day", length.out = 30)

# Combinar datos reales y predichos
df_zoom <- data.frame(
  Fecha = c(fechas_finales, fechas_futuras),
  Valor = c(reales_final, as.numeric(forecast_result$mean)),
  Tipo = c(rep("Observado", 60), rep("Pronóstico", 30))
)

# ------------------------
# 4. Visualización con zoom
# ------------------------

library(ggplot2)

ggplot(df_zoom, aes(x = Fecha, y = Valor, color = Tipo)) +
  geom_line(size = 1) +
  geom_point() +
  scale_color_manual(values = c("Observado" = "steelblue", "Pronóstico" = "darkred")) +
  labs(title = "Zoom: últimos 60 días + 30 días de pronóstico",
       y = "Retraso medio (min)", x = "Fecha")

######## Errores ########


modelo <- auto.arima(serie_ts,seasonal = TRUE)
#modelo <- Arima(serie_ts, order = c(3,1,3), seasonal = c(3,1,3))

ultimaspredicciones <- modelo$fitted[(length(modelo$fitted)-59):length(modelo$fitted)]
error <- reales_final - ultimaspredicciones
rmse <- sqrt(mean(error^2))
mae <- mean(abs(error))
mae

df_evaluado <- data.frame(real = reales_final, predicho = ultimaspredicciones )

df_evaluado$indice <- seq_len(nrow(df_evaluado))

# Reorganizar en formato largo para graficar más fácilmente
df_long <- reshape2::melt(df_evaluado, id.vars = "indice")

# Crear el gráfico
ggplot(df_long, aes(x = indice, y = value, color = variable)) +
  geom_line(size = 1) +
  labs(title = "Comparación entre valores reales y predichos",
       x = "Observación",
       y = "Valor",
       color = "Serie") +
  theme_minimal()

