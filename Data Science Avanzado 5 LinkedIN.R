
ruta <- "C:/Users/linkedin/Dropbox/LinkedIn/"

df <- read.csv(paste0(ruta,"flights_small.csv"), sep=";")
df


############
# OUTLIERS #
############


# Detección de anomalías y outliers en R

# ------------------------
# 1. Detección básica con boxplot
# ------------------------

# Boxplot clásico para ver valores extremos
boxplot(df$ARRIVAL_DELAY,
        main = "Boxplot de ARRIVAL_DELAY",
        ylab = "Minutos de retraso")

# ------------------------
# 2. Detección con IQR
# ------------------------

# Calcular los límites de IQR
q1 <- quantile(df$ARRIVAL_DELAY, 0.25, na.rm = TRUE)
q3 <- quantile(df$ARRIVAL_DELAY, 0.75, na.rm = TRUE)
iqr <- q3 - q1

lim_inf <- q1 - 3 * iqr
lim_sup <- q3 + 3 * iqr

# Marcar outliers
df$outlier_iqr <- ifelse(df$ARRIVAL_DELAY < lim_inf | df$ARRIVAL_DELAY > lim_sup, 1, 0)

cat("Porcentaje de outliers según IQR:", round(mean(df$outlier_iqr, na.rm = TRUE) * 100, 2), "%\n")

# ------------------------
# 3. Detección con z-score
# ------------------------

media <- mean(df$ARRIVAL_DELAY, na.rm = TRUE)
desv <- sd(df$ARRIVAL_DELAY, na.rm = TRUE)

z_scores <- (df$ARRIVAL_DELAY - media) / desv
df$outlier_z <- ifelse(abs(z_scores) > 3, 1, 0)

cat("Outliers por z-score:", round(mean(df$outlier_z, na.rm = TRUE)*100,2), "%\n")

# ------------------------
# 4. Visualización de outliers detectados
# ------------------------

library(ggplot2)

ggplot(df, aes(x = ARRIVAL_DELAY)) +
  geom_histogram(bins = 100, fill = "grey", alpha = 0.5) +
  geom_vline(xintercept = c(lim_inf, lim_sup), col = "red", linetype = "dashed") +
  labs(title = "Detección de outliers por IQR")


# Detección de outliers con Isolation Forest en R

# ------------------------
# 1. Preparar datos
# ------------------------

# Seleccionar variables numéricas relevantes
df_iso <- df[, c("DEPARTURE_DELAY", "DISTANCE", "AIR_TIME", "SCHEDULED_TIME")]
df_iso <- na.omit(df_iso)

# Escalar variables (opcional, pero recomendable)
df_scaled <- scale(df_iso)

# ------------------------
# 2. Aplicar Isolation Forest
# ------------------------

# install.packages("solitude")  # si no lo tienes
library(solitude)

# Crear modelo Isolation Forest
modelo_iso <- isolationForest$new()

# Entrenar
modelo_iso$fit(df_scaled)

# Obtener puntuaciones de anomalía
anomalías <- modelo_iso$predict(df_scaled)

# Añadir al dataset original
df_iso$anomaly_score <- anomalías$anomaly_score
df_iso$outlier <- ifelse(anomalías$anomaly_score > 0.65, 1, 0)

# ------------------------
# 3. Visualización
# ------------------------

library(ggplot2)

ggplot(df_iso, aes(x = DISTANCE, y = DEPARTURE_DELAY, color = as.factor(outlier))) +
  geom_point(alpha = 0.6) +
  scale_color_manual(values = c("0" = "black", "1" = "red")) +
  labs(title = "Outliers detectados por Isolation Forest",
       x = "Distancia", y = "Retraso en salida", color = "Outlier")

# ------------------------
# 4. Número de outliers detectados
# ------------------------

cat("Número de outliers detectados:", sum(df_iso$outlier), "\n")


#############
# PCA y LDA #
#############

# Reducción de dimensionalidad en R: PCA (no supervisado) y LDA (supervisado)

# ------------------------
# 1. Preparar los datos
# ------------------------

# Creamos una variable binaria como objetivo (para LDA)
df$RETRASO_LLEGADA <- ifelse(df$ARRIVAL_DELAY > 15, 1, 0)
df$RETRASO_LLEGADA <- as.factor(df$RETRASO_LLEGADA)

# Subconjunto numérico para análisis
df_red <- df[, c("RETRASO_LLEGADA", "DEPARTURE_DELAY", "DISTANCE", "AIR_TIME", "SCHEDULED_TIME", "ELAPSED_TIME")]
df_red <- na.omit(df_red)

# Separar variables
x <- df_red[, -1]
y <- df_red$RETRASO_LLEGADA

# ------------------------
# 2. PCA: Análisis de Componentes Principales
# ------------------------

# Escalar variables
x_scaled <- scale(x)

pca$sdev

# Calcular PCA
pca <- prcomp(x_scaled)

pca$rotation 

# Ver la proporción de varianza explicada
summary(pca)

# Gráfico del porcentaje de varianza acumulada
plot(cumsum(pca$sdev^2 / sum(pca$sdev^2)),
     type = "b", xlab = "Número de componentes",
     ylab = "Varianza acumulada",
     main = "PCA - Varianza explicada")

# ------------------------
# 3. Visualización con dos primeras componentes
# ------------------------

library(ggplot2)

pca_df <- data.frame(pca$x[, 1:2], Retraso = y)

ggplot(pca_df, aes(x = PC1, y = PC2, color = Retraso)) +
  geom_point(alpha = 0.6) +
  labs(title = "PCA: primeros dos componentes",
       x = "Componente 1", y = "Componente 2")

# ------------------------
# 4. LDA: Análisis Discriminante Lineal
# ------------------------

# install.packages("MASS")
library(MASS)

modelo_lda <- lda(RETRASO_LLEGADA ~ ., data = df_red)

# Proyecciones lineales
lda_pred <- predict(modelo_lda)
lda_df <- data.frame(lda_pred$x, Retraso = df_red$RETRASO_LLEGADA)

ggplot(lda_df, aes(x = LD1, fill = Retraso)) +
  geom_density(alpha = 0.5) +
  labs(title = "LDA: separación de clases en LD1")

# ------------------------
# 5. Comparar PCA vs LDA (objetivo)
# ------------------------

# PCA es no supervisado → maximiza varianza general
# LDA es supervisado → maximiza separación entre clases

# LDA también puede usarse para clasificación
tabla <- table(Predicho = lda_pred$class, Real = y)
print(tabla)

accuracy <- mean(lda_pred$class == y)
cat("Precisión LDA:", round(accuracy * 100, 2), "%\n")

#######################
# Selección variables #
#######################

# Selección automática de variables en R

# ------------------------
# 1. Preparar los datos
# ------------------------

# Selección de variables
vars <- c("ARRIVAL_DELAY", "DEPARTURE_DELAY", "DISTANCE", "SCHEDULED_TIME",
          "AIR_TIME", "ELAPSED_TIME", "TAXI_OUT", "TAXI_IN", "WHEELS_OFF", "WHEELS_ON")

df_sel <- df[, vars]
df_sel <- na.omit(df_sel)

# Dividir variables
x <- df_sel[, -1]
y <- df_sel$ARRIVAL_DELAY

# ------------------------
# 2. Selección paso a paso (stepwise)
# ------------------------

modelo_full <- lm(ARRIVAL_DELAY ~ ., data = df_sel)

modelo_step <- step(modelo_full, direction = "both", trace = 1)

summary(modelo_step)

cat("Variables seleccionadas por stepwise:\n")
print(names(coef(modelo_step)))

`%notin%` <- Negate(`%in%`)
cat("Variables descartadas por stepwise:\n")
print(vars[vars %notin% names(coef(modelo_step))])

# ------------------------
# 3. LASSO para selección automática
# ------------------------

# install.packages("glmnet")
library(glmnet)

x_matrix <- as.matrix(x)

# Ajustar modelo LASSO
modelo_lasso <- cv.glmnet(x_matrix, y, alpha = 1)

# Coeficientes seleccionados
coef_lasso <- coef(modelo_lasso, s = "lambda.min")

cat("Variables seleccionadas por LASSO:\n")
print(coef_lasso[coef_lasso[,1] != 0, , drop = FALSE])

# ------------------------
# 4. Importancia de variables con Random Forest
# ------------------------

# install.packages("randomForest")
library(randomForest)

modelo_rf <- randomForest(ARRIVAL_DELAY ~ ., data = df_sel, ntree = 1000, importance = TRUE)

importancia <- importance(modelo_rf)

cat("Importancia de variables según Random Forest:\n")
print(round(importancia[, "%IncMSE"], 2))

# Visualizar importancia
library(ggplot2)

df_imp <- data.frame(Variable = rownames(importancia), Importancia = importancia[, "%IncMSE"])

ggplot(df_imp, aes(x = reorder(Variable, Importancia), y = Importancia)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Importancia de variables según Random Forest",
       x = "Variable", y = "% aumento en MSE al eliminar")

############
# Big Data #
############

# Machine Learning en Big Data con data.table y h2o

# ------------------------
# 1. Cargar datos con data.table (lectura rápida)
# ------------------------

# install.packages("data.table")
library(data.table)

# Supongamos que estamos cargando un archivo grande
# df_big <- fread("flights_big.csv")  # aquí usarías un CSV real

# Para este ejemplo seguimos con df (ya en memoria)
df_big <- as.data.table(df)

# ------------------------
# 2. Trabajar con data.table para eficiencia
# ------------------------

# Calcular retraso medio por aeropuerto de origen
df_big[, .(retraso_medio = mean(ARRIVAL_DELAY, na.rm = TRUE)),
       by = ORIGIN_AIRPORT][order(-retraso_medio)]

# Filtrar filas de forma eficiente
df_filtrado <- df_big[ARRIVAL_DELAY > 60 & DEPARTURE_DELAY > 60]

# ------------------------
# 3. Iniciar h2o para computación distribuida
# ------------------------


# install.packages("h2o")
library(h2o)
h2o.init(max_mem_size = "4G", nthreads = -1)

# Convertir a H2OFrame
df_h2o <- as.h2o(df_big[, .(ARRIVAL_DELAY, DEPARTURE_DELAY, DISTANCE, AIR_TIME, SCHEDULED_TIME)])

# ------------------------
# 4. División de datos y entrenamiento automático
# ------------------------

splits <- h2o.splitFrame(df_h2o, ratios = 0.7, seed = 123)
train <- splits[[1]]
test <- splits[[2]]

y <- "ARRIVAL_DELAY"
x <- setdiff(colnames(df_h2o), y)

modelo_auto <- h2o.automl(
  x = x,
  y = y,
  training_frame = train,
  leaderboard_frame = test,
  max_runtime_secs = 60,
  stopping_metric = "RMSE",
  seed = 123
)

# ------------------------
# 5. Evaluación y predicción
# ------------------------

mejor_modelo <- modelo_auto@leader
pred <- h2o.predict(mejor_modelo, test)

perf <- h2o.performance(mejor_modelo, newdata = test)
r2 <- h2o.r2(perf)
rmse <- h2o.rmse(perf)

cat("R²:", round(r2, 4), "\n")
cat("RMSE:", round(rmse, 2), "\n")

# ------------------------
# 6. Apagar h2o cuando termines
# ------------------------

h2o.shutdown(prompt = FALSE)


##########################
# Inferencia Estadística #
##########################

# Inferencia estadística aplicada en R

# ------------------------
# 1. Comparación de retraso medio entre dos aerolíneas
# ------------------------

# Seleccionar las dos aerolíneas con más vuelos
tabla <- sort(table(df$AIRLINE), decreasing = TRUE)
top2 <- names(tabla)[1:2]

# Filtrar vuelos de esas dos aerolíneas
df_comp <- df[df$AIRLINE %in% top2, ]
df_comp <- na.omit(df_comp[, c("AIRLINE", "ARRIVAL_DELAY")])

# Boxplot visual
boxplot(ARRIVAL_DELAY ~ AIRLINE, data = df_comp,
        main = "Retraso medio por aerolínea",
        ylab = "Minutos de retraso")

# ------------------------
# 2. Prueba t para diferencia de medias
# ------------------------

grupo1 <- df_comp$ARRIVAL_DELAY[df_comp$AIRLINE == top2[1]]
grupo2 <- df_comp$ARRIVAL_DELAY[df_comp$AIRLINE == top2[2]]

# H0: iguales
# H1: distintas
# pvalor < 0.05

t_test <- t.test(grupo1, grupo2, var.equal = FALSE)
print(t_test)

# Interpretación rápida
if (t_test$p.value < 0.05) {
  cat("Diferencia significativa entre aerolíneas (p <", round(t_test$p.value, 4), ")\n")
} else {
  cat("No hay evidencia de diferencia significativa (p =", round(t_test$p.value, 4), ")\n")
}

# ------------------------
# 3. Intervalo de confianza para la media del retraso
# ------------------------

media <- mean(df$ARRIVAL_DELAY, na.rm = TRUE)
error <- sd(df$ARRIVAL_DELAY, na.rm = TRUE) / sqrt(sum(!is.na(df$ARRIVAL_DELAY)))
ic_inf <- media - 1.96 * error
ic_sup <- media + 1.96 * error

cat("Intervalo de confianza 95% para la media del ARRIVAL_DELAY:\n")
cat("(", round(ic_inf, 2), ",", round(ic_sup, 2), ")\n")

# ------------------------
# 4. Prueba de proporciones: ¿hay más vuelos con retraso en fin de semana?
# ------------------------

df$FIN_SEMANA <- ifelse(df$DAY_OF_WEEK %in% c(1, 7), "fin", "laboral")
df$CON_RETRASO <- ifelse(df$ARRIVAL_DELAY > 15, 1, 0)

tabla_prop <- table(df$FIN_SEMANA, df$CON_RETRASO)
print(tabla_prop)

prop.table(tabla_prop,1)

cat("Proporción de retrasos:\n")
print(round(prop_fin, 3))

# Test de chi-cuadrado
chi_test <- chisq.test(tabla_prop)
print(chi_test)
# H0: no hay relación (igual)
# H1: Sí hay relación (distinto)

# ------------------------
# 5. Conclusión
# ------------------------

if (chi_test$p.value < 0.05) {
  cat("Existe asociación significativa entre día y retraso ( p <", round(chi_test$p.value, 4), ")\n")
} else {
  cat("No se detecta asociación significativa ( p =", round(chi_test$p.value, 4), ")\n")
  }
                                                           




