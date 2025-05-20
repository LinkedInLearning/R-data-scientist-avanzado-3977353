# SOLUCIONES AL DESAFÍO - Análisis de datos de vuelos

ruta <- "C:/Users/linkedin/Dropbox/LinkedIn/"
df <- read.csv(paste0(ruta,"flights_small.csv"), sep=";")
df

df <- df[!is.na(df$ARRIVAL_DELAY),]

# 1. Limpieza y transformación de datos
colSums(is.na(df))  # Conteo de valores NA

umbral_faltantes <- ncol(df) * 0.5
df <- df[rowSums(is.na(df)) < umbral_faltantes, ]


# 2. Ingeniería de variables
df$LATE_ARRIVAL <- ifelse(df$ARRIVAL_DELAY > 15, 1, 0)

df$HORA_DIA <- cut(df$SCHEDULED_DEPARTURE,
                   breaks = c(-1, 600, 1200, 1800, 2400),
                   labels = c("noche", "mañana", "tarde", "noche_2"))

df$HORA_DIA <- factor(df$HORA_DIA, levels = c("mañana", "tarde", "noche", "noche_2"))

# Convertir a factor
cols_categoricas <- c("AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "HORA_DIA")
df[cols_categoricas] <- lapply(df[cols_categoricas], factor)

# 3. Normalización y escalado
df$DISTANCE_NORM <- (df$DISTANCE - min(df$DISTANCE)) / (max(df$DISTANCE) - min(df$DISTANCE))
df$AIR_TIME_NORM <- (df$AIR_TIME - min(df$AIR_TIME)) / (max(df$AIR_TIME) - min(df$AIR_TIME))

df$DEPARTURE_DELAY_STD <- scale(df$DEPARTURE_DELAY)
df$ARRIVAL_DELAY_STD <- scale(df$ARRIVAL_DELAY)

# 4. Análisis exploratorio
prop.table(table(df$AIRLINE, df$LATE_ARRIVAL), 1)

plot(df$DISTANCE, df$ARRIVAL_DELAY, pch = 20, col = rgb(0,0,1,0.2), main = "Relación DISTANCE vs ARRIVAL_DELAY")
modelo_simple <- lm(ARRIVAL_DELAY ~ DISTANCE, data = df)
abline(modelo_simple, col = "red")

# 5. Clasificación: KNN
set.seed(123)
n <- nrow(df)
indices <- sample(1:n, size = 0.7 * n)
train <- df[indices, ]
test <- df[-indices, ]

# Variables predictoras normalizadas
train_x <- na.omit(data.frame(train$DISTANCE_NORM, train$DEPARTURE_DELAY_STD))
test_x <- na.omit(data.frame(test$DISTANCE_NORM, test$DEPARTURE_DELAY_STD))
train_y <- train$LATE_ARRIVAL

library(class)
knn_pred <- knn(train_x, test_x, cl = train_y, k = 5)

table(Predicho = knn_pred, Real = test$LATE_ARRIVAL)
mean(knn_pred == test$LATE_ARRIVAL)  # Accuracy

# 6. Regresión Lasso
library(glmnet)
x <- model.matrix(ARRIVAL_DELAY ~ DISTANCE + DEPARTURE_DELAY + AIR_TIME, data = df)[, -1]
y <- df$ARRIVAL_DELAY

modelo_lasso <- glmnet(x, y, alpha = 1)
coef(modelo_lasso, s = 0.1)

modelo_lineal <- lm(ARRIVAL_DELAY ~ DISTANCE + DEPARTURE_DELAY + AIR_TIME, data = df)
summary(modelo_lineal)

# 7. Árbol de decisión
library(rpart)
modelo_arbol <- rpart(LATE_ARRIVAL ~ DEPARTURE_DELAY + AIR_TIME + DISTANCE, data = df, 
                      method = "class", control = list(cp = 0.001))
printcp(modelo_arbol)

library(rpart.plot)
rpart.plot(modelo_arbol)

# 8. Clustering (no supervisado)
set.seed(42)
kmeans_res <- kmeans(df[, c("AIR_TIME", "DISTANCE")], centers = 3)
df$cluster <- factor(kmeans_res$cluster)

table(df$cluster, df$LATE_ARRIVAL)

# 9. Serie temporal
df$FECHA <- as.Date(with(df, paste(YEAR, MONTH, DAY, sep = "-")))
vuelos_por_dia <- aggregate(FLIGHT_NUMBER ~ FECHA, data = df, FUN = length)
ts_diaria <- ts(vuelos_por_dia$FLIGHT_NUMBER, frequency = 7)

plot(ts_diaria, main = "Número de vuelos por día")

library(forecast)
modelo_arima <- auto.arima(ts_diaria)
modelo_arima <- arima(ts_diaria, order = c(1,1,1), seasonal = c(2,0,2))

forecast::forecast(modelo_arima, h = 7)
plot(forecast(modelo_arima, h = 70))

# 10. Evaluación de modelos de regresión
library(Metrics)
y_real <- test$ARRIVAL_DELAY
y_pred_lineal <- predict(modelo_lineal, newdata = test[,c("DISTANCE" , "DEPARTURE_DELAY", "AIR_TIME")])

r2 <- 1 - sum((y_real - y_pred_lineal)^2, na.rm = TRUE) / sum((y_real - mean(y_real, na.rm = TRUE))^2, na.rm = TRUE)
r2

# Outliers con IQR
q1 <- quantile(df$DEPARTURE_DELAY, 0.25, na.rm = TRUE)
q3 <- quantile(df$DEPARTURE_DELAY, 0.75, na.rm = TRUE)
iqr <- q3 - q1
limite_inf <- q1 - 1.5 * iqr
limite_sup <- q3 + 1.5 * iqr
outliers <- df$DEPARTURE_DELAY < limite_inf | df$DEPARTURE_DELAY > limite_sup
sum(outliers)
