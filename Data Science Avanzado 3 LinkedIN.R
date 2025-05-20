ruta <- "C:/Users/linkedin/Dropbox/LinkedIn/"

df <- read.csv(paste0(ruta,"flights_small.csv"), sep=";")
df

#################
# Random Forest #
#################

# ------------------------
# 1. Clasificación: ¿el vuelo llegará con retraso?
# ------------------------

# Creamos variable binaria
df$RETRASO_LLEGADA <- ifelse(df$ARRIVAL_DELAY > 15, 1, 0)
df$RETRASO_LLEGADA <- as.factor(df$RETRASO_LLEGADA)

# Dataset de entrada
df_rf_clas <- df[, c("RETRASO_LLEGADA", "DEPARTURE_DELAY", "DISTANCE", "AIR_TIME", "SCHEDULED_TIME")]
df_rf_clas <- na.omit(df_rf_clas)

# Dividir en entrenamiento y prueba
set.seed(123)
n <- nrow(df_rf_clas)
train_idx <- sample(1:n, size = 0.7 * n)
train <- df_rf_clas[train_idx, ]
test <- df_rf_clas[-train_idx, ]

# Entrenar árbol de decisión
library(rpart)
modelo_arbol <- rpart(RETRASO_LLEGADA ~ ., data = train, method = "class")

# Entrenar Random Forest
# install.packages("randomForest")
library(randomForest)
modelo_rf <- randomForest(RETRASO_LLEGADA ~ ., data = train, ntree = 1000)

# Comparar predicciones
pred_arbol <- predict(modelo_arbol, newdata = test, type = "class")
pred_rf <- predict(modelo_rf, newdata = test)

cat("Matriz de confusión - Árbol de decisión:\n")
table(Pred = pred_arbol, Real = test$RETRASO_LLEGADA)
mean(pred_arbol == test$RETRASO_LLEGADA)

cat("\nMatriz de confusión - Random Forest:\n")
table(Pred = pred_rf, Real = test$RETRASO_LLEGADA)
mean(pred_rf == test$RETRASO_LLEGADA)

# ------------------------
# 2. Importancia de variables
# ------------------------

print(importance(modelo_rf))

# Visualizar importancia
varImpPlot(modelo_rf)

# ------------------------
# 3. Regresión: predecir ARRIVAL_DELAY
# ------------------------

df_rf_reg <- df[, c("ARRIVAL_DELAY", "DEPARTURE_DELAY", "DISTANCE", "AIR_TIME", "SCHEDULED_TIME")]
df_rf_reg <- na.omit(df_rf_reg)

set.seed(321)
n <- nrow(df_rf_reg)
train_idx <- sample(1:n, size = 0.7 * n)
train <- df_rf_reg[train_idx, ]
test <- df_rf_reg[-train_idx, ]

# Entrenar modelo
modelo_rf_reg <- randomForest(ARRIVAL_DELAY ~ ., data = train, ntree = 1000)

# Predicciones
pred_rf_reg <- predict(modelo_rf_reg, newdata = test)

# Calcular R²
rss <- sum((test$ARRIVAL_DELAY - pred_rf_reg)^2)
tss <- sum((test$ARRIVAL_DELAY - mean(test$ARRIVAL_DELAY))^2)
r2 <- 1 - rss / tss
cat("R² Random Forest (regresión):", round(r2, 4), "\n")

# ------------------------
# 4. Comparar real vs predicho
# ------------------------

library(ggplot2)

ggplot(data.frame(real = test$ARRIVAL_DELAY, pred = pred_rf_reg),
       aes(x = real, y = pred)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, col = "red") +
  labs(title = "Predicción ARRIVAL_DELAY con Random Forest",
       x = "Real", y = "Predicho")

######################
# XGBoost y LightGBM #
######################

# ------------------------
# 1. Preparar datos para XGBoost
# ------------------------

# Usamos como target ARRIVAL_DELAY
vars <- c("ARRIVAL_DELAY", "DEPARTURE_DELAY", "DISTANCE", "AIR_TIME", "SCHEDULED_TIME")
df_xgb <- df[, vars]
df_xgb <- na.omit(df_xgb)

# Dividir en train y test
set.seed(123)
n <- nrow(df_xgb)
train_idx <- sample(1:n, size = 0.7 * n)
train <- df_xgb[train_idx, ]
test <- df_xgb[-train_idx, ]

# Convertir a formato de matriz para XGBoost
x_train <- as.matrix(train[, -1])
y_train <- train$ARRIVAL_DELAY

x_test <- as.matrix(test[, -1])
y_test <- test$ARRIVAL_DELAY

# ------------------------
# 2. Entrenar modelo XGBoost
# ------------------------

# install.packages("xgboost")
library(xgboost)

modelo_xgb <- xgboost(data = x_train, label = y_train,
                      nrounds = 1000,
                      objective = "reg:squarederror",
                      verbose = 0)

# ------------------------
# 3. Predicción y evaluación
# ------------------------

pred_xgb <- predict(modelo_xgb, newdata = x_test)

# Calcular R²
rss <- sum((y_test - pred_xgb)^2)
tss <- sum((y_test - mean(y_test))^2)
r2 <- 1 - rss / tss
cat("R² XGBoost:", round(r2, 4), "\n")

# ------------------------
# 4. Importancia de variables
# ------------------------

importance <- xgb.importance(model = modelo_xgb)
print(importance)

xgb.plot.importance(importance_matrix = importance)

# ------------------------
# 5. Visualización: real vs predicho
# ------------------------

library(ggplot2)

ggplot(data.frame(real = y_test, pred = pred_xgb),
       aes(x = real, y = pred)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, col = "red") +
  labs(title = "Predicción ARRIVAL_DELAY con XGBoost",
       x = "Real", y = "Predicho")

# ------------------------
# 6.  LightGBM
# ------------------------

# LightGBM requiere instalación manual y configuración previa

library(lightgbm)
dtrain <- lgb.Dataset(data = x_train, label = y_train)

modelo_lgb <- lgb.train(params = list(objective = "regression", metric = "rmse"),
                        data = dtrain,
                        nrounds = 1000)

pred_lgb <- predict(modelo_lgb, x_test)
rss_lgb <- sum((y_test - pred_lgb)^2)
r2_lgb <- 1 - rss_lgb / tss
cat("R² LightGBM:", round(r2_lgb, 4), "\n")

###############
# SHAP VALUES #
###############

# Interpretación de modelos en R con SHAP
# SHapley Additive exPlanations
# cuánto contribuye cada variable individual a la predicción de un modelo

# ------------------------
# 1. Preparar los datos
# ------------------------

vars <- c("ARRIVAL_DELAY", "DEPARTURE_DELAY", "DISTANCE", "AIR_TIME", "SCHEDULED_TIME")
df_shap <- df[, vars]
df_shap <- na.omit(df_shap)

set.seed(123)
n <- nrow(df_shap)
train_idx <- sample(1:n, size = 0.7 * n)
train <- df_shap[train_idx, ]
test <- df_shap[-train_idx, ]

x_train <- as.matrix(train[, -1])
y_train <- train$ARRIVAL_DELAY

x_test <- as.matrix(test[, -1])
y_test <- test$ARRIVAL_DELAY

# ------------------------
# 2. Entrenar modelo XGBoost
# ------------------------

library(xgboost)

modelo_xgb <- xgboost(data = x_train, label = y_train,
                      nrounds = 100,
                      objective = "reg:squarederror",
                      verbose = 0)

# ------------------------
# 3. Calcular valores SHAP
# ------------------------

# install.packages("SHAPforxgboost")
library(SHAPforxgboost)

# Calcular los valores SHAP para los datos de test
shap_values <- shap.values(xgb_model = modelo_xgb, X_train = x_test)

# Importancia global de las variables
shap_importance <- shap_values$mean_shap_score
print(round(shap_importance, 3))

# ------------------------
# 4. Gráfico de importancia global
# ------------------------


# Preparamos los datos SHAP en formato largo
shap_long <- shap.prep(shap_contrib = shap_values$shap_score, X_train = x_test)

# Ahora sí, se puede graficar correctamente
shap.plot.summary(shap_long)


# ------------------------
# 5. Dependencia individual (efecto marginal de una variable)
# ------------------------

shap.plot.dependence(data_long = shap.prep(shap_contrib = shap_values$shap_score,
                                           X_train = x_test),
                     x = "DEPARTURE_DELAY", 
                     y = "DEPARTURE_DELAY")

# ------------------------
# 6. Interpretación local: observación individual
# ------------------------

library(fastshap)

# Supón que model es un modelo con función predict
predict_wrapper <- function(object, newdata) {
  predict(object, newdata = as.matrix(newdata))
}

# Paso 2: calcular SHAP values
shap_vals <- explain(modelo_xgb, X = x_test, pred_wrapper = predict_wrapper, nsim = 100)

# Paso 3: elegir una observación para visualizar (por ejemplo, la primera)
obs <- 3
df_plot <- data.frame(
  feature = colnames(x_test),
  shap_value = shap_vals[obs, ],
  feature_value = unlist(x_test[obs, ])
)

# Paso 4: crear gráfico tipo barplot horizontal
ggplot(df_plot, aes(x = reorder(feature, shap_value), y = shap_value, fill = shap_value > 0)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  labs(
    title = paste("SHAP values for observation", obs),
    x = "Feature",
    y = "SHAP value"
  ) +
  theme_minimal()

###################
# Neural Networks #
###################

# Redes neuronales en R con nnet (regresión simple)
# https://playground.tensorflow.org/

# ------------------------
# 1. Preparación de datos
# ------------------------

# Selección de variables numéricas
vars <- c("ARRIVAL_DELAY", "DEPARTURE_DELAY", "DISTANCE", "SCHEDULED_TIME", "AIR_TIME")
df_nn <- df[, vars]
df_nn <- na.omit(df_nn)

# Normalizar todas las variables entre 0 y 1
normalizar <- function(x) (x - min(x)) / (max(x) - min(x))
df_nn_norm <- as.data.frame(lapply(df_nn, normalizar))

# Dividir en train y test
set.seed(123)
n <- nrow(df_nn_norm)
train_idx <- sample(1:n, size = 0.7 * n)
train <- df_nn_norm[train_idx, ]
test <- df_nn_norm[-train_idx, ]

# ------------------------
# 2. Entrenar red neuronal
# ------------------------

# install.packages("nnet")
library(nnet)

# Fórmula: ARRIVAL_DELAY ~ todas las demás
modelo_nn <- nnet(ARRIVAL_DELAY ~ ., data = train, size = 5, linout = TRUE, maxit = 500, trace = FALSE)

# ------------------------
# 3. Evaluar el modelo
# ------------------------

# Predicciones sobre test
pred <- predict(modelo_nn, newdata = test)

# R² (como regresión)
rss <- sum((test$ARRIVAL_DELAY - pred)^2)
tss <- sum((test$ARRIVAL_DELAY - mean(test$ARRIVAL_DELAY))^2)
r2 <- 1 - rss / tss
cat("R² red neuronal:", round(r2, 4), "\n")

# ------------------------
# 4. Visualización de predicción
# ------------------------

library(ggplot2)

ggplot(data.frame(real = test$ARRIVAL_DELAY, pred = pred),
       aes(x = real, y = pred)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, col = "red") +
  labs(title = "Predicción ARRIVAL_DELAY con red neuronal",
       x = "Real", y = "Predicho")

# ------------------------
# 5. Red con más nodos o capas
# ------------------------

# Otra red con 10 nodos ocultos
modelo_nn_2 <- nnet(ARRIVAL_DELAY ~ ., data = train, size = 10, linout = TRUE, maxit = 500, trace = FALSE)

pred2 <- predict(modelo_nn_2, newdata = test)
rss2 <- sum((test$ARRIVAL_DELAY - pred2)^2)
r2_2 <- 1 - rss2 / tss
cat("R² red neuronal (10 nodos):", round(r2_2, 4), "\n")

###########
# Halving #
###########


# ------------------------
# 1. Preparar los datos
# ------------------------

vars <- c("ARRIVAL_DELAY", "DEPARTURE_DELAY", "DISTANCE", "SCHEDULED_TIME", "AIR_TIME")
df_hp <- df[, vars]
df_hp <- na.omit(df_hp)

set.seed(123)
n <- nrow(df_hp)
train_idx <- sample(1:n, size = 0.7 * n)
train <- df_hp[train_idx, ]
test <- df_hp[-train_idx, ]

# ------------------------
# 2. Configurar estrategia de búsqueda adaptativa
# ------------------------

# CP

# install.packages("caret")
library(caret)

control <- trainControl(
  method = "adaptive_cv",
  number = 10,
  repeats = 2,
  verboseIter = TRUE,
  adaptive = list(
    min = 3,
    alpha = 0.05,
    method = "gls",
    complete = TRUE
  )
)

# ------------------------
# 3. Entrenamiento con árbol de decisión (rpart)
# ------------------------

set.seed(123)
modelo_arbol <- train(
  ARRIVAL_DELAY ~ ., data = train,
  method = "rpart",
  trControl = control,
  tuneLength = 100 # combinaciones de cp va a probar
)

print(modelo_arbol)

modelo_arbol$bestTune

# Optimización de hiperparámetros en árboles de decisión con rpart y adaptive_cv

# ------------------------
# 1. Preparación de los datos
# ------------------------

# Búsqueda manual de hiperparámetros en rpart

# ------------------------
# 1. Preparar datos
# ------------------------

vars <- c("ARRIVAL_DELAY", "DEPARTURE_DELAY", "DISTANCE", "SCHEDULED_TIME", "AIR_TIME")
df_hp <- df[, vars]
df_hp <- na.omit(df_hp)

set.seed(123)
n <- nrow(df_hp)
train_idx <- sample(1:n, size = 0.7 * n)
train <- df_hp[train_idx, ]
test <- df_hp[-train_idx, ]

# ------------------------
# 2. Crear combinaciones de hiperparámetros
# ------------------------

param_grid <- expand.grid(
  maxdepth = c(4, 6, 8),
  minsplit = c(10, 20),
  minbucket = c(5, 10)
)

# Inicializar tabla de resultados
resultados <- data.frame()

# ------------------------
# 3. Bucle para entrenar un modelo por combinación
# ------------------------

library(rpart)

for (i in 1:nrow(param_grid)) {
  
  params <- param_grid[i, ]
  
  modelo <- rpart(
    ARRIVAL_DELAY ~ ., data = train,
    method = "anova",
    control = rpart.control(
      maxdepth = params$maxdepth,
      minsplit = params$minsplit,
      minbucket = params$minbucket,
      cp = 0  # sin poda automática, queremos ver el efecto de los parámetros
    )
  )
  pred <- predict(modelo, newdata = test)
  
  # Calcular R²
  rss <- sum((test$ARRIVAL_DELAY - pred)^2)
  tss <- sum((test$ARRIVAL_DELAY - mean(test$ARRIVAL_DELAY))^2)
  r2 <- 1 - rss / tss
  
  resultados <- rbind(resultados, cbind(params, R2 = round(r2, 4)))
}

# ------------------------
# 4. Ver resultados ordenados por R²
# ------------------------

resultados <- resultados[order(-resultados$R2), ]
print(resultados)

# ------------------------
# 5. Visualizar el mejor modelo
# ------------------------

# Volver a entrenar el mejor modelo
mejores <- resultados[1, ]
modelo_final <- rpart(
  ARRIVAL_DELAY ~ ., data = train,
  method = "anova",
  control = rpart.control(
    maxdepth = mejores$maxdepth,
    minsplit = mejores$minsplit,
    minbucket = mejores$minbucket,
    cp = 0
  )
)

# Mostrar árbol
library(rpart.plot)
rpart.plot(modelo_final)

# Visualización de predicción vs real
library(ggplot2)
pred_mejor <- predict(modelo_final, newdata = test)

ggplot(data.frame(real = test$ARRIVAL_DELAY, pred = pred_mejor),
       aes(x = real, y = pred)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, col = "red") +
  labs(title = "Árbol optimizado: predicción ARRIVAL_DELAY",
       x = "Real", y = "Predicho")


#######
# H2O #
#######

# Automatización de modelos con h2o

# ------------------------
# 1. Preparar entorno
# ------------------------

# install.packages("h2o")  # si no está instalado
library(h2o) # Java version

# Iniciar el entorno h2o
h2o.init(max_mem_size = "2G", nthreads = -1)
df <- df[sample(1:10000),]

# ------------------------
# 2. Preparar los datos
# ------------------------

# Selección de variables
vars <- c("ARRIVAL_DELAY", "DEPARTURE_DELAY", "DISTANCE", "SCHEDULED_TIME", "AIR_TIME")
df_h2o <- df[, vars]
df_h2o <- na.omit(df_h2o)

# Convertir a formato h2o
df_h2o_hex <- as.h2o(df_h2o)

# Separar en train/test (70/30)
splits <- h2o.splitFrame(df_h2o_hex, ratios = 0.7, seed = 123)
train <- splits[[1]]
test <- splits[[2]]

# Definir variables
y <- "ARRIVAL_DELAY"
x <- setdiff(colnames(df_h2o), y)

# ------------------------
# 3. Ejecutar H2O AutoML
# ------------------------

automl_modelos <- h2o.automl(
  x = x,
  y = y,
  training_frame = train,
  leaderboard_frame = test,
  max_runtime_secs = 60,     # límite de tiempo
  stopping_metric = "RMSE",
  sort_metric = "RMSE",
  seed = 123
)

# ------------------------
# 4. Ver los mejores modelos
# ------------------------

leaderboard <- automl_modelos@leaderboard
print(leaderboard)

# Mejor modelo
mejor_modelo <- automl_modelos@leader
print(mejor_modelo)

# ------------------------
# 5. Evaluar sobre test
# ------------------------

perf <- h2o.performance(mejor_modelo, newdata = test)
r2 <- h2o.r2(perf)
rmse <- h2o.rmse(perf)

cat("R² modelo automático:", round(r2, 4), "\n")
cat("RMSE:", round(rmse, 2), "\n")

# ------------------------
# 6. Visualizar predicción vs real
# ------------------------

pred <- as.data.frame(h2o.predict(mejor_modelo, test))[,1]
real <- as.data.frame(test)[, y]

library(ggplot2)

ggplot(data.frame(real = real, pred = pred), aes(x = real, y = pred)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Predicción ARRIVAL_DELAY con H2O AutoML",
       x = "Real", y = "Predicho")

# ------------------------
# 7. Finalizar sesión h2o
# ------------------------

h2o.shutdown(prompt = FALSE)


###############
# Exportación #
###############

# Exportación y despliegue de modelos en R

# ------------------------
# 1. Entrenamiento del modelo
# ------------------------

# Crear variable objetivo binaria
df$RETRASO_LLEGADA <- ifelse(df$ARRIVAL_DELAY > 15, 1, 0)
df$RETRASO_LLEGADA <- as.factor(df$RETRASO_LLEGADA)

# Selección de variables
vars <- c("RETRASO_LLEGADA", "DEPARTURE_DELAY", "DISTANCE", "AIR_TIME", "SCHEDULED_TIME")
df_export <- df[, vars]
df_export <- na.omit(df_export)

# Dividir en train/test
set.seed(123)
n <- nrow(df_export)
train_idx <- sample(1:n, size = 0.7 * n)
train <- df_export[train_idx, ]
test <- df_export[-train_idx, ]

# Entrenar modelo
library(randomForest)
modelo_rf <- randomForest(RETRASO_LLEGADA ~ ., data = train, ntree = 100)

# ------------------------
# 2. Exportar modelo entrenado
# ------------------------

saveRDS(modelo_rf, file = "modelo_retraso_rf.rds")
cat("Modelo guardado como modelo_retraso_rf.rds\n")

# ------------------------
# 3. Cargar el modelo en otro entorno
# ------------------------

# En un nuevo script, o sesión distinta:
modelo_cargado <- readRDS("modelo_retraso_rf.rds")
cat("Modelo cargado correctamente desde disco\n")

# ------------------------
# 4. Usar modelo para hacer predicciones
# ------------------------

# Predicción sobre test
pred <- predict(modelo_cargado, newdata = test)

# Evaluación
accuracy <- mean(pred == test$RETRASO_LLEGADA)
cat("Precisión del modelo cargado:", round(accuracy * 100, 2), "%\n")

# ------------------------
# 5. Guardar predicciones como resultado
# ------------------------

resultado <- data.frame(
  id = rownames(test),
  PREDICCION = pred
)

write.csv(resultado, file = "predicciones_retraso.csv", row.names = FALSE)
cat("Predicciones guardadas en predicciones_retraso.csv\n")



