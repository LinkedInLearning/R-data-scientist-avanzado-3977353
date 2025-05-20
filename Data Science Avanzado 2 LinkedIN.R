ruta <- "C:/Users/linkedin/Dropbox/LinkedIn/"

df <- read.csv(paste0(ruta,"flights_small.csv"), sep=";")
df


# KNN en R con el dataset df

# ------------------------
# 1. Preparar datos
# ------------------------

# Creamos variable objetivo binaria
df$RETRASO_LLEGADA <- ifelse(df$ARRIVAL_DELAY > 15, 1, 0)

# Usamos un subconjunto de variables numéricas simples
df_knn <- df[, c("RETRASO_LLEGADA", "DISTANCE", "DEPARTURE_DELAY", "AIR_TIME")]
df_knn <- na.omit(df_knn)

# Convertir la variable objetivo a factor (requisito de knn)
df_knn$RETRASO_LLEGADA <- as.factor(df_knn$RETRASO_LLEGADA)

# ------------------------
# 2. Dividir en entrenamiento y prueba
# ------------------------

set.seed(123)
n <- nrow(df_knn)
indices <- sample(1:n, size = 0.7 * n)

train <- df_knn[indices, ]
test <- df_knn[-indices, ]

# ------------------------
# 3. Escalar variables numéricas
# ------------------------

# Usamos media y desviación del conjunto de entrenamiento
media <- apply(train[, -1], 2, mean)
desv <- apply(train[, -1], 2, sd)

train_scaled <- scale(train[, -1], center = media, scale = desv)
test_scaled <- scale(test[, -1], center = media, scale = desv)

# ------------------------
# 4. Aplicar KNN
# ------------------------

library(class)

# Elegimos un valor inicial de k = 5
predicciones <- knn(train = train_scaled,
                    test = test_scaled,
                    cl = train$RETRASO_LLEGADA,
                    k = 5)

# ------------------------
# 5. Evaluar resultados
# ------------------------

# Matriz de confusión
table(predicciones, test$RETRASO_LLEGADA)

prop.table(table(test$RETRASO_LLEGADA))
# Calcular accuracy
accuracy <- mean(predicciones == test$RETRASO_LLEGADA)
print(paste("Precisión con k = 5:", round(accuracy * 100, 2), "%"))

# ------------------------
# 6. Probar distintos valores de k
# ------------------------

resultados <- data.frame(k = integer(), accuracy = numeric())

for (k in 1:100) {
  pred <- knn(train = train_scaled,
              test = test_scaled,
              cl = train$RETRASO_LLEGADA,
              k = k)
  acc <- mean(pred == test$RETRASO_LLEGADA)
  resultados <- rbind(resultados, data.frame(k = k, accuracy = acc))
}

# ------------------------
# 7. Visualización del resultado
# ------------------------

library(ggplot2)

ggplot(resultados, aes(x = k, y = accuracy)) +
  geom_line() +
  geom_point() +
  labs(title = "Precisión según número de vecinos (k)",
       x = "k", y = "Precisión") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1))

# ------------------------
# 8. Elegir el mejor modelo
# ------------------------

mejor_k <- resultados$k[which.max(resultados$accuracy)]
cat("Mejor valor de k:", mejor_k, "\n")

########################
# Modelos de Regresión #
########################

# Predicción de ARRIVAL_DELAY con regresión lineal, Lasso y Ridge

# ------------------------
# 1. Preparación de datos
# ------------------------

# Seleccionamos variables predictoras numéricas
vars <- c("ARRIVAL_DELAY", "DISTANCE", "DEPARTURE_DELAY", # "SCHEDULED_TIME",
          "TAXI_OUT", "TAXI_IN", "AIR_TIME" , "ELAPSED_TIME", "AIRLINE"
)

df_reg <- df[, vars]
df_reg <- na.omit(df_reg)

# Matriz de predictores y variable objetivo
x <- as.matrix(df_reg[, -1])  # todas menos ARRIVAL_DELAY
y <- df_reg$ARRIVAL_DELAY

# ------------------------
# 2. Regresión lineal clásica
# ------------------------

modelo_lm <- lm(ARRIVAL_DELAY ~ ., data = df_reg)
summary(modelo_lm)

# Predicciones
pred_lm <- predict(modelo_lm, newdata = df_reg)

# Cálculo de R²
rss_lm <- sum((y - pred_lm)^2)
tss <- sum((y - mean(y))^2)
r2_lm <- 1 - rss_lm / tss
cat("R² modelo lineal:", round(r2_lm, 4), "\n")

# ------------------------
# 3. Regresión Ridge
# ------------------------

library(glmnet)

cv_ridge <- cv.glmnet(x, y, alpha = 0)
modelo_ridge <- glmnet(x, y, alpha = 0, lambda = cv_ridge$lambda.min)

pred_ridge <- predict(modelo_ridge, newx = x)
rss_ridge <- sum((y - pred_ridge)^2)
r2_ridge <- 1 - rss_ridge / tss
cat("R² modelo Ridge:", round(r2_ridge, 4), "\n")

# ------------------------
# 4. Regresión Lasso
# ------------------------

cv_lasso <- cv.glmnet(x, y, alpha = 1)
modelo_lasso <- glmnet(x, y, alpha = 1, lambda = cv_lasso$lambda.min)

pred_lasso <- predict(modelo_lasso, newx = x)
rss_lasso <- sum((y - pred_lasso)^2)
r2_lasso <- 1 - rss_lasso / tss
cat("R² modelo Lasso:", round(r2_lasso, 4), "\n")

# ------------------------
# 5. Comparación de coeficientes
# ------------------------

coef_lm <- coef(modelo_lm)
coef_ridge <- as.vector(coef(modelo_ridge))
coef_lasso <- as.vector(coef(modelo_lasso))

nombres_vars <- rownames(coef(modelo_lasso))
tabla_coef <- data.frame(
  Variable = nombres_vars,
  Lineal = round(coef_lm, 4),
  Ridge = round(coef_ridge, 4),
  Lasso = round(coef_lasso, 4)
)
print(tabla_coef)

# ------------------------
# 6. Visualización de los coeficientes
# ------------------------

library(ggplot2)
library(reshape2)

tabla_coef_plot <- tabla_coef[-1, ]  # quitamos el "punto de corte"
datos_plot <- melt(tabla_coef_plot, id.vars = "Variable",
                   variable.name = "Modelo", value.name = "Coeficiente")

ggplot(datos_plot, aes(x = Variable, y = Coeficiente, fill = Modelo)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Coeficientes: predicción de ARRIVAL_DELAY") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

###########
# Árboles #
###########

# Árboles de decisión para regresión y clasificación en R

# ------------------------
# 1. Clasificación: ¿habrá retraso en llegada?
# ------------------------

# Variable binaria: 1 si el retraso es mayor a 15 minutos
df$RETRASO_LLEGADA <- ifelse(df$ARRIVAL_DELAY > 15, 1, 0)
df$RETRASO_LLEGADA <- as.factor(df$RETRASO_LLEGADA)

# Selección de variables simples
df_arbol_clas <- df[, c("RETRASO_LLEGADA", "DEPARTURE_DELAY", "DISTANCE", "SCHEDULED_TIME")]
df_arbol_clas <- na.omit(df_arbol_clas)

# Dividir en entrenamiento y prueba
set.seed(123)
n <- nrow(df_arbol_clas)
train_idx <- sample(1:n, size = 0.7 * n)
train <- df_arbol_clas[train_idx, ]
test <- df_arbol_clas[-train_idx, ]

# Entrenar árbol de clasificación
library(rpart)
modelo_clas <- rpart(RETRASO_LLEGADA ~ ., data = train, method = "class",control = rpart.control(cp = 0.001))

# Visualizar árbol
library(rpart.plot) # install.packages("install.packages")
rpart.plot(modelo_clas)

# Predicciones y matriz de confusión
pred_clas <- predict(modelo_clas, newdata = test, type = "class")
table(Predicho = pred_clas, Real = test$RETRASO_LLEGADA)

# ------------------------
# 2. Regresión: predecir ARRIVAL_DELAY
# ------------------------

df_arbol_reg <- df[, c("ARRIVAL_DELAY", "DEPARTURE_DELAY", "DISTANCE", "TAXI_OUT", "AIR_TIME", "AIRLINE")]
df_arbol_reg <- na.omit(df_arbol_reg)

# División de datos
set.seed(456)
n <- nrow(df_arbol_reg)
train_idx <- sample(1:n, size = 0.7 * n)
train <- df_arbol_reg[train_idx, ]
test <- df_arbol_reg[-train_idx, ]

# Entrenar árbol de regresión
modelo_reg <- rpart(ARRIVAL_DELAY ~ ., data = train, method = "anova"#,control = rpart.control(cp = 0.001)
)

# Visualizar árbol
rpart.plot(modelo_reg)

# Predicciones
pred_reg <- predict(modelo_reg, newdata = test)

# Evaluar con R²
rss <- sum((test$ARRIVAL_DELAY - pred_reg)^2)
tss <- sum((test$ARRIVAL_DELAY - mean(test$ARRIVAL_DELAY))^2)
r2 <- 1 - rss / tss
cat("R² árbol de regresión:", round(r2, 4), "\n")

# ------------------------
# 3. Comparar valores reales y predichos (regresión)
# ------------------------

library(ggplot2)

ggplot(data.frame(real = test$ARRIVAL_DELAY, pred = pred_reg),
       aes(x = real, y = pred)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, col = "red") +
  labs(title = "Árbol de regresión: ARRIVAL_DELAY",
       x = "Valor real", y = "Valor predicho")

##############
# Clustering #
##############

# Comparación de algoritmos de clustering en R

# ------------------------
# 1. Preparar datos
# ------------------------

# Seleccionamos dos variables numéricas para visualizar fácilmente
df_clust <- df[, c("DISTANCE", "AIR_TIME")]
df_clust <- na.omit(df_clust)

# Escalar las variables
df_scaled <- scale(df_clust)

# ------------------------
# 2. K-means clustering
# ------------------------

set.seed(123)
modelo_kmeans <- kmeans(df_scaled, centers = 3, nstart = 25)

modelo_kmeans$centers

# Añadir cluster al dataset
df_clust$cluster_kmeans <- as.factor(modelo_kmeans$cluster)

# Visualizar
library(ggplot2)

ggplot(df_clust, aes(x = DISTANCE, y = AIR_TIME, color = cluster_kmeans)) +
  geom_point(alpha = 0.5) +
  labs(title = "Clustering con K-means (3 grupos)")

# ------------------------
# 3. Clustering jerárquico
# ------------------------

# Distancia euclídea
distancias <- dist(df_scaled)

# Algoritmo jerárquico
modelo_hclust <- hclust(distancias, method = "ward.D2")

# Visualizar dendrograma
plot(modelo_hclust, labels = FALSE, main = "Dendrograma - Clustering jerárquico")

# Cortar en 3 grupos
grupos_hclust <- cutree(modelo_hclust, k = 3)

df_clust$cluster_hclust <- as.factor(grupos_hclust)

# Visualizar
ggplot(df_clust, aes(x = DISTANCE, y = AIR_TIME, color = cluster_hclust)) +
  geom_point(alpha = 0.5) +
  labs(title = "Clustering jerárquico (3 grupos)")

# ------------------------
# 4. DBSCAN
# ------------------------

# install.packages("dbscan")
library(dbscan)

# Usamos eps y minPts simples
modelo_dbscan <- dbscan(df_scaled, eps = 0.1, minPts = 10)

df_clust$cluster_dbscan <- as.factor(modelo_dbscan$cluster)

# Visualizar
ggplot(df_clust, aes(x = DISTANCE, y = AIR_TIME, color = cluster_dbscan)) +
  geom_point(alpha = 0.5) +
  labs(title = "Clustering con DBSCAN")

# ------------------------
# 5. Comparación general
# ------------------------

# Tabla con número de observaciones por grupo
cat("K-means:\n")
print(table(df_clust$cluster_kmeans))

cat("\nJerárquico:\n")
print(table(df_clust$cluster_hclust))

cat("\nDBSCAN:\n")
print(table(df_clust$cluster_dbscan))

