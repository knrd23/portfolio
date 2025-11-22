# === 0. KONFIGURACJA: Ładowanie Bibliotek ===
set.seed(123)

library(mlbench)     # Dla danych DNA
library(dplyr)       # Do manipulacji danymi
library(tidymodels)  # Do podziału danych i metryk
library(MASS)        # Dla stepAIC
library(caret)       # Do walidacji .632 (train)
library(pls)         # Do plsr
# === 0. SETUP: Loading Libraries ===
# Set seed for reproducibility
set.seed(123)

library(mlbench)     # For DNA data
library(dplyr)       # For data manipulation
library(tidymodels)  # For data splitting and metrics
library(MASS)        # For stepAIC
library(caret)       # For .632 validation (train)
library(pls)         # For plsr
library(boot)        # For cv.glm

# === 1. TASK 1: Data Preparation and Splitting ===

data("DNA")

# Step 1.1: Filtering and preparing data
df_filtered <- DNA %>%
  filter(Class %in% c("ei", "ie")) %>%
  mutate(
    Class = factor(Class, levels = c("ei", "ie")),
    temp_id = row_number()
  )

# Step 1.2: Splitting into sets (guaranteeing 100/100 in test set)
test_set <- df_filtered %>%
  group_by(Class) %>%
  slice_sample(n = 100) %>%
  ungroup()

train_set <- df_filtered %>%
  anti_join(test_set, by = "temp_id")

# Step 1.3: Cleaning up
train_set <- train_set %>% dplyr::select(-temp_id)
test_set <- test_set %>% dplyr::select(-temp_id)

# Verification
cat("Training set size (rows):", nrow(train_set), "\n")
cat("Test set size (rows):", nrow(test_set), "\n")
print(table(test_set$Class))


# === 2. TASK 2: Model Building and Selection ===

# --- Model 1: Partial Least Squares (PLS) ---
# Step 2.1: Finding optimal number of PLS components (LOO on training set)
train_set_for_pls <- train_set %>%
  mutate(Class = ifelse(Class == "ei", 0, 1))

pls_loocv <- plsr(Class ~ .,
                  data = train_set_for_pls,
                  ncomp = 20,
                  validation = "LOO")

ncomp_optimal <- 9 # Determined based on summary/plot analysis
cat("\nSelected PLS model with optimal components:", ncomp_optimal, "\n")


# --- Model 2: Logistic Regression (BIC vs AIC Selection) ---
# Step 2.2: Building candidate models
model_null <- glm(Class ~ 1, data = train_set, family = "binomial")
model_full <- glm(Class ~ ., data = train_set, family = "binomial")

# Step 2.3: Forward Selection (AIC and BIC)
model_aic <- stepAIC(model_null, scope = list(lower=model_null, upper=model_full), direction="forward", trace=FALSE)
model_bic <- stepAIC(model_null, scope = list(lower=model_null, upper=model_full), direction="forward", k=log(nrow(train_set)), trace=FALSE)

# --- METHODOLOGICAL CORRECTION ---
# Step 2.4: Selection between AIC and BIC using ONLY the training set.
# We use simple 10-fold Cross-Validation (CV) to check which of the two models
# performs better/is more stable on the training data.

# Cost function (classification error)
cost_func <- function(r, pi = 0) mean(abs(r - pi) > 0.5)

set.seed(123)
cv_aic <- cv.glm(train_set, model_aic, cost = cost_func, K = 10)$delta[1]

set.seed(123)
cv_bic <- cv.glm(train_set, model_bic, cost = cost_func, K = 10)$delta[1]

cat("\nCV Error (10-fold) on training set:\n")
cat("Model AIC:", cv_aic, "\n")
cat("Model BIC:", cv_bic, "\n")

# Automatic decision based on lower CV error on training set
if (cv_bic <= cv_aic) {
  cat("DECISION: Selecting BIC model (simpler and/or better CV error).\n")
  form_logistic_final <- formula(model_bic)
  best_logreg_name <- "LogReg_BIC"
} else {
  cat("DECISION: Selecting AIC model (lower CV error).\n")
  form_logistic_final <- formula(model_aic)
  best_logreg_name <- "LogReg_AIC"
}


# === 3. TASK 3: Error Estimation (.632 vs. Test Error) ===

# Step 3.1: List of models to test (only the winners from the training phase)
models_to_test <- list(
  PLS_Optimal = list(
    formula = Class ~ ., method = "pls", 
    tuneGrid = data.frame(ncomp = ncomp_optimal), 
    preProc = c("center", "scale"), family = NULL
  ),
  LogReg_Winner = list( # Either AIC or BIC, depending on the decision above
    formula = form_logistic_final, method = "glm", 
    tuneGrid = NULL, preProc = NULL, family = "binomial"
  )
)

# Step 3.2: .632 Validation Settings
ctrl_boot632 <- trainControl(method = "boot632", number = 300, savePredictions = "final")

# Step 3.3: Evaluation Loop
estimation_results <- list()

for (model_name in names(models_to_test)) {
  cat(paste("\nProcessing:", model_name, "\n"))
  model_params <- models_to_test[[model_name]]
  
  set.seed(123)
  trained_model <- train(
    model_params$formula, data = train_set, 
    method = model_params$method, family = model_params$family,
    trControl = ctrl_boot632, tuneGrid = model_params$tuneGrid, 
    preProc = model_params$preProc
  )
  
  # .632 Estimated Error
  err_632 <- 1 - trained_model$results$Accuracy
  
  # Actual Test Error (Only touching test_set now!)
  pred_test <- predict(trained_model, newdata = test_set)
  err_test <- 1 - mean(pred_test == test_set$Class)
  
  comparison <- "perfect estimation"
  if (err_632 > err_test) comparison <- "OVERESTIMATED"
  if (err_632 < err_test) comparison <- "UNDERESTIMATED"
  
  estimation_results[[model_name]] <- list(
    Est_Error_632 = err_632,
    Actual_Test_Error = err_test,
    Conclusion = comparison
  )
}

cat("\n=== TASK 3 SUMMARY: Error Comparison ===\n")
print(do.call(rbind, lapply(estimation_results, as.data.frame.list)))


# === 4. TASK 4: Synthetic Data (PLS vs PCR) ===
cat("\n--- Task 4: Synthetic Data ---\n")
set.seed(123)
# Noise (high variance X, no correlation with Y)
noise = rnorm(100, 0, 3)
X1 = noise + rnorm(100, 0.1, 0.1)
# Signal (low variance X, high correlation with Y)
signal = rnorm(100, 0, 1)
X2 = signal + rnorm(100, 0, 0.01)
# Y depends only on signal
Y = 2 * signal + rnorm(100, 0, 0.001)

df_syn = data.frame(Y, X1, X2)

# PCR (will select X1 - noise)
pcr_res <- pcr(Y ~ ., data = df_syn, ncomp = 1, validation = "LOO")
# PLS (will select X2 - signal)
pls_res <- plsr(Y ~ ., data = df_syn, ncomp = 1, validation = "LOO")

cat("PCR Loadings (Comp 1):\n"); print(pcr_res$loadings)
cat("PLS Loadings (Comp 1):\n"); print(pls_res$loadings)
# === 1. ZADANIE 1: Przygotowanie i Podział Danych ===

data("DNA")

# Krok 1.1: Filtrowanie i przygotowanie danych
df_filtered <- DNA %>%
  filter(Class %in% c("ei", "ie")) %>%
  mutate(
    Class = factor(Class, levels = c("ei", "ie")),
    temp_id = row_number()
  )

# Krok 1.2: Podział na zbiory (100/100 w teście)
test_set <- df_filtered %>%
  group_by(Class) %>%
  slice_sample(n = 100) %>%
  ungroup()

train_set <- df_filtered %>%
  anti_join(test_set, by = "temp_id")

# Krok 1.3: Czyszczenie
train_set <- train_set %>% dplyr::select(-temp_id)
test_set <- test_set %>% dplyr::select(-temp_id)

# Sprawdzenie
cat("Zbiór treningowy:", nrow(train_set), "\n")
cat("Zbiór testowy:", nrow(test_set), "\n")
print(table(test_set$Class))


# === 2. ZADANIE 2: Budowanie i Wybór Modeli ===

# --- Model 1: Partial Least Squares (PLS) ---
# Krok 2.1: Znalezienie optymalnej liczby komponentów PLS (LOO na treningowym)
train_set_dla_pls <- train_set %>%
  mutate(Class = ifelse(Class == "ei", 0, 1))

pls_loocv <- plsr(Class ~ .,
                  data = train_set_dla_pls,
                  ncomp = 20,
                  validation = "LOO")

ncomp_optimal <- 9 # (Na podstawie analizy wykresu/summary)
cat("\nWybrano model PLS z liczbą komponentów:", ncomp_optimal, "\n")


# --- Model 2: Regresja Logistyczna (Wybór BIC vs AIC) ---
# Krok 2.2: Budowa modeli kandydatów
model_null <- glm(Class ~ 1, data = train_set, family = "binomial")
model_full <- glm(Class ~ ., data = train_set, family = "binomial")

# Krok 2.3: Selekcja progresywna (AIC i BIC)
model_aic <- stepAIC(model_null, scope = list(lower=model_null, upper=model_full), direction="forward", trace=FALSE)
model_bic <- stepAIC(model_null, scope = list(lower=model_null, upper=model_full), direction="forward", k=log(nrow(train_set)), trace=FALSE)

# --- POPRAWKA METODOLOGICZNA ---
# Krok 2.4: Wybór między AIC a BIC przy użyciu  zbioru treningowego.
# Używamy prostej 10-krotnej walidacji krzyżowej (CV), aby sprawdzić, 
# który z tych dwóch modeli jest lepszy/bardziej stabilny na danych treningowych.

library(boot) # Do cv.glm

# Funkcja kosztu (błąd klasyfikacji)
cost_func <- function(r, pi = 0) mean(abs(r - pi) > 0.5)

set.seed(123)
cv_aic <- cv.glm(train_set, model_aic, cost = cost_func, K = 10)$delta[1]

set.seed(123)
cv_bic <- cv.glm(train_set, model_bic, cost = cost_func, K = 10)$delta[1]

cat("\nBłąd CV (10-fold) na zbiorze treningowym:\n")
cat("Model AIC:", cv_aic, "\n")
cat("Model BIC:", cv_bic, "\n")

# Automatyczna decyzja na podstawie mniejszego błędu CV na treningu
if (cv_bic <= cv_aic) {
  cat("DECYZJA: Wybieram model BIC (prostszy i/lub lepszy błąd CV).\n")
  form_logistic_final <- formula(model_bic)
  best_logreg_name <- "LogReg_BIC"
} else {
  cat("DECYZJA: Wybieram model AIC (mniejszy błąd CV).\n")
  form_logistic_final <- formula(model_aic)
  best_logreg_name <- "LogReg_AIC"
}


# === 3. ZADANIE 3: Szacowanie Błędów (.632 vs. Testowy) ===

# Krok 3.1: Lista modeli do testowania (tylko zwycięzcy z etapu treningu)
models_to_test <- list(
  PLS_Optimal = list(
    formula = Class ~ ., method = "pls", 
    tuneGrid = data.frame(ncomp = ncomp_optimal), 
    preProc = c("center", "scale"), family = NULL
  ),
  LogReg_Winner = list( # To będzie albo AIC albo BIC, zależnie od decyzji wyżej
    formula = form_logistic_final, method = "glm", 
    tuneGrid = NULL, preProc = NULL, family = "binomial"
  )
)

# Krok 3.2: Ustawienia walidacji .632
ctrl_boot632 <- trainControl(method = "boot632", number = 300, savePredictions = "final")

# Krok 3.3: Pętla oceniająca
estimation_results <- list()

for (model_name in names(models_to_test)) {
  cat(paste("\nPrzetwarzam:", model_name, "\n"))
  model_params <- models_to_test[[model_name]]
  
  set.seed(123)
  trained_model <- train(
    model_params$formula, data = train_set, 
    method = model_params$method, family = model_params$family,
    trControl = ctrl_boot632, tuneGrid = model_params$tuneGrid, 
    preProc = model_params$preProc
  )
  
  # Błąd .632
  err_632 <- 1 - trained_model$results$Accuracy
  
  # Rzeczywisty Błąd Testowy (dopiero teraz dotykamy test_set!)
  pred_test <- predict(trained_model, newdata = test_set)
  err_test <- 1 - mean(pred_test == test_set$Class)
  
  comparison <- "idealnie"
  if (err_632 > err_test) comparison <- "PRZESZACOWAŁ"
  if (err_632 < err_test) comparison <- "NIEDOSZACOWAŁ"
  
  estimation_results[[model_name]] <- list(
    Blad_Estymowany_632 = err_632,
    Rzeczywisty_Blad_Testowy = err_test,
    Wniosek = comparison
  )
}

cat("\n=== PODSUMOWANIE ZADANIA 3 ===\n")
print(do.call(rbind, lapply(estimation_results, as.data.frame.list)))


# === 4. ZADANIE 4: Dane Syntetyczne (PLS vs PCR) ===
cat("\n--- Zadanie 4: Dane Syntetyczne ---\n")
set.seed(123)
# Szum (duża wariancja, brak korelacji z Y)
noise = rnorm(100, 0, 3)
X1 = noise + rnorm(100, 0.1, 0.1)
# Sygnał (mała wariancja, duża korelacja z Y)
signal = rnorm(100, 0, 1)
X2 = signal + rnorm(100, 0, 0.01)
Y = 2 * signal + rnorm(100, 0, 0.001)

df_syn = data.frame(Y, X1, X2)

# PCR (wybierze X1 - szum)
pcr_res <- pcr(Y ~ ., data = df_syn, ncomp = 1, validation = "LOO")
# PLS (wybierze X2 - sygnał)
pls_res <- plsr(Y ~ ., data = df_syn, ncomp = 1, validation = "LOO")

cat("PCR Loadings (Comp 1):\n"); print(pcr_res$loadings)
cat("PLS Loadings (Comp 1):\n"); print(pls_res$loadings)