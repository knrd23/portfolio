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
