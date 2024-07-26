###########################################
#
# Title: Logistic Regression 
# Name: Derrick Baruga
#
###########################################

# Set directory
setwd("/Users/derrickmarkbavaudbaruga/Documents/spring 2023/csc 461 machine learning/topic 8")

# Load csv
customer_churn <- read.csv("Telco-Customer-Churn.csv")

# Load Libraries
library(tidyverse)
library(tidymodels)

# Remove rows with NAs
customer_churn <- na.omit(customer_churn)

# Coercing each column to the appropriate type
customer_churn <- customer_churn %>%
  mutate(
    
    # Coerce to character since customerID is a unique identifier
    customerID = as.character(customerID),
    
    # Coerce to factor since these are categorical variables
    gender = as.factor(gender),
    SeniorCitizen = as.logical(SeniorCitizen),
    Partner = as.factor(Partner),
    Dependents = as.factor(Dependents),
    PhoneService = as.factor(PhoneService),
    MultipleLines = as.factor(MultipleLines),
    InternetService = as.factor(InternetService),
    OnlineSecurity = as.factor(OnlineSecurity),
    OnlineBackup = as.factor(OnlineBackup),
    DeviceProtection = as.factor(DeviceProtection),
    TechSupport = as.factor(TechSupport),
    StreamingTV = as.factor(StreamingTV),
    StreamingMovies = as.factor(StreamingMovies),
    Contract = as.factor(Contract),
    PaperlessBilling = as.factor(PaperlessBilling),
    PaymentMethod = as.factor(PaymentMethod),
    Churn = as.factor(Churn),
    
    # Coerce to numeric
    tenure = as.numeric(tenure),
    MonthlyCharges = as.numeric(MonthlyCharges),
    
    # TotalCharges might need to have commas removed and then be converted to numeric
    TotalCharges = as.numeric(gsub(",", "", TotalCharges))
  )

# Set seed for reproducibility
set.seed(123)

# Splitting the data
data_split <- initial_split(customer_churn, prop = 0.75)
train_data <- training(data_split)
test_data <- testing(data_split)

# Define the model
logistic_model <- logistic_reg() %>% 
  set_engine("glm") %>% 
  set_mode("classification")

# Define the formula
formula <- Churn ~ gender + SeniorCitizen + Partner + Dependents + tenure + 
  PhoneService + MultipleLines + InternetService + OnlineSecurity + 
  OnlineBackup + DeviceProtection + TechSupport + StreamingTV + 
  StreamingMovies + Contract + PaperlessBilling + PaymentMethod + 
  MonthlyCharges + TotalCharges

# Fit the model
model_fit <- logistic_model %>% 
  fit(formula, data = train_data)

# Get the summary of the fitted model
model_summary <- summary(model_fit$fit)

# Print the summary
print(model_summary)

# updated formula with significant predictor variables
updated_formula <- Churn ~ tenure + MultipleLines + InternetService + 
  StreamingTV + StreamingMovies + Contract + PaperlessBilling + MonthlyCharges + TotalCharges

# Fit the model
model_fit <- logistic_model %>% 
  fit(updated_formula, data = train_data)

# Get the summary of the fitted model
model_summary <- summary(model_fit$fit)

# Print the summary
print(model_summary)

# Make predictions
predictions <- predict(model_fit, test_data, type = "prob")

# Adding a column for predicted class based on the probability threshold
results <- test_data %>%
  select(customerID, Churn) %>%
  bind_cols(predictions) %>%
  mutate(predicted_class = ifelse(.pred_Yes > 0.5, "Yes", "No")) %>%
  mutate(predicted_class = factor(predicted_class, levels = levels(Churn)))

# Confusion Matrix
conf_matrix <- conf_mat(results, truth = Churn, estimate = predicted_class)

# Extract elements from the confusion matrix
true_positives <- conf_matrix$table["Yes", "Yes"]
false_positives <- conf_matrix$table["No", "Yes"]
true_negatives <- conf_matrix$table["No", "No"]
false_negatives <- conf_matrix$table["Yes", "No"]

# Calculate accuracy, sensitivity, and specificity
accuracy <- (true_positives + true_negatives) / sum(conf_matrix$table)
sensitivity <- true_positives / (true_positives + false_negatives)
specificity <- true_negatives / (true_negatives + false_positives)

# Print the metrics
cat("Accuracy:", accuracy, "\nSensitivity (Recall):", sensitivity, "\nSpecificity:", specificity, "\n")

# Load necessary library
library(yardstick)

# Generate the ROC Curve Data
roc_data <- roc_curve(results, truth = Churn, .pred_Yes)

# Plot the ROC Curve
roc_plot <- ggplot(roc_data, aes(x = 1 - specificity, y = sensitivity)) +
  geom_line() +
  geom_abline(linetype = "dashed") +
  labs(x = "1 - Specificity", y = "Sensitivity", title = "ROC Curve")

# Print the ROC Curve plot
print(roc_plot)

# Calculate AUC using roc_auc on the original results
auc_value <- roc_auc(results, truth = Churn, .pred_Yes)

# Print the AUC
cat("ROC AUC:", auc_value$.estimate, "\n")