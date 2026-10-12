# Customer churn with logistic regression

An R analysis of the Telco Customer Churn dataset. The workflow prepares customer attributes, creates a stratified 75/25 split, fits logistic regression, and evaluates churn predictions with a confusion matrix and ROC curve.

## Run locally

Install R and the required packages:

```r
install.packages(c("tidyverse", "tidymodels"))
```

```bash
git clone https://github.com/derrickbaruga7/R-Logistic-Regression.git
cd R-Logistic-Regression
Rscript "Logistic Regression.R"
```

The source CSV is **not included**. Place an authorized copy of `Telco-Customer-Churn.csv` in `data/`, or set `CHURN_DATA_PATH` to its location. The script expects the original Telco column names, including `customerID`, `Churn`, `tenure`, `Contract`, `MonthlyCharges`, and `TotalCharges`. See the preparation and formula sections for the complete input schema. `Churn` must contain `No` and `Yes`.

## Evaluation

`Yes` is the positive churn class. Predictions use a 0.5 probability threshold. Confusion-matrix rows represent predictions and columns represent truth; ROC calculations explicitly use the second factor level (`Yes`) to match `.pred_Yes`.

The previous README's accuracy, sensitivity, and AUC came from an earlier run with inconsistent metric orientation. Those values are not retained as verified results. Re-run the corrected script with the source data to obtain current metrics. The [yardstick documentation](https://yardstick.tidymodels.org/reference/roc_auc.html) explains the positive-class convention.

This is an exploratory model. Its reduced formula reflects the original analysis and still needs independent validation before operational use.
