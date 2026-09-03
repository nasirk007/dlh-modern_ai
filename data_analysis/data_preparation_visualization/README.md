# Data Analysis and Preparation
Prepare, explore, visualize, and transform customer-churn data into a dataset ready for responsible machine-learning use.

## Why this module matters
Reliable analysis starts with data that is understood and fit for purpose. This module presents a practical preparation workflow using the Telco Customer Churn dataset: inspect structure and quality, correct data types, handle missing and duplicate records, explore distributions and relationships, test selected associations, engineer features, encode categories, scale numeric values, and create reproducible train/test data.

The workflow helps distinguish data-quality issues from meaningful patterns before modeling. It also makes preprocessing decisions visible and repeatable instead of hiding them inside a later modeling step.

## Business problems it can support
The methods in this module can support analysis and decision-making such as:
- Identifying customer segments with higher churn risk and informing retention outreach.
- Understanding how contract type, services, tenure, and charges relate to customer outcomes.
- Preparing transaction, customer, or operational data for risk-scoring and forecasting models.
- Detecting missing, duplicated, inconsistent, or incorrectly typed records before reporting.
- Comparing groups and prioritizing variables for further investigation or model development.
- Creating standardized, encoded datasets for repeatable analytics workflows.

The same techniques are useful in audit, finance, and advisory work for data-quality reviews, exception analysis, portfolio segmentation, customer or account analytics, and evidence-based recommendations.

## Limitations in real business settings
This module demonstrates exploratory analysis and preprocessing patterns, not a complete production data pipeline. In practice:
- Findings are limited by the quality, completeness, representativeness, and permitted use of the source data.
- The Telco churn dataset and its variables may not reflect another organization, customer population, time period, or business process.
- Correlation and statistical tests identify associations; they do not establish causation or guarantee predictive value.
- Decisions about dropping, imputing, encoding, binning, and scaling can change results and require domain justification.
- Fitting preprocessing steps on all data before splitting can cause leakage; production pipelines should fit transformations on training data only.
- Visualizations and p-values do not replace business context, validation, fairness review, privacy controls, or monitoring for data drift.
- The scripts do not provide automated data contracts, orchestration, lineage, access controls, deployment, or model governance.

Treat the outputs as analysis inputs and decision support. Validate assumptions with subject-matter experts and evaluate any downstream model on independent, representative data.

## Repository contents
| `0-describe_data.py` | Inspect shape, types, sample rows, missing values, and duplicates |
| `1-plot_missingness.py` | Visualize missing-data patterns |
| `2-convert_columns.py` | Convert columns to appropriate data types |
| `3-clean_total_charges.py` | Clean and convert `TotalCharges` |
| `4-remove_duplicates.py` | Remove duplicate records |
| `5-drop_customerID.py` | Drop the identifier column before analysis |
| `6-plot_churn_distribution.py` | Plot the target distribution |
| `7-plot_categorical_distributions.py` | Plot categorical feature distributions |
| `8-plot_continuous_distributions.py` | Plot continuous feature distributions |
| `9-plot_correlation_heatmap.py` | Visualize numeric correlations |
| `10-plot_categorical_vs_churn.py` | Compare categorical features with churn |
| `11-plot_numeric_vs_churn.py` | Compare numeric features with churn |
| `12-chi_square_tests.py` | Test associations between categorical variables |
| `13-ttest_numeric.py` | Apply Welch's t-test to numeric groups |
| `14-create_features.py` | Create service-count and tenure-group features |
| `15-encode_features.py` | Encode target and categorical features |
| `16-scale_numeric.py` | Standardize selected numeric features |
| `17-split_data.py` | Create reproducible training and test sets |

## Setup and workflow
### 1. Create a Python environment
From this directory, create and activate an isolated environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
On Windows, activate the environment with `.venv\\Scripts\\activate`.

### 2. Install the dependencies
The module uses the following tested package versions:
```bash
python -m pip install --upgrade pip
python -m pip install \
	numpy==2.0.2 \
	pandas==2.2.2 \
	scikit-learn==1.6.1 \
	matplotlib==3.10.0 \
	seaborn==0.13.2 \
	scipy==1.16.0 \
	pillow==11.3.0
```

Verify the main libraries:
```bash
python -c "import numpy, pandas, sklearn, matplotlib, seaborn, scipy; print('Environment ready')"
```

### 3. Run the preparation workflow
Run scripts from this directory so the bundled CSV files resolve correctly:
```bash
python 0-describe_data.py
python 1-plot_missingness.py
```

Then continue through the numbered scripts, using the cleaned output from one stage as the input to the next. The practice notebooks provide an interactive way to inspect selected visualization and feature-preparation steps.

Recommended sequence:
1. Inspect the raw data and document its shape, types, missing values, and duplicates.
2. Correct column types and clean `TotalCharges` before numerical analysis.
3. Remove duplicates and identifiers that should not be used as features.
4. Plot target, categorical, continuous, correlation, and feature-versus-churn distributions.
5. Use chi-square tests for categorical associations and Welch's t-tests for selected numeric group comparisons.
6. Create domain-relevant features such as service count and tenure groups.
7. Encode categorical variables and the target, preserving the transformation rules for later data.
8. Scale numeric variables using parameters learned from the training data in a production workflow.
9. Split features and target into training and test sets with a fixed random state. For imbalanced targets, use a stratified split and confirm class proportions.

## Learning objectives
After completing this module, you should be able to explain:
- Why data should be prepared before modeling and how visualization informs feature engineering.
- How to detect and handle missing values, incorrect types, duplicates, and unnecessary columns.
- How to visualize target and feature distributions, correlations, and feature relationships with churn.
- When to use chi-square tests and Welch's t-test, and why statistical association is not causation.
- How binning, feature creation, categorical encoding, feature scaling, and reproducible splitting work.

## References
- [Pandas documentation](https://pandas.pydata.org/docs/)
- [Matplotlib documentation](https://matplotlib.org/stable/)
- [Seaborn documentation](https://seaborn.pydata.org/)
- [SciPy statistical functions](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Scikit-learn preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Scikit-learn model selection](https://scikit-learn.org/stable/model_selection.html)
