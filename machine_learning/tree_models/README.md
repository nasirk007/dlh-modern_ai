# Tree-Based Models Module
This module provides a professional introduction to tree-based classification methods. It covers decision trees, random forests, and boosting techniques used to model complex decision boundaries and ranked feature importance.

## Why This ML Approach Is Used
Tree-based methods are powerful for structured data because they model non-linear relationships, handle mixed feature types, and offer interpretable decision rules. They are widely used when explainability and robust performance are both important.

### How It Differs from Other ML Types
- Tree-based models use recursive partitioning rather than linear equations.
- They naturally handle categorical and numeric features without extensive transformation.
- Unlike neural networks, they are easier to interpret and less sensitive to feature scaling.

## Algorithms in This Module

### Decision Tree Classifier
A decision tree splits data using Gini impurity and grows until stopping criteria are met.
- Function: `build_decision_tree(min_samples_leaf, min_samples_split, random_state)`
- Output: `DecisionTreeClassifier` model

### Train a Tree-Based Classifier
This task fits a tree-based classifier on training data.
- Function: `train_tree(clf, X, y)`
- Output: None (model is trained in place)

### View Decision Rules
This task prints the structure of a trained decision tree.
- Function: `draw(clf, feature_names, class_names)`
- Output: None (prints human-readable tree rules)

### Generate Predictions
This task uses a trained classifier to generate predicted labels.
- Function: `generate_predictions(clf, X)`
- Output: NumPy array of predicted class labels

### Evaluate Classifier Performance
This task produces a classification report including precision, recall, and F1-score.
- Function: `evaluate(true_labels, predicted_labels, class_names)`
- Output: string classification report

### Pre-Pruning
This task performs grid search over decision tree hyperparameters to find the best pre-pruning settings.
- Function: `prepruning(X, y, clf)`
- Output: dictionary of best hyperparameters

### Cost-Complexity Pruning Path
This task retrieves the pruning path for a trained decision tree.
- Function: `get_pruning_path(clf, X, y)`
- Output: `ccp_alphas`, `impurities`

### Train and Evaluate Pruned Trees
This task trains multiple pruned trees and tracks performance for each alpha.
- Function: `prune_and_evaluate_trees(X_train, y_train, X_test, y_test, ccp_alphas, random_state, min_samples_leaf, min_samples_split)`
- Output: `clfs`, `train_scores`, `test_scores`

### Best ccp_alpha Selection
This task selects the best pruning parameter balancing test accuracy and generalization.
- Function: `get_best_alpha(clfs, train_scores, test_scores, ccp_alphas)`
- Output: `best_alpha`, `best_clf`

### Random Forest Classifier
This task initializes a random forest model for ensemble classification.
- Function: `random_forest(n_estimators, random_state)`
- Output: `RandomForestClassifier` model

### Feature Importance with Random Forest
This task extracts importance scores from a trained forest.
- Function: `feature_importance(rf)`
- Output: `importances`, `indices`

### Boosting Classifier Comparison
This task builds a selected boosting classifier from AdaBoost, Gradient Boosting, XGBoost, or LightGBM.
- Function: `compare_boosting_classifiers(name, n_estimators, random_state)`
- Output: untrained boosting classifier instance

## Finance and Audit Applications
This module can support finance, audit, risk management, business valuation, and investment workflows by:
- Classifying credit risk tiers or transaction categories
- Detecting anomalous accounting or fraud signals
- Ranking features for valuation and risk drivers
- Supporting investment screening and portfolio decisions
- Comparing model performance for regulatory and audit use cases

## Project Workflow
1. Define and initialize tree-based models
2. Train classifiers on labeled data
3. Visualize tree structure and decision rules
4. Generate predictions for new samples
5. Evaluate performance with classification metrics
6. Tune pruning and ensemble hyperparameters
7. Compare models and interpret feature importance

## Task Overview

- `0-decision_tree.py`: build decision tree models with split constraints
- `1-train.py`: train a tree-based classifier on input data
- `2-draw.py`: print the textual decision tree structure
- `3-predict.py`: generate predictions from a trained model
- `4-evaluate.py`: generate a classification report
- `5-prepruning.py`: search for pre-pruning hyperparameters
- `6-pruning_path.py`: retrieve cost-complexity pruning path
- `7-prune_evaluate.py`: train and compare pruned decision trees
- `8-best_alpha.py`: select the best pruning alpha value
- `9-random_forest.py`: create a random forest classifier
- `10-feature_importance.py`: compute feature importance from a random forest
- `11-boosting.py`: initialize boosting models by name

## Skills Developed
- Building and tuning decision tree classifiers
- Training and evaluating tree-based models
- Understanding pruning and model complexity control
- Comparing bagging and boosting ensemble methods
- Extracting feature importance for interpretability
- Applying tree-based methods to finance and audit problems

