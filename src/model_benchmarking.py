from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

def cross_validate_model(model, X, y, cv=5, scoring='accuracy', problem_type='classification', random_state=42):
    """
    Performs cross-validation on the given model and returns aggregated metrics.

    Args:
        model: Scikit-learn compatible model.
        X (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): Target.
        cv (int): Number of folds for cross-validation.
        scoring (str or list): Metric(s) to evaluate during cross-validation.
        problem_type (str): 'classification' or 'regression'.
        random_state (int): Seed for reproducibility in fold generation.

    Returns:
        dict: Aggregated metrics across all folds.
    """
    # Choose the appropriate cross-validation strategy
    if problem_type == 'classification':
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    elif problem_type == 'regression':
        cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        raise ValueError("Invalid problem_type. Use 'classification' or 'regression'.")

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)

    # Return results
    return {
        "mean_score": np.mean(scores),
        "std_dev_score": np.std(scores),
        "all_scores": scores.tolist()
    }
