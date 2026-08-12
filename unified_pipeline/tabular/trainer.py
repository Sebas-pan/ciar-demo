from sklearn.model_selection import GridSearchCV

from .models import create_estimator


def train_estimator(model_type, task, params, search, X_train, y_train, cv=3):
    estimator = create_estimator(model_type, task, params)
    if search:
        grid = GridSearchCV(estimator, search, cv=cv, n_jobs=-1)
        grid.fit(X_train, y_train)
        return grid.best_estimator_, dict(grid.best_params_)
    estimator.fit(X_train, y_train)
    return estimator, None