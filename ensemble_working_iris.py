import logging
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline, Pipeline

class StackingFramework:
    def __init__(self, base_classifiers, meta_model_name, preprocessing_steps=None):
        self.base_classifiers = base_classifiers
        self.meta_model_name = meta_model_name
        self.preprocessing_steps = preprocessing_steps
        self.trained_base_models = {}
        self.ensemble_model = None

    def train_base_models(self, X_train, y_train, cv, scoring):
        trained_base_models = {}
        for name, (model, hyperparameters) in self.base_classifiers.items():
            pipeline_steps = [('scaler', StandardScaler()), ('model', clone(model))]

            # Create pipeline
            pipeline = Pipeline(pipeline_steps)

            # Combine hyperparameters for the scaler and model
            all_hyperparameters = {f'model__{param}': value for param, value in hyperparameters.items()}

            try:
                # Perform hyperparameter tuning using RandomizedSearchCV
                random_search = RandomizedSearchCV(pipeline, all_hyperparameters, n_iter=10, cv=cv, scoring=scoring, random_state=42)
                random_search.fit(X_train, y_train)

                # Get the best model from the random search
                best_model = random_search.best_estimator_

                # Train the best model on the entire training set
                best_model.fit(X_train, y_train)

                trained_base_models[name] = best_model

                # Check if the model supports feature importance analysis
                self.try_feature_importance_analysis(name, best_model)

            except Exception as e:
                logging.error(f"Error training base model {name}: {str(e)}")

        return trained_base_models

    def try_feature_importance_analysis(self, name, model):
        try:
            # Access feature importances
            importances = model.named_steps['model'].feature_importances_
            self.log_feature_importance(name, importances)
        except (AttributeError, KeyError):
            # Model doesn't support feature importances
            self.log_feature_importance_error(name, None)

    def log_feature_importance(self, name, importances):
        print(f"Feature importances for {name}: {importances}")

    def log_feature_importance_error(self, name, error):
        logging.warning(f'{name} does not support feature importance analysis. Error: {error}')



    def log_feature_importance(self, model_name, importances):
        if importances is not None:
            logging.info(f'Feature importances for {model_name}: {importances}')
        else:
            logging.warning(f'{model_name} does not support feature importance analysis.')

    def log_feature_importance_error(self, model_name, error):
        logging.warning(f'Error analyzing feature importance for {model_name}: {str(error)}')

    def train(self, X_train, y_train, meta_hyperparameters=None, cv=None, scoring='accuracy'):
        # Apply preprocessing steps if provided
        if self.preprocessing_steps:
            base_classifiers = {
                name: (clone(model).set_params(**hyperparameters), hyperparameters) for name, (model, hyperparameters) in self.base_classifiers.items()
            }
        else:
            base_classifiers = {name: (clone(model).set_params(**hyperparameters), hyperparameters) for name, (model, hyperparameters) in self.base_classifiers.items()}

        # Train the base models
        self.trained_base_models = self.train_base_models(X_train, y_train, cv, scoring)

        # Create the stacking ensemble model with the specified meta-model
        meta_model_class = self.get_meta_model_class()
        self.ensemble_model = StackingClassifier(
            estimators=[(name, model) for name, model in self.trained_base_models.items()],
            final_estimator=meta_model_class(**meta_hyperparameters) if meta_hyperparameters else meta_model_class(),
            cv=cv
        )

        # Train the stacking ensemble model
        self.ensemble_model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        # Make predictions using the stacking ensemble model
        ensemble_predictions = self.ensemble_model.predict(X_test)

        # Evaluate the performance of the ensemble model
        accuracy = accuracy_score(y_test, ensemble_predictions)
        print(f'Stacking Ensemble Model Accuracy: {accuracy}')

        # Confusion Matrix and Classification Report
        cm = confusion_matrix(y_test, ensemble_predictions)
        print(f'Confusion Matrix:\n{cm}')

        # Classification Report
        cr = classification_report(y_test, ensemble_predictions)
        print(f'Classification Report:\n{cr}')

        # Feature Importance Analysis
        self.feature_importance_analysis(X_test, y_test)

    def feature_importance_analysis(self, X_test, y_test):
        # Perform feature importance analysis for each base model
        for name, model in self.trained_base_models.items():
            try:
                importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
                self.log_feature_importance(name, importances)
            except Exception as e:
                self.log_feature_importance_error(name, e)

    def get_meta_model_class(self):
        if 'sklearn' in self.meta_model_name.lower():
            return getattr(__import__('sklearn.linear_model', fromlist=[self.meta_model_name]), self.meta_model_name)
        elif 'xgboost' in self.meta_model_name.lower():
            return getattr(__import__('xgboost', fromlist=[self.meta_model_name]), self.meta_model_name)
        elif 'lightgbm' in self.meta_model_name.lower():
            return getattr(__import__('lightgbm', fromlist=[self.meta_model_name]), self.meta_model_name)
        elif 'logisticregression' in self.meta_model_name.lower():
            return LogisticRegression
        else:
            raise ValueError(f"Unsupported meta-model: {self.meta_model_name}")

# Example usage with preprocessing and hyperparameter tuning
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base classifiers and their optional hyperparameters
base_classifiers = {
    'RandomForest': (RandomForestClassifier(), {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}),
    'SVM': (SVC(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    # ... add other classifiers as needed
}

# Specify the meta-model name (e.g., 'LogisticRegression', 'XGBClassifier', 'LGBMClassifier')
meta_model_name = 'LogisticRegression'

# Define meta-model hyperparameters
meta_hyperparameters = {'C': 1.0, 'penalty': 'l2'}  # Adjust the 'C' value as needed

# Define preprocessing steps
preprocessing_steps = [('scaler', StandardScaler())]

# Set up logging
logging.basicConfig(level=logging.INFO)

# Create and train the stacking framework
stacking_framework = StackingFramework(base_classifiers, meta_model_name, preprocessing_steps)

stacking_framework.train(X_train, y_train, meta_hyperparameters, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))

# Evaluate the stacking framework on the test set
stacking_framework.evaluate(X_test, y_test)
