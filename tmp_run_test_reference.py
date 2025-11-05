import mlflow
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import mlflow.sklearn
from src.preprocessor import create_preprocessor

def run_tuning_and_training(self):
    # Set MLflow experiment explicitly (parent folder must exist in Databricks)
    experiment_name = self.config.tracking.experiment_name  # e.g., /Users/... or /Shared/...
    mlflow.set_experiment(experiment_name)
    mlflow.autolog()  # optional: keeps sklearn/xgboost autologging

    results = {}
    best_accuracy = 0.0
    best_run_name = None

    for model_name in self.config.tuning.models_to_run:
        ModelClass = self.get_model_class(model_name)
        hyperparam_grid = self.config.tuning.hyperparameters.get(model_name, [{}])

        for i, hyperparams in enumerate(hyperparam_grid):
            run_name = f"{model_name}_run_{i}"

            with mlflow.start_run(run_name=run_name):
                # Instantiate the model
                classifier = ModelClass(random_state=self.config.random_seed, **hyperparams)

                # Preprocessor + pipeline
                numerical_features = self.config.data.features.numerical
                categorical_features = self.config.data.features.categorical
                preprocessor = create_preprocessor(numerical_features, categorical_features)
                full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

                # Training
                full_pipeline.fit(self.X_train, self.y_train)

                # Compute metrics
                y_pred_train = full_pipeline.predict(self.X_train)
                y_pred_val = full_pipeline.predict(self.X_val)
                train_acc = accuracy_score(self.y_train, y_pred_train)
                val_acc = accuracy_score(self.y_val, y_pred_val)

                # Log metrics
                mlflow.log_metric("train_accuracy", train_acc)
                mlflow.log_metric("val_accuracy", val_acc)

                # Log model
                mlflow.sklearn.log_model(full_pipeline, model_name)

                # Track best model
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    best_run_name = run_name

            results[run_name] = {
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "model": full_pipeline
            }