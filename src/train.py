import sys
sys.path.append("../")

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from urllib.parse import urlparse

# load the training and testing data from other module
from data_preparation import DataPrepration

data_ = DataPrepration()

class ModelTraining:

    def __init__(self) -> None:

        self.param_grid = {
                        'n_estimators': [100],
                        # 'max_depth': [5],
                        # 'min_samples_split': [2, 5],
                        # 'min_samples_leaf': [1, 2]
                    }
        
        self.X_train, self.X_test, self.y_train, self.y_test = data_.prepare_data()
        
        self.rf = RandomForestRegressor()

    def hyperparameter_tuning(self):
        
        grid_search = GridSearchCV(
                        estimator=self.rf,
                        param_grid=self.param_grid,
                        cv=3,
                        n_jobs=-1,
                        verbose=2,
                        scoring="neg_mean_squared_error"
                    )
        
        grid_search.fit(self.X_train, self.y_train)
        return grid_search
    
    def run_mlflow(self):

        mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
        # mlflow.set_tracking_uri("http://host.docker.internal:5000")
        mlflow.set_experiment("First ML Run")

        signature = infer_signature(self.X_train, self.y_train)

        try:
            with mlflow.start_run():

                ## Perform hyperparameter tuning
                grid_search = self.hyperparameter_tuning()

                ## Get the best model
                best_model = grid_search.best_estimator_

                ## Evaluate the best model
                y_pred = best_model.predict(self.X_test)
                mse = mean_squared_error(self.y_test, y_pred)

                joblib.dump(best_model, "model.pkl")

                ## Log best parameters and metrics
                mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
                # mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
                # mlflow.log_param("best_min_samples_split", grid_search.best_params_['min_samples_split'])
                # mlflow.log_param("best_min_samples_leaf", grid_search.best_params_['min_samples_leaf'])
                mlflow.log_metric("mse", mse)

                ## Tracking url
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                print(urlparse(mlflow.get_tracking_uri()))

                print(f"tracking_url_type_store: {tracking_url_type_store}")

                # if tracking_url_type_store !='file':
                #     mlflow.sklearn.log_model(best_model, "model", registered_model_name="Best Randomforest Model")
                # else:
                #     mlflow.sklearn.log_model(best_model, "model", signature=signature)

                mlflow.sklearn.log_model(best_model, "model", signature=signature)

                print(f"Best Hyperparameters: {grid_search.best_params_}")
                print(f"Mean Squared Error: {mse}")
                mlflow.end_run(status="FINISHED")
        except Exception as e:
                print("Error during MLflow run:", str(e))
                mlflow.end_run(status="FAILED")  # Log failure explicitly
                raise e

if __name__=="__main__":
    obj = ModelTraining()
    obj.run_mlflow()