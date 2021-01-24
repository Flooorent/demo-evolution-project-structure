import hyperopt as hp
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def evaluate_hyperparams_wrapper(X_train, X_test, y_train, y_test):
    def evaluate_hyperparams(params):
        min_samples_leaf = int(params['min_samples_leaf'])
        max_depth = params['max_depth']
        n_estimators = int(params['n_estimators'])

        rf = RandomForestRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_estimators=n_estimators,
        )
        rf.fit(X_train, y_train)

        mlflow.sklearn.log_model(rf, "model")

        predictions = rf.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        return {'loss': mse, 'status': STATUS_OK}
  
    return evaluate_hyperparams


def train(df, experiment_name, run_name):
    mlflow.set_experiment(experiment_name)

    data = df.toPandas()
    X_train, X_test, y_train, y_test = train_test_split(data.drop(["quality"], axis=1), data[["quality"]].values.ravel(), random_state=42)

    search_space = {
        'n_estimators': hp.uniform('n_estimators', 10, 100),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 1, 20),
        'max_depth': hp.uniform('max_depth', 2, 10),
    }

    spark_trials = SparkTrials(parallelism=4)

    with mlflow.start_run(run_name=run_name):
      fmin(
          fn=evaluate_hyperparams_wrapper(X_train, X_test, y_train, y_test),
          space=search_space,
          algo=tpe.suggest,
          max_evals=10,
          trials=spark_trials,
      )
