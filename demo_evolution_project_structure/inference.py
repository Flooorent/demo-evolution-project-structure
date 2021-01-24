import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F


def make_predictions(df, model_name, spark):
    client = MlflowClient()
    model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}/Production")
    model_version = client.get_latest_versions(model_name, stages=["Production"])[0].version
  
    return (
      df.withColumn("prediction", model_udf(*df.columns))
        .withColumn("model_version", F.lit(model_version))
    )
