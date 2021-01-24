from pyspark.sql import functions as F


def engineer_features(df):
    return df.withColumn("total_acidity", F.col("fixed acidity") + F.col("volatile acidity"))
