from pyspark.ml.regression import LinearRegression
from config.constants import FEATURES, TARGET
from pyspark.sql.functions import col

def train_model(train_df_transformed):
    """
    get fitted model
    """
    # fit model
    model = LinearRegression(featuresCol=FEATURES, labelCol=TARGET)
    fitted_model = model.fit(train_df_transformed)

    # get residuals
    resid = fitted_model.transform(train_df_transformed).select("Fare", "prediction")
    resid = resid.withColumn("residuals", col("Fare") - col("prediction"))


    return fitted_model, resid



def get_predictions(model, test_df_transformed):
    """
    get predictions
    """
    preds = model.transform(test_df_transformed).select("Fare", "prediction")
    preds = preds.withColumn("prediction_error", col("prediction") - col("Fare"))
    return preds