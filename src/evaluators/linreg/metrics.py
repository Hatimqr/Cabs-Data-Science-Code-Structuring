from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd



def evaluation_suite(predictions):
    # Create a RegressionEvaluator
    evaluator_rmse = RegressionEvaluator(labelCol="Fare", predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="Fare", predictionCol="prediction", metricName="r2")
    evaluator_mae = RegressionEvaluator(labelCol="Fare", predictionCol="prediction", metricName="mae")

    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)

    return {
        "rmse": rmse,
        "r2": r2,
        "mae": mae
        }



def feature_importance(df_transformed, feature_col, model):
        schema = df_transformed.schema

        attrs_metadata = schema[feature_col].metadata.get("ml_attr", {})
        
        # Check if 'attrs' or 'attributes' key exists (Spark versions might vary slightly)
        attrs_list_container = attrs_metadata.get("attrs", attrs_metadata.get("attributes"))

        feature_names = []
        # attrs_list_container can have 'numeric' and 'binary' (or 'nominal') keys
        all_attrs_list = []
        if "numeric" in attrs_list_container:
            all_attrs_list.extend(attrs_list_container["numeric"])
        if "binary" in attrs_list_container: # For OHE features typically
            all_attrs_list.extend(attrs_list_container["binary"])
        if "nominal" in attrs_list_container: # Sometimes used instead of/with binary
             all_attrs_list.extend(attrs_list_container["nominal"])
        
        if not all_attrs_list:
            print(f"Warning: No 'numeric', 'binary', or 'nominal' attributes found in metadata for '{features_col_name}'.")
            return None


        for attr in all_attrs_list:
            if "idx" not in attr:
                print(f"Warning: Attribute {attr.get('name', 'Unknown')} is missing 'idx'. Cannot guarantee order.")
        
        # Let's try sorting if 'idx' is present, otherwise use the order as received.
        if all(isinstance(attr, dict) and "idx" in attr for attr in all_attrs_list):
            all_attrs_sorted = sorted(all_attrs_list, key=lambda x: x["idx"])
        else:
            print("Warning: Not all attributes have an 'idx' field. Using order as received from metadata.")
            all_attrs_sorted = all_attrs_list # Rely on Spark's internal order

        feature_names = [attr.get("name", f"unknown_feature_{i}") for i, attr in enumerate(all_attrs_sorted)]
        
        coefs = model.coefficients
        coefs_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
        coefs_df = coefs_df.sort_values(by="coef", ascending=False)
        return coefs_df.sort_values(by="coef", ascending=False)