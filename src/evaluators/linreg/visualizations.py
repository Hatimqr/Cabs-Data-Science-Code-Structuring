import matplotlib.pyplot as plt


def plot_residuals(residuals):
    # plot predicted vs fare
    residuals = residuals.toPandas()
    plt.scatter(residuals["prediction"], residuals["Fare"])
    plt.xlabel("Predicted Fare")
    plt.ylabel("Actual Fare")
    plt.title("Predicted vs Actual Fare")
    plt.show()


    # plot residuals histogram
    plt.hist(residuals["residuals"], bins=20)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Residuals Distribution")
    plt.show()

    mean_residual = residuals["residuals"].mean()
    print(f"Mean residual: {mean_residual}")

    std_residual = residuals["residuals"].std()
    print(f"Std residual: {std_residual}")



def plot_predictions(predictions):
    # plot preds vs true
    predictions = predictions.toPandas()
    plt.scatter(predictions["prediction"], predictions["Fare"])
    plt.xlabel("Predicted Fare")
    plt.ylabel("Actual Fare")
    plt.title("Predicted vs Actual Fare")
    plt.show()
    
    
    # plot error term histogram
    plt.hist(predictions["prediction_error"], bins=20)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Prediction Error Distribution")
    plt.show()
    
    
    


def plot_feature_importance(coefs_df, top_n=10):
    # plot feature importance
    plt.barh(coefs_df["feature"][:top_n], coefs_df["coef"][:top_n])
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importance")
    plt.show()