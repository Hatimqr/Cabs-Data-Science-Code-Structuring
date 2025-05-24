from src.data.loader import create_spark_session, load_cab_data
from src.features.linreg.preprocessor import (remove_unnecessary_columns,
                                              get_target_column,
                                              add_day_of_week,
                                              transform_binary_columns,
                                              test_train_split)
from src.features.linreg.features import build_pipeline

from src.models.linear_regression import train_model, get_predictions

from src.evaluators.linreg.visualizations import plot_residuals, plot_predictions, plot_feature_importance

from src.evaluators.linreg.metrics import evaluation_suite, feature_importance

from config.constants import FEATURES, TARGET, OHE_COLUMNS, OTHER_COLUMNS


class LinearRegressionPipe:
    def __init__(self, scale=False):
        self.features = FEATURES
        self.target = TARGET

        self.scale = scale


        self.spark = create_spark_session()
        self.df = load_cab_data(self.spark)


        # split data
        self.train_df = None
        self.test_df = None



        # pipeline 
        self.fitted_feature_pipeline = None


        # transform splits
        self.train_df_transformed = None
        self.test_df_transformed = None


        # model
        self.model = None
        self.residuals = None

        # get predictions
        self.predictions = None


    def preprocess_data(self):
        # fix target column
        print("Fixing target column\n")
        self.df = get_target_column(self.df)

        # add day of week
        print("Adding day of week\n")
        self.df = add_day_of_week(self.df)

        # transform binary columns
        print("Transforming binary columns\n")
        # transform binary columns
        self.df = transform_binary_columns(self.df)

    def split_data(self):
        """
        split data into train and test
        """
        # split data into train and test
        print("Splitting data into train and test")
        self.train_df, self.test_df = test_train_split(self.df)
        print("Train rows =", self.train_df.count())
        print("Test rows =", self.test_df.count(), '\n')
    
    def clean_data(self):
        """
        clean data
        """
        # remove unnecessary columns
        print("Removing unnecessary columns")
        self.train_df = remove_unnecessary_columns(self.train_df)
        self.test_df = remove_unnecessary_columns(self.test_df)
        print('Using columns: ', self.train_df.columns, '\n')


    def fit_feature_pipeline(self):
        """
        fit feature pipeline
        """
        # fit pipeline
        print("Fitting feature pipeline")
        pipeline = build_pipeline(scale=False)
        self.fitted_feature_pipeline = pipeline.fit(self.train_df)

        # transform train data
        self.train_df_transformed = self.fitted_feature_pipeline.transform(self.train_df)

        # transform test data
        self.test_df_transformed = self.fitted_feature_pipeline.transform(self.test_df)
        print("OHE columns: ", OHE_COLUMNS)
        print("Other columns: ", OTHER_COLUMNS)
        print()



    def train_model(self):
        print("Fitting model and saving residuals")
        self.model, self.residuals = train_model(self.train_df_transformed)



    def plot_residuals(self):
        print("Plotting residuals")
        plot_residuals(self.residuals)



    def get_predictions(self):
        print("Getting predictions")
        self.predictions = get_predictions(self.model, self.test_df_transformed)


    def evaluate_predictions(self):
        print("Evaluating predictions")
        # first plot true vs predicted
        plot_predictions(self.predictions)

        # get error metrics
        metrics = evaluation_suite(self.predictions)

        # print metrics
        for metric, value in metrics.items():
            print(f"{metric}: {value}")



    def feature_importance(self, k=10):
        print("Pulling feature importance")
        coefs_df = feature_importance(self.train_df_transformed, "features", self.model)

        # plot feature importance
        plot_feature_importance(coefs_df, k)

