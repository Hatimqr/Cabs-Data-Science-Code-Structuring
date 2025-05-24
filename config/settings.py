"""Global configuration settings for the cab analysis project."""

import os
# get the root directory of the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath('...')))

# Spark configuration
SPARK_CONFIG = {
    "spark.sql.repl.eagerEval.enabled": "true",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true"
}





# Data paths
DATA_PATHS = {
    "raw_data": os.path.join(ROOT_DIR, "data", "Colombo-Cab-data.csv")
}




# Statistical test parameters
STATISTICAL_CONFIG = {
    "significance_level": 0.05
}



# ML
ML_CONFIG = {
    "test_size": 0.3,
    "random_state": 42
}


