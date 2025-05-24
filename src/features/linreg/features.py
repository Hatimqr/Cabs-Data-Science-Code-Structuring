from pyspark.ml.feature import OneHotEncoder,VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from config.constants import FEATURES

from config.constants import OHE_COLUMNS, OTHER_COLUMNS


def build_pipeline(scale=False):
    """
    build pipeline
    """
    # one hot encode
    ohe_out = ["OHE_" + col for col in OHE_COLUMNS]
    ohe = OneHotEncoder(inputCols=OHE_COLUMNS, outputCols=ohe_out)

    # vector assemble
    vector_assembler = VectorAssembler(inputCols=ohe_out+OTHER_COLUMNS, outputCol=FEATURES)

    # standard scale
    if scale:
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
        pipeline = Pipeline(stages=[ohe, vector_assembler, scaler])
    else:
        pipeline = Pipeline(stages=[ohe, vector_assembler])
    
    return pipeline