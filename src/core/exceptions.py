class MLOpsBaseException(Exception):
    pass


class DataSchemaError(MLOpsBaseException):
    pass


class DataLeakageError(MLOpsBaseException):
    pass


class PipelineConfigurationError(MLOpsBaseException):
    pass


class ModelConvergenceError(MLOpsBaseException):
    pass
