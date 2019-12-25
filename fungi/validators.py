from cerberus import Validator

paths_schema = {'BASE_DIR': {'type': 'string'},
                'DATA_DIR': {'type': 'string'},
                'BOTTLENECK_DIR': {'type': 'string'},
                'GRAPH_DIR': {'type': 'string'}}

model_schema = {}

data_schema = {}

training_schema = {}


'''
--------------------------------------------------------------------------------
                        Creating Validators
--------------------------------------------------------------------------------
'''

SCHEMAS = {'paths': paths_schema,
           'model': model_schema,
           'data': data_schema,
           'training': training_schema}

VALIDATORS = {key: Validator(schema) for key, schema in SCHEMAS.items()}


def get_schema(module=None):
    if module is not None:
        return SCHEMAS[module]
    # return all
    return SCHEMAS


def get_validator(module=None):
    if module is not None:
        return VALIDATORS[module]
    # return all
    return VALIDATORS
