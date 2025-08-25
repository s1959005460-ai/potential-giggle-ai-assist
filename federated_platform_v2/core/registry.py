# Central registry for trainers, metrics, evaluators, etc.
_TRAINERS = {}
_METRICS = {}
_EVALUATORS = {}

def register_trainer(name):
    def decorator(cls):
        _TRAINERS[name] = cls
        return cls
    return decorator

def get_trainer(name, **kwargs):
    cls = _TRAINERS.get(name)
    if cls is None:
        raise KeyError(f"Trainer '{name}' not found. Available: {list(_TRAINERS.keys())}")
    inst = cls(**kwargs)
    # runtime interface check: must implement train_one_epoch
    if not hasattr(inst, 'train_one_epoch'):
        raise TypeError(f"Trainer {name} does not implement train_one_epoch()")
    return inst

def list_trainers():
    return list(_TRAINERS.keys())

def register_metric(name):
    def decorator(fn):
        _METRICS[name] = fn
        return fn
    return decorator

def get_metric(name):
    return _METRICS[name]

def list_metrics():
    return list(_METRICS.keys())

def register_evaluator(name):
    def decorator(fn):
        _EVALUATORS[name] = fn
        return fn
    return decorator

def get_evaluator(name):
    return _EVALUATORS[name]

def list_evaluators():
    return list(_EVALUATORS.keys())
