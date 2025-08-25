import importlib

def import_from_string(path: str):
    # path examples: 'module.sub:func' or 'module.sub.ClassName'
    if ':' in path:
        module_name, _, attr = path.partition(':')
        mod = importlib.import_module(module_name)
        return getattr(mod, attr)
    elif '.' in path:
        parts = path.split('.')
        module_name = '.'.join(parts[:-1])
        attr = parts[-1]
        mod = importlib.import_module(module_name)
        return getattr(mod, attr)
    else:
        raise ValueError('Invalid import path: ' + path)
