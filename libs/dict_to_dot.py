from types import SimpleNamespace

def nested_namespace_to_dict(ns):
    if type(ns) == SimpleNamespace:
        d = ns.__dict__
    else:
        return ns

    for k, v in d.items():
        if type(ns) == SimpleNamespace:
            d[k] = nested_namespace_to_dict(v)
        else:
            d[k] = v

    return d

def nested_dict_to_namespace(d):
    new_dict = {}
    for k, v in d.items():
    
        if type(v) == dict:
            new_dict[k] = nested_dict_to_namespace(v)
        else:
            new_dict[k] = v
    
    return SimpleNamespace(**new_dict)