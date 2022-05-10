import matplotlib as mpl
import re
import numpy as np

class BackupAndRestoreMatplotlibRC(object):
    def __enter__(self):
        self.matplotlibrc = mpl.rcParams.copy()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        mpl.rcParams = self.matplotlibrc

def backup_mpl_rccontext():
    def decorator(function):
        def wrapper(*args, **kwargs):
            with BackupAndRestoreMatplotlibRC():
                result = function(*args, **kwargs)
            return result
        return wrapper
    return decorator

def run_after(f2):
    def decorator(function):
        def wrapper(*args, **kwargs):
            f2(*args, **kwargs)
            result = function(*args, **kwargs)
            return result
        return wrapper
    return decorator

def run_before(f2,store_result_in="plt"):
    def decorator(function):
        def wrapper(*args, **kwargs):
            result = function(*args, **kwargs)
            f2(*args, **kwargs)
            return result
        return wrapper
    return decorator

# Thanks to: https://stackoverflow.com/a/25875504/6018688
def tex_escape(text):
    """
        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX
    """
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(conv.keys(), key = lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], text)

def gini(arr):
    count = arr.size
    coefficient = 2 / count
    indexes = np.arange(1, count + 1)
    weighted_sum = (indexes * arr).sum()
    total = arr.sum()
    constant = (count + 1) / count
    return coefficient * weighted_sum / total - constant

def lorenz_gini(arr):
    arr = np.sort(arr)
    scaled_prefix_sum = arr.cumsum() / arr.sum()
    np.insert(scaled_prefix_sum, 0, 0)
    lorenz_y = scaled_prefix_sum 
    lorenz_x = np.linspace(0.0, 1.0, lorenz_y.size) 
    gini_index = gini(arr)
    return ((lorenz_x, lorenz_y), gini_index)

def lorenz_gini_multi(data, name_format="{} (GI={:0.2f})"):
    data_new = []
    for name, samples in data:
        (lorenz_x, lorenz_y), gini_index = lorenz_gini(samples)
        name_new = name_format.format(name, gini_index)
        data_new.append( (name_new, (lorenz_x,lorenz_y) )   )
    return data_new
  
