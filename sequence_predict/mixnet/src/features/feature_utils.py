import numpy as np


def pad_vec(vec, length, padding_value=0):
    """Padding a list.

    Args:
        vec: list, list of values.
        length: int, output length.
        padding_value: int, padding value. Default: 0.

    Returns:
        res_vec: NumPy array.
    """
    if len(vec) < length:
        res_vec = np.pad(vec, (0, length - len(vec)), 'constant', constant_values=padding_value)
    else:
        res_vec = vec[-length:]

    return res_vec


def build_index_by_count(count_dict, min_count, offset=0):
    sorted_items = sorted(count_dict.items(), key=lambda vv: vv[1]['count'], reverse=True)
    sorted_dict = {}
    for i, (k, v) in enumerate(sorted_items):
        v = {'index': i + offset, **v}
        if v['count'] < min_count:
            break
        sorted_dict[k] = v
    return sorted_dict