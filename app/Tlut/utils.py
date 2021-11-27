from math import pi
import yaml
import json

def parse_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.full_load(f)
    return data

def store_json(json_path, data, indent=None):
    with open(json_path, "w") as f:
        json.dump(data, f, indent=indent)

def list_to_comma_separated_str_with_padding(_list, dim_hw):
    assert(len(_list) <= dim_hw)
    _str = ''
    # for item in _list: _str += f'{item},'
    for i in range(dim_hw): 
        if i < len(_list): _str += f'{_list[i]},'
        else: 
            _str += 'nan,'
    _str += '\n'
    return _str

def im2col_addr(input_layout, patch_P, patch_Q, patch_N, pixel, pad, 
    R, S, C, N, Wdilation, Hdilation, Wstride, Hstride):
    # RSC along a col, pixel is the col index
    assert input_layout == 'NWHC' #TODO: implemet flex input layout
    # pixel = 9 # set override val for debugging
    c_index = int(pixel/(R*S))
    s_index = int(pixel/R % R)
    r_index = pixel % R
    # print(f'({r_index}, {s_index}, {c_index})')

    w_from_P = (patch_P - 1 -1) * Wstride + R - 2 * pad
    h_from_Q = (patch_Q - 1-1) * Hstride + S - 2 * pad

    w_from_P_index = 1
    h_from_Q_index = 1

    w_index = w_from_P + r_index - w_from_P_index
    h_index = h_from_Q + s_index - h_from_Q_index

    # print(f'{w_index}, {h_index}, {c_index}')
    addr = (R*S)*c_index + (R)*h_index + (1)*w_index
    # print(addr)
    
    return addr
