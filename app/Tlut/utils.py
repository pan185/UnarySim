from math import pi
import yaml
import json

# logger
from time import strftime, gmtime
import random
import string
import logging
import sys
logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)  # capture everything
logger.disabled = True

class bcolors:
    """
    Reference from: 
    https://svn.blender.org/svnroot/bf-blender/trunk/blender/build_files/scons/tools/bcolors.py
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def setup_logging(module_name, logger):
    # logging setup
    def logfilename():
        """ Construct a unique log file name from: date + 16 char random. """
        timeline = strftime("%Y-%m-%d--%H-%M-%S", gmtime())
        randname = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(16))
        return module_name + "-" + timeline + "-" + randname + ".log"

    # log to file
    full_log_filename = logfilename()
    fileHandler = logging.FileHandler(full_log_filename)
    # formatting for log to file
    # TODO: filehandler should be handler 0 (firesim_topology_with_passes expects this
    # to get the filename) - handle this more appropriately later
    logFormatter = logging.Formatter("%(asctime)s [%(funcName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.NOTSET)  # log everything to file
    logger.addHandler(fileHandler)

    # log to stdout, without special formatting
    consoleHandler = logging.StreamHandler(stream=sys.stdout)
    consoleHandler.setLevel(logging.DEBUG)  # show only INFO and greater in console
    logger.addHandler(consoleHandler)

def parse_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.full_load(f)
    return data

def store_json(json_path, data, indent=None):
    with open(json_path, "w") as f:
        json.dump(data, f, indent=indent)

def parse_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data

def list_to_comma_separated_str_with_padding(_list, dim_hw):
    assert(len(_list) <= dim_hw)
    _str = ''
    # for item in _list: _str += f'{item},'
    for i in range(dim_hw): 
        if i < len(_list): _str += f'{_list[i]},'
        else: _str += ','
    _str += '\n'
    return _str

def im2col_addr(input_layout, patch_P, patch_Q, patch_N, pixel, pad, 
    R, S, C, N, Wdilation, Hdilation, Wstride, Hstride, W, H):
    # RSC along a col, pixel is the col index
    assert input_layout == 'NCHW' #TODO: implemet flex input layout
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
    addr = patch_N * (W*H*C) + (W*H)*c_index + (W)*h_index + (1)*w_index
    # print(addr)
    
    return addr
