from math import pi
from os import stat
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

class cor:
    grey1 = "#D3D3D3"
    grey2 = "#AAAAAA"
    grey3 = "#808080"
    grey4 = '#606060'

    mint1 = '#BBDBBD'
    mint2 = '#8BD8BD'
    mint3 = '#4BD4BD'
    mint4 = '#0BD0BD'

    blue1 = '#829cb8'
    blue2 = '#6d8cac'
    blue3 = '#587ba0'
    blue4 = '#436b94'

    tlut_mint = '#8BD8BD'
    tlut_navy = '#243665'

    # accuracy color plots
    tlut_color = "#6ACCBC"
    fp_color = "#FF7F7F"
    fxp_color = "#D783FF"
    hub_color = '#7A81FF'

    # patterns
    patterns = [ "////" ,  "...."]

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

def construct_dict(status_dict, 
    cg_lat, cg_util, 
    ideal_layer_cycle, ideal_layer_sec, ideal_layer_throughput, 
    sram_bw_ideal_ifmap_rd, sram_bw_ideal_filter_rd, sram_bw_ideal_ofmap_wr, sram_bw_ideal_total,
    real_layer_cycle, real_layer_sec, real_layer_throughput, 
    sram_bw_real_ifmap_rd, sram_bw_real_filter_rd, sram_bw_real_ofmap_wr, sram_bw_real_total,
    dynamic_cycle_ireg, dynamic_cycle_wreg, dynamic_cycle_mac
):
    cg_dict = dict()
    cg_dict['pe_cycle'] = cg_lat
    cg_dict['utilization'] = cg_util
    status_dict['cg'] = cg_dict

    ideal_dict = dict()
    ideal_rt_dict = dict()
    ideal_rt_dict['layer_cycle'] = ideal_layer_cycle
    ideal_rt_dict['layer_sec'] = ideal_layer_sec
    ideal_rt_dict['layer_throughput'] = ideal_layer_throughput
    ideal_dict['runtime'] = ideal_rt_dict
    ideal_bw_dict = dict()
    ideal_bw_dict['input_rd'] = sram_bw_ideal_ifmap_rd
    ideal_bw_dict['weight_rd'] = sram_bw_ideal_filter_rd
    ideal_bw_dict['output_wr'] = sram_bw_ideal_ofmap_wr
    ideal_bw_dict['total'] = sram_bw_ideal_total
    ideal_dict['bandwidth'] = ideal_bw_dict
    status_dict['ideal'] = ideal_dict

    real_dict = dict()
    real_rt_dict = dict()
    real_rt_dict['layer_cycle'] = real_layer_cycle
    real_rt_dict['layer_sec'] = real_layer_sec
    real_rt_dict['layer_throughput'] = real_layer_throughput
    real_dict['runtime'] = real_rt_dict
    real_bw_dict = dict()
    real_bw_dict['input_rd'] = sram_bw_real_ifmap_rd
    real_bw_dict['weight_rd'] = sram_bw_real_filter_rd
    real_bw_dict['output_wr'] = sram_bw_real_ofmap_wr
    real_bw_dict['total'] = sram_bw_real_total
    real_dict['bandwidth'] = real_bw_dict
    status_dict['real'] = real_dict

    dynamic_cycle_dict = dict()
    dynamic_cycle_dict['ireg'] = dynamic_cycle_ireg
    dynamic_cycle_dict['wreg'] = dynamic_cycle_wreg
    dynamic_cycle_dict['mac'] = dynamic_cycle_mac
    status_dict['dynamic_cycle'] = dynamic_cycle_dict

    return status_dict

def get_layer_runtime(arch_name, nn_name, layer, dtf_name, base_out_dir): 
    """
    Returns ideal runtime cycle and stall cycles
    """
    data = parse_json(base_out_dir + '/' + arch_name + '/' + dtf_name + '/' + nn_name + '/' + layer + '/stats.json')
    ideal_rt_cyc = data['ideal']['runtime']['layer_cycle']
    real_rt_cyc = data['real']['runtime']['layer_cycle']
    return ideal_rt_cyc, real_rt_cyc-ideal_rt_cyc

def get_network_stats(arch_name, nn_name, dtf_name, base_out_dir): 
    """
    Returns ideal runtime cycle and stall cycles for nn
    """
    data = parse_json(base_out_dir + '/' + arch_name + '/' + dtf_name + f'/{nn_name}.json')
    ideal_rt_cyc = data['ideal']['runtime']['layer_cycle']
    real_rt_cyc = data['real']['runtime']['layer_cycle']
    real_input_rd = data['real']['bandwidth']['input_rd']
    real_weight_rd = data['real']['bandwidth']['weight_rd']
    real_output_wr = data['real']['bandwidth']['output_wr']
    return ideal_rt_cyc, real_rt_cyc-ideal_rt_cyc, real_input_rd, real_weight_rd, real_output_wr


