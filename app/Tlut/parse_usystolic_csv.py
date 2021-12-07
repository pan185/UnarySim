import utils
import os

_USYS_DIR = os.environ['USYS_OUT_DIR']
_TLUT_HOME = os.environ['TLUT_HOME']

# file postfix
lat_post = '_detail_real.csv'

# cell header name
# lat file
byte_fr_key_str = '	SRAM F RD bytes'
byte_ir_key_str = '	SRAM I RD bytes'
byte_or_key_str = '	SRAM O RD bytes'
byte_ow_key_str = '	SRAM O WR bytes'
lat_key_str = '	SRAM O WR stop'

def construct_file_path(result_name):
    return f'{_USYS_DIR}/{result_name}/simHwOut/{result_name}{lat_post}'

def construct_names(design, nn_name, conv_only, smallmemory):
    if conv_only==True and smallmemory==False:
        ind = 0
    elif conv_only==False and smallmemory==False:
        ind = 1
    elif conv_only==True and smallmemory==True:
        ind = 2
    elif conv_only==False and smallmemory==True:
        ind = 3
    
    if nn_name=='alexnet': postfix='_alex'
    else: postfix=''

    if design == 'usys':
        file_path = f'{_TLUT_HOME}/sys_u.yml'
    elif design == 'bsys':
        file_path = f'{_TLUT_HOME}/sys_b.yml'

    data = utils.parse_yaml(file_path)
    arr = [i+postfix for i in data[ind]['results']]
    return arr

def get_data_across_names(names, keystr, post_processing='sum'):
    sys_ = []
    for name in names:
        path = construct_file_path(name)
        csvFile = open(path, "r")
        arr = utils.get_all_values_for_given_key(csvFile=csvFile, key_str=keystr)
        if post_processing == 'sum':
            data = sum(arr)
        elif post_processing == 'max':
            data = max(arr)
        sys_.append(data)
    return sys_

def get_sys_bw_lat(design, nn_name, conv_only, smallmemory):
    """
    Returns bw, lat lists.
    """
    names = construct_names(design, nn_name, conv_only, smallmemory)
    byte_ir = get_data_across_names(names, byte_ir_key_str, 'sum')
    byte_fr = get_data_across_names(names, byte_fr_key_str, 'sum')
    byte_or = get_data_across_names(names, byte_or_key_str, 'sum')
    byte_ow = get_data_across_names(names, byte_ow_key_str, 'sum')
    byte = [sum(x) for x in zip(byte_ir, byte_fr, byte_or, byte_ow)]

    lat = get_data_across_names(names, lat_key_str, 'sum')

    bw = [i / j for i, j in zip(byte, lat)]

    return bw, lat

def main():
    nn_name = 'convnet'
    conv_only = True
    design = 'usys'
    # nn_name = 'alexnet'
    smallmemory = False
    bw, lat = get_sys_bw_lat(design, nn_name, conv_only, smallmemory)
    print(bw, lat)

if __name__ == "__main__":
    main()