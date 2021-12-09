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

def construct_names(design, nn_name, conv_only, bank, block, dim_arr=[16, 32, 64, 128]):
    arr = []
    for dim in dim_arr:
        if design == 'usys':
            if conv_only:
                str_name = f'proj{dim}_{dim}_04b_ut_016c_{nn_name}_convonly_ddr3_w__spm_sram_{bank}_{block}'
            else: 
                str_name = f'proj{dim}_{dim}_04b_ut_016c_{nn_name}_ddr3_w__spm_sram_{bank}_{block}'
        elif design == 'bsys':
            if conv_only:
                str_name = f'proj{dim}_{dim}_04b_bp_001c_{nn_name}_convonly_ddr3_w__spm_sram_{bank}_{block}'
            else:
                str_name = f'proj{dim}_{dim}_04b_bp_001c_{nn_name}_ddr3_w__spm_sram_{bank}_{block}'
        arr.append(str_name)
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

def get_sys_bw_lat(design, nn_name, conv_only, bank, block, dim_arr = [16, 32, 64, 128]):
    """
    Returns bw, lat lists.
    """
    names = construct_names(design, nn_name, conv_only, bank, block, dim_arr)
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
    design = 'bsys'
    # nn_name = 'alexnet'
    bank = 16
    block = 8
    bw, lat = get_sys_bw_lat(design, nn_name, conv_only, bank, block)
    print(bw, lat)

if __name__ == "__main__":
    main()