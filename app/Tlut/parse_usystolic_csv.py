import utils
import os

_USYS_DIR = os.environ['USYS_OUT_DIR']

bsys_bw_names = ['bsys_4_bw', 'bsys_8_bw', 'bsys_16_bw', 'bsys_32_bw']
bsys_lat_names = ['bsys_4_lat', 'bsys_8_lat', 'bsys_16_lat', 'bsys_32_lat']
usys_bw_names = ['usys_4_bw', 'usys_8_bw', 'usys_16_bw', 'usys_32_bw']
usys_lat_names = ['usys_4_lat', 'usys_8_lat', 'usys_16_lat', 'usys_32_lat']

# cell header name
bw_key_str = '	SRAM BW Total (GBytes/sec)'
lat_key_str = '	Cycle Total (Cycles)'


def get_data_across_names(nn_name, conv_only, names, keystr):
    sys_ = []
    for name in names:
        if conv_only:
            csvFile = open(f'{_USYS_DIR}/{nn_name}/conv_only/{name}.csv', "r")
        else:
            csvFile = open(f'{_USYS_DIR}/{nn_name}/{name}.csv', "r")
        arr = utils.get_all_values_for_given_key(csvFile=csvFile, key_str=keystr)
        data = arr[len(arr)-1]
        sys_.append(data)
    return sys_

def get_bsys_bw_lat(nn_name, conv_only):
    """
    Returns bw, lat lists.
    """
    bsys_bw = get_data_across_names(nn_name, conv_only, bsys_bw_names, bw_key_str)
    bsys_lat = get_data_across_names(nn_name, conv_only, bsys_lat_names, lat_key_str)
    return bsys_bw, bsys_lat

def get_usys_bw_lat(nn_name, conv_only):
    """
    Returns bw, lat lists.
    """
    usys_bw = get_data_across_names(nn_name, conv_only, usys_bw_names, bw_key_str)
    usys_lat = get_data_across_names(nn_name, conv_only, usys_lat_names, lat_key_str)
    return usys_bw, usys_lat

def main():
    nn_name = 'convnet'
    conv_only = True
    print(get_bsys_bw_lat(nn_name, conv_only))
    print(get_usys_bw_lat(nn_name, conv_only))

if __name__ == "__main__":
    main()