import trace_gen
import argparse
import os
import pathlib
import utils
import subprocess
import numpy as np
import os.path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager
from utils import cor
import parse_usystolic_csv as systolic_data
from os.path import exists

_TRANCEGEN_DIR = os.environ['TRANCEGEN_DIR']

def construct_argparser():
    parser = argparse.ArgumentParser(description='Run Configuration')
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        help='Output Folder',
                        default=f'{_TRANCEGEN_DIR}/output_dir',
                        )
    parser.add_argument('-ap',
                        '--arch_proj_file_path',
                        type=str,
                        help='Hardware Architecture Projection Def file Path',
                        default=f'{_TRANCEGEN_DIR}/configs/arch/arch_tlut_systolic_projection_bank4_block16.yml',
                        )

    parser.add_argument('-nn',
                        '--nn_path',
                        type=str,
                        help='Problem Dimension Path',
                        default=f'{_TRANCEGEN_DIR}/configs/workloads/convnet_graph/layers.yaml',
                        )

    parser.add_argument('-dtfs',
                        '--dtf_path',
                        type=str,
                        help='Datfflow Path',
                        default=f'{_TRANCEGEN_DIR}/configs/dataflow/dtfs.yaml',
                        )
    parser.add_argument('--no_update',
                        action='store_true',
                        help='Bypassing trace gen',
                        default=False,
                        )
    parser.add_argument('--conv_only',
                        action='store_true',
                        help='Only comparing conv layers',
                        default=False,
                        )
    parser.add_argument('--plot_rect',
                        action='store_true',
                        help='Also plot rectangular version',
                        default=False,
                        )

    return parser

def gen_all(arch_names, nn_layer_names, dtf_names, output_path, network_name):
    
    # print(nn_layer_names, dtf_names)
    for arch in arch_names:
        print(utils.bcolors.OKGREEN + f'Processing {arch}' + utils.bcolors.ENDC)
        arch_n = arch + '.yml'
        for dtf in dtf_names:
            print(utils.bcolors.OKBLUE +f'  Using {dtf} dataflow' + utils.bcolors.ENDC)
            dtf_n = dtf + '.yaml'
            for layer in nn_layer_names:
                print(utils.bcolors.OKCYAN+ f'      Processing {layer}' + utils.bcolors.ENDC)
                layer_n = layer + '.yaml'
                in_arr = ['python3', f'{_TRANCEGEN_DIR}/trace_gen.py', 
                    '-pp', f'{_TRANCEGEN_DIR}/configs/workloads/{network_name}_graph/'+layer_n, 
                    '-ap', f'{_TRANCEGEN_DIR}/configs/arch/'+arch_n,
                    '-dp', f'{_TRANCEGEN_DIR}/configs/dataflow/'+dtf_n, 
                    '-o', output_path]
                # print(in_arr)
                p = subprocess.Popen(in_arr)
                output, error = p.communicate()
                if output != None: print(output)

def parse_names_proj(arch_proj_path, nn_path, dtf_path):
    arch_names = utils.parse_yaml(arch_proj_path) 
    nn_layer_names = utils.parse_yaml(nn_path)
    dtf_names = utils.parse_yaml(dtf_path)
    return arch_names, nn_layer_names, dtf_names

def plot_per_layer_data(data1, data2, data3, type_str, nn_layer_names, output_path, nn_name):
    # TODO: Make pretty
    font = {'family':'Times New Roman', 'size': 5}
    matplotlib.rc('font', **font)
    my_dpi = 300
    fig_h = 1
    fig_w = 3.3115

    x_axis = nn_layer_names
    x_idx = np.arange(len(x_axis))

    width = 0.2

    fig, rt_ax = plt.subplots(figsize=(fig_w, fig_h))

    # runtime plot
    if type_str == 'lat':
        type_str_ = 'Latency in cycle'
        # assert data2 != None
        # assert data3 == None
        rt_ax.bar(x_idx, data1, width, alpha=0.99, color=cor.grey1, hatch=None, label='ideal')
        rt_ax.bar(x_idx, data2, width, bottom=data1, alpha=0.99, color=cor.grey2, hatch=None, label='stall')
    elif type_str == 'bw':
        # assert data2 != None
        # assert data3 != None
        type_str_ = 'Bandwidth (GB/s)'
        rt_ax.bar(x_idx, data1, width, alpha=0.99, color=cor.grey1, hatch=None, label='ireg')
        rt_ax.bar(x_idx, data2, width, bottom=data1, alpha=0.99, color=cor.grey2, hatch=None, label='wreg')
        rt_ax.bar(x_idx, data2, width, bottom=np.array(data1) + np.array(data2), alpha=0.99, color=cor.grey3, hatch=None, label='oreg')
    elif type_str == 'util':
        # assert data2 == None
        # assert data3 == None
        type_str_ = 'Utilization %'
        rt_ax.bar(x_idx, data1, width, alpha=0.99, color=cor.grey1, hatch=None, label=None)
        
    rt_ax.set_ylabel(type_str_)
    rt_ax.minorticks_off()
    bars, labels = rt_ax.get_legend_handles_labels()
    
    plt.xlim(x_idx[0]-0.5, x_idx[-1]+0.5)
    rt_ax.set_xticks(x_idx)
    rt_ax.set_xticklabels(x_axis, rotation=45)
    plt.yscale("linear")
    if type_str == 'lat' or type_str == 'bw':
        rt_ax.legend(bars, labels, loc="lower right", ncol=1, frameon=True)

    fig.tight_layout()
    plt.savefig(output_path + f'{nn_name}_{type_str}.pdf', bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)

def gen_network_stats(arch_name, nn_layer_names, dtf_name, output_path, nn_name, conv_only=False):
    """
    This function generates stats for 1 arch and 1 dtf
    """
    # parsing output stats across all layers
    cg_pe_cycle = 0
    cg_util = 0
    ideal_rt_cyc = 0
    ideal_rt_sec = 0
    ideal_rt_thrpt = 0
    ideal_bw_input_rd = 0
    ideal_bw_wght_rd = 0
    ideal_bw_output_wr = 0
    ideal_bw_total = 0
    real_rt_cyc = 0
    real_rt_sec = 0
    real_rt_thrpt = 0
    real_bw_input_rd = 0
    real_bw_wght_rd = 0
    real_bw_output_wr = 0
    real_bw_total = 0
    dyn_ireg = 0
    dyn_wreg = 0
    dyn_mac = 0

    # array for plots
    cg_util_ = [] # util plot
    ideal_rt_cyc_ = [] # lat plot
    real_rt_cyc_ = [] 
    real_bw_input_rd_ = [] # bw plot
    real_bw_wght_rd_ = []
    real_bw_output_wr_ = []

    for layer in nn_layer_names:
        data = utils.parse_json(output_path + '/' + arch_name + '/' + dtf_name + '/' + nn_name + '/' + layer + '/stats.json')
        cg_pe_cycle += data['cg']['pe_cycle']
        cg_util += data['cg']['utilization']
        cg_util_.append(data['cg']['utilization'])
        ideal_rt_cyc += data['ideal']['runtime']['layer_cycle']
        ideal_rt_cyc_.append(data['ideal']['runtime']['layer_cycle'])
        ideal_layer_sec = data['ideal']['runtime']['layer_sec']
        ideal_layer_thrpt = data['ideal']['runtime']['layer_throughput']
        ideal_rt_sec += ideal_layer_sec
        ideal_total_data_so_far = ideal_rt_sec * ideal_rt_thrpt
        ideal_rt_thrpt = float(ideal_total_data_so_far + ideal_layer_sec * ideal_layer_thrpt) / ideal_rt_sec
        ideal_bw_input_rd += data['ideal']['bandwidth']['input_rd'] * ideal_layer_sec
        ideal_bw_wght_rd += data['ideal']['bandwidth']['weight_rd'] * ideal_layer_sec
        ideal_bw_output_wr += data['ideal']['bandwidth']['output_wr'] * ideal_layer_sec
        ideal_bw_total += data['ideal']['bandwidth']['total'] * ideal_layer_sec
        real_rt_cyc += data['real']['runtime']['layer_cycle']
        real_rt_cyc_.append(data['real']['runtime']['layer_cycle'])
        real_layer_sec = data['real']['runtime']['layer_sec']
        real_layer_thrpt = data['real']['runtime']['layer_throughput']
        real_rt_sec += real_layer_sec
        real_total_data_so_far = real_rt_sec * real_rt_thrpt
        real_rt_thrpt = float(real_total_data_so_far + real_layer_sec * real_layer_thrpt) / real_rt_sec
        real_bw_input_rd += data['real']['bandwidth']['input_rd'] * real_layer_sec
        real_bw_input_rd_.append(data['real']['bandwidth']['input_rd'])
        real_bw_wght_rd += data['real']['bandwidth']['weight_rd'] * real_layer_sec
        real_bw_wght_rd_.append(data['real']['bandwidth']['weight_rd'])
        real_bw_output_wr += data['real']['bandwidth']['output_wr'] * real_layer_sec
        real_bw_output_wr_.append(data['real']['bandwidth']['output_wr'])
        real_bw_total += data['real']['bandwidth']['total'] * real_layer_sec
        dyn_ireg += data['dynamic_cycle']['ireg']
        dyn_wreg += data['dynamic_cycle']['wreg']
        dyn_mac += data['dynamic_cycle']['mac']
    
    out_path = output_path + '/' + arch_name + '/' + dtf_name + '/'
    stall_rt_cyc_ = np.array(real_rt_cyc_)-np.array(ideal_rt_cyc_)

    if conv_only == False: # only gen per layer plots for all layers
        plot_per_layer_data(data1=cg_util_, data2=None, data3=None, type_str='util', 
            nn_layer_names=nn_layer_names, output_path=out_path, nn_name=nn_name)
        plot_per_layer_data(data1=real_bw_input_rd_, data2=real_bw_wght_rd_, data3=real_bw_output_wr_, type_str='bw', 
            nn_layer_names=nn_layer_names, output_path=out_path, nn_name=nn_name)
        plot_per_layer_data(data1=ideal_rt_cyc_, data2=stall_rt_cyc_, data3=None, type_str='lat', 
            nn_layer_names=nn_layer_names, output_path=out_path, nn_name=nn_name)

    # taking average
    cg_util /= len(nn_layer_names)
    ideal_bw_input_rd /= ideal_rt_sec
    ideal_bw_wght_rd /= ideal_rt_sec
    ideal_bw_output_wr /= ideal_rt_sec
    ideal_bw_total /= ideal_rt_sec
    real_bw_input_rd /= real_rt_sec
    real_bw_wght_rd /= real_rt_sec
    real_bw_output_wr /= real_rt_sec
    real_bw_total /= real_rt_sec

    if conv_only == True:
        json_file = output_path + '/' + arch_name + '/' + dtf_name + '/' + f"{nn_name}_convonly.json"
    else: 
        json_file = output_path + '/' + arch_name + '/' + dtf_name + '/' + f"{nn_name}.json"
    status_dict = dict()
    status_dict = utils.construct_dict(status_dict,
        cg_pe_cycle, cg_util,
        ideal_rt_cyc, ideal_rt_sec, ideal_rt_thrpt,
        ideal_bw_input_rd, ideal_bw_wght_rd, ideal_bw_output_wr,ideal_bw_total,
        real_rt_cyc, real_rt_sec, real_rt_thrpt, 
        real_bw_input_rd, real_bw_wght_rd, real_bw_output_wr, real_bw_total, 
        dyn_ireg, dyn_wreg, dyn_mac)
    utils.store_json(json_file, status_dict, indent=4)
    return cg_util_, ideal_rt_cyc_, stall_rt_cyc_, real_bw_input_rd_, real_bw_wght_rd_, real_bw_output_wr_

def projection(tlut_arch_names, output_path, nn_name, conv_only, bank, block, plot_rect):
    # read tlut bw and lat data
    # runtime: stacked bar
    tlut_lat = []
    # bw: stacked bar
    tlut_bw = []

    for arch_name in tlut_arch_names:
        ideal, stall, i, w, o = utils.get_network_stats(arch_name, nn_name, dtf_name, output_path, conv_only)
        tlut_lat.append(ideal+stall)
        tlut_bw.append(i+w+o)
    
    if plot_rect == True:
        rect_names = [n+'_rect' for n in tlut_arch_names]
        rect_lat = []
        rect_bw = []
        for name in rect_names:
            ideal, stall, i, w, o = utils.get_network_stats(name, nn_name, dtf_name, output_path, conv_only)
            rect_lat.append(ideal+stall)
            rect_bw.append(i+w+o)
    
    # get sys data
    bsys_bw, bsys_lat = systolic_data.get_sys_bw_lat(design='bsys', nn_name=nn_name, conv_only=conv_only, bank=bank, block=block, dim_arr=[16,32,64,128])
    usys_bw, usys_lat = systolic_data.get_sys_bw_lat(design='usys', nn_name=nn_name, conv_only=conv_only, bank=bank, block=block, dim_arr=[16,32,64,128])

    print("usys=", usys_lat)
    print('bsys=', bsys_lat)
    print('tlut=', tlut_lat)
    if plot_rect: print('tlut-rec=', rect_lat)

    # **************** start ploting ****************
    # TODO: Make pretty
    font = {'family':'Times New Roman', 'size': 5}
    matplotlib.rc('font', **font)
    my_dpi = 300
    fig_h = 1.2
    fig_w = 3.3115

    x_axis = tlut_arch_names
    x_idx = np.arange(len(x_axis))

    # runtime line plot
    fig, rt_ax = plt.subplots(figsize=(fig_w, fig_h))
    
    ncol = 3
    rt_ax.plot(x_idx, usys_lat, '-s', color=cor.tlut_mint, ms=4, label='Unary systolic')
    rt_ax.plot(x_idx, bsys_lat, '-o', color=cor.tlut_nude, ms=4, label='Binary systolic')
    rt_ax.plot(x_idx, tlut_lat, '-^', color=cor.tlut_blue, ms=4, label='Temporal_LUT')
    if plot_rect: 
        rt_ax.plot(x_idx, rect_lat, '-^', color=cor.tlut_pink, ms=4, label='Temporal_LUT (rectangular)')
        ncol = 4
    rt_ax.set_ylabel('Latency in cycle')
    rt_ax.minorticks_off()

    bars, labels = rt_ax.get_legend_handles_labels()
    
    plt.xlim(x_idx[0]-0.5, x_idx[-1]+0.5)
    rt_ax.set_xticks(x_idx)
    rt_ax.set_xticklabels(x_axis, rotation=10)
    plt.yscale("linear")
    rt_ax.legend(bars, labels, loc="upper center", ncol=ncol, frameon=True)
    
    print("rt_ax ylim: ", rt_ax.get_ylim())

    # Fine tuning limit and ticks
    # rt_ax.set_ylim((0, 3700000))
    # rt_ax.set_yticks((0, 1000000, 2000000, 3000000))

    fig.tight_layout()
    if conv_only==True:
        append_convonly = '_convonly'  
    else: 
        append_convonly = ''
   
    append_mem = f'_bank{bank}_block{block}'
    
    plt.savefig(output_path + f'/proj_{nn_name}_{dtf_name}_rt' + append_mem + append_convonly + '.pdf', bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)

    # bw line plot
    fig, bw_ax = plt.subplots(figsize=(fig_w, fig_h))
    
    bw_ax.plot(x_idx, usys_bw, '-s', color=cor.tlut_mint, ms=4, label='Unary systolic')
    bw_ax.plot(x_idx, bsys_bw, '-o', color=cor.tlut_nude, ms=4, label='Binary systolic')
    bw_ax.plot(x_idx, tlut_bw, '-^', color=cor.tlut_blue, ms=4, label='Temporal_LUT')
    if plot_rect: bw_ax.plot(x_idx, rect_bw, '-^', color=cor.tlut_pink, ms=4, label='Temporal_LUT (rectangular)')
    bw_ax.set_ylabel('Bandwidth (GB/s)')
    bw_ax.minorticks_off()

    bars, labels = bw_ax.get_legend_handles_labels()
    
    plt.xlim(x_idx[0]-0.5, x_idx[-1]+0.5)
    bw_ax.set_xticks(x_idx)
    bw_ax.set_xticklabels(x_axis, rotation=10)
    plt.yscale("linear")
    bw_ax.legend(bars, labels, loc="upper center", ncol=ncol, frameon=True)
    
    print("bw_ax ylim: ", bw_ax.get_ylim())

    # Fine tuning limit and ticks
    # bw_ax.set_ylim((0, 3700000))
    # bw_ax.set_yticks((0, 1000000, 2000000, 3000000))

    fig.tight_layout()
    
    plt.savefig(output_path + f'/proj_{nn_name}_{dtf_name}_bw' + append_mem + append_convonly + '.pdf', bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)
    
    return usys_lat, bsys_lat, tlut_lat, usys_bw, bsys_bw, tlut_bw

if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    # extract bank and block
    bank = int(args.arch_proj_file_path.split('bank')[1].split('_block')[0])
    block = int(args.arch_proj_file_path.split('block')[1].split('.yml')[0])

    nn_path = pathlib.Path(args.nn_path).resolve()
    arch_proj_file_path = pathlib.Path(args.arch_proj_file_path).resolve()
    dtf_path = pathlib.Path(args.dtf_path).resolve()
    output_path = args.output_dir
    network_name = args.nn_path.split('workloads/')[1].split('_graph')[0]

    conv_only = args.conv_only
    if conv_only: 
        nn_conv_only_path = args.nn_path.split('layers')[0]+'layers_conv.yaml'
        nn_path = pathlib.Path(nn_conv_only_path).resolve()

    arch_proj_file_name = args.arch_proj_file_path.split('arch/')[1].split('.yml')[0]
    arch_names, nn_layer_names, dtf_names = parse_names_proj(arch_proj_file_path, nn_path, dtf_path)
    print(utils.bcolors.CHEEZY + f'============= Perf projection kick off for arch proj file {arch_proj_file_name} =============' + utils.bcolors.ENDC)
    print(f'- arch: {arch_names}\n- dtf: {dtf_names}\n- nn layers: {nn_layer_names}')
    arch_names_flat = arch_names # already flattened

    rect_arch_names_flat = [n+'_rect' for n in arch_names_flat]
    if args.no_update == False: # redoing trace gen
        print(utils.bcolors.OKBLUE + f'********** Regenerating all traces ***********'+ utils.bcolors.ENDC)
        gen_all(arch_names_flat, nn_layer_names, dtf_names, output_path, network_name)
        if args.plot_rect == True: gen_all(rect_arch_names_flat, nn_layer_names, dtf_names, output_path, network_name)

    print(utils.bcolors.HEADER + f'********** Generating network stats ***********'+ utils.bcolors.ENDC)
    for arch_name in arch_names_flat:
        for dtf_name in dtf_names:
            print(utils.bcolors.HEADER + f'{arch_name}/{dtf_name}' + utils.bcolors.ENDC)
            cg_util_, ideal_rt_cyc_, stall_rt_cyc_, real_bw_input_rd_, real_bw_wght_rd_, real_bw_output_wr_ = gen_network_stats(
                arch_name, nn_layer_names, dtf_name, output_path, network_name, conv_only)
    if args.plot_rect == True:
        for arch_name_rect in rect_arch_names_flat:
            for dtf_name in dtf_names:
                print(utils.bcolors.HEADER + f'{arch_name_rect}/{dtf_name}' + utils.bcolors.ENDC)
                cg_util_, ideal_rt_cyc_, stall_rt_cyc_, real_bw_input_rd_, real_bw_wght_rd_, real_bw_output_wr_ = gen_network_stats(
                    arch_name_rect, nn_layer_names, dtf_name, output_path, network_name, conv_only)

    print(utils.bcolors.OKGREEN + f'********** Projection ***********'+ utils.bcolors.ENDC)
    usys_lat, bsys_lat, tlut_lat, usys_bw, bsys_bw, tlut_bw = projection(
        arch_names, output_path, network_name, conv_only, bank, block, args.plot_rect)

    if not os.path.exists(output_path + f'/projection'):
        os.makedirs(output_path + f'/projection')

    projection_stats_file = utils.get_mem_sensitivity_stats_file_name(output_path, block, network_name, dtf_name, conv_only)
    
    print(utils.bcolors.HEADER + f'********** Appending stats to {projection_stats_file}**********' + utils.bcolors.ENDC)
    
    file_exists = exists(projection_stats_file)
    if file_exists:
        data = utils.parse_json(projection_stats_file)
    else:
        data = dict()

    cfg_str = f'{bank}_{block}'
    bsys_lat_pctg = [float(i) / float(j) for i, j in zip(bsys_lat, tlut_lat)]
    bsys_bw_pctg = [float(i) / float(j) for i, j in zip(bsys_bw, tlut_bw)]
    usys_lat_pctg = [float(i) / float(j) for i, j in zip(usys_lat, tlut_lat)]
    usys_bw_pctg = [float(i) / float(j) for i, j in zip(usys_bw, tlut_bw)]
    bsys_dict = dict()
    bsys_dict['lat'] = bsys_lat_pctg
    bsys_dict['bw'] = bsys_bw_pctg
    usys_dict = dict()
    usys_dict['lat'] = usys_lat_pctg
    usys_dict['bw'] = usys_bw_pctg
    
    cfg_dict = dict()
    cfg_dict['bsys'] = bsys_dict
    cfg_dict['usys'] = usys_dict

    data[cfg_str] = cfg_dict

    utils.store_json(projection_stats_file, data, indent=4)