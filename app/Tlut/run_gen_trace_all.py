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
                        '--arch_set_file_path',
                        type=str,
                        help='Hardware Architecture Path',
                        default=f'{_TRANCEGEN_DIR}/configs/arch/archs.yml',
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

def parse_names_vec(arch_set_path, nn_path, dtf_path):
    arch_data = utils.parse_yaml(arch_set_path)
    arch_sets = [arch_data[i]['name'] for i in range(len(arch_data))] 
    arch_names = [arch_data[i]['archs'] for i in range(len(arch_data))] 
    nn_layer_names = utils.parse_yaml(nn_path)
    dtf_names = utils.parse_yaml(dtf_path)
    return arch_sets, arch_names, nn_layer_names, dtf_names

def gen_network_stats(arch_name, nn_layer_names, dtf_name, output_path, nn_name):
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
    for layer in nn_layer_names:
        data = utils.parse_json(output_path + '/' + arch_name + '/' + dtf_name + '/' + nn_name + '/' + layer + '/stats.json')
        cg_pe_cycle += data['cg']['pe_cycle']
        cg_util += data['cg']['utilization']
        ideal_rt_cyc += data['ideal']['runtime']['layer_cycle']
        ideal_layer_sec = data['ideal']['runtime']['layer_sec']
        ideal_layer_thrpt = data['ideal']['runtime']['layer_throughput']
        ideal_rt_sec += ideal_layer_sec
        ideal_total_data_so_far = ideal_rt_sec * ideal_rt_thrpt
        ideal_rt_thrpt = float(ideal_total_data_so_far + ideal_layer_sec * ideal_layer_thrpt) / ideal_rt_sec
        ideal_bw_input_rd += data['ideal']['bandwidth']['input_rd']
        ideal_bw_wght_rd += data['ideal']['bandwidth']['weight_rd']
        ideal_bw_output_wr += data['ideal']['bandwidth']['output_wr']
        ideal_bw_total += data['ideal']['bandwidth']['total']
        real_rt_cyc += data['real']['runtime']['layer_cycle']
        real_layer_sec = data['real']['runtime']['layer_sec']
        real_layer_thrpt = data['real']['runtime']['layer_throughput']
        real_rt_sec += real_layer_sec
        real_total_data_so_far = real_rt_sec * real_rt_thrpt
        real_rt_thrpt = float(real_total_data_so_far + real_layer_sec * real_layer_thrpt) / real_rt_sec
        real_bw_input_rd += data['real']['bandwidth']['input_rd']
        real_bw_wght_rd += data['real']['bandwidth']['weight_rd']
        real_bw_output_wr += data['real']['bandwidth']['output_wr']
        real_bw_total += data['real']['bandwidth']['total']
        dyn_ireg += data['dynamic_cycle']['ireg']
        dyn_wreg += data['dynamic_cycle']['wreg']
        dyn_mac += data['dynamic_cycle']['mac']
    
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

def compare_dtf(arch_name, nn_name, dtf_names, nn_layer_names, out_dir): 
    """
    This function reads two dataflow stats files across an arch across L layers and also total network
    Metric: [real][layer_cycle]
    """
    assert len(dtf_names) == 2
    arch_output_path = out_dir + '/' + arch_name

    ideal0 = []
    ideal1 = []
    stall0 = []
    stall1 = []

    for layer in nn_layer_names:
        ideal, stall = utils.get_layer_runtime(arch_name, nn_name, layer, dtf_names[0], out_dir)
        ideal0.append(ideal)
        stall0.append(stall)

    for layer in nn_layer_names:
        ideal, stall = utils.get_layer_runtime(arch_name, nn_name, layer, dtf_names[1], out_dir)
        ideal1.append(ideal)
        stall1.append(stall)

    # TODO: Make pretty
    font = {'family':'Times New Roman', 'size': 6}
    matplotlib.rc('font', **font)
    my_dpi = 300
    fig_h = 1
    fig_w = 3.3115

    x_axis = nn_layer_names
    x_idx = np.arange(len(x_axis))

    width = 0.3

    fig, power_ax = plt.subplots(figsize=(fig_w, fig_h))
    power_ax.bar(x_idx - 0.5 * width, ideal0, width, alpha=0.99, color=cor.grey1, hatch=cor.patterns[0], label='ideal '+dtf_names[0])
    power_ax.bar(x_idx - 0.5 * width, stall0, width, bottom=ideal0, alpha=0.99, color=cor.grey2, hatch=cor.patterns[0], label='stall '+dtf_names[0])
    power_ax.set_ylabel('Latency per layer in cycle')
    power_ax.minorticks_off()

    power2_ax = power_ax.twinx()
    power2_ax.bar(x_idx + 0.5 * width, ideal1, width, alpha=0.99, color=cor.grey1, hatch=cor.patterns[1], label='ideal '+dtf_names[1])
    power2_ax.bar(x_idx + 0.5 * width, stall1, width, bottom=ideal1, alpha=0.99, color=cor.grey2, hatch=cor.patterns[1], label='stall '+dtf_names[1])
    power2_ax.yaxis.set_visible(False)
    # power2_ax.set_yticklabels('')

    bars, labels = power_ax.get_legend_handles_labels()
    bars2, labels2 = power2_ax.get_legend_handles_labels()
    

    plt.xlim(x_idx[0]-0.5, x_idx[-1]+0.5)
    power_ax.set_xticks(x_idx)
    power_ax.set_xticklabels(x_axis)
    plt.yscale("linear")
    power_ax.legend(bars + bars2, labels + labels2, loc="lower right", ncol=1, frameon=True)
    
    # print("ax ylim: ", power_ax.get_ylim())

    if '12_10' in arch_name:
        power_ax.set_ylim((0, 3000000))
        power2_ax.set_ylim((0, 3000000))
        power_ax.set_yticks((0, 1000000, 2000000, 3000000))
        power_ax.set_yticklabels(("{:2d}".format(0), "{:2d}".format(1000000), "{:2d}".format(2000000), "{:2d}".format(3000000)))
    else:
        power_ax.set_ylim((0, 1500000))
        power2_ax.set_ylim((0, 1500000))
        power_ax.set_yticks((0, 500000, 1000000, 1500000))
        power_ax.set_yticklabels(("{:2d}".format(0), "{:2d}".format(500000), "{:2d}".format(1000000), "{:2d}".format(1500000)))


    fig.tight_layout()
    plt.savefig(arch_output_path + f'/{nn_name}_dtf_comparison.pdf', bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)

def compare_arch_set(arch_set, arch_names, nn_name, dtf_name, out_dir): 
    """
    This function compares archs stats
    """
    # runtime: stacked bar
    ideal_arr = []
    stall_arr = []

    # bw: stacked bar
    i_bw = []
    w_bw = []
    o_bw = []

    for arch_name in arch_names:
        ideal, stall, i, w, o = utils.get_network_stats(arch_name, nn_name, dtf_name, out_dir)
        ideal_arr.append(ideal)
        stall_arr.append(stall)
        i_bw.append(i)
        w_bw.append(w)
        o_bw.append(o)

    # TODO: Make pretty
    font = {'family':'Times New Roman', 'size': 5}
    matplotlib.rc('font', **font)
    my_dpi = 300
    fig_h = 1
    fig_w = 3.3115

    x_axis = arch_names
    x_idx = np.arange(len(x_axis))

    width = 0.2

    # runtime plot
    fig, rt_ax = plt.subplots(figsize=(fig_w, fig_h))
    rt_ax.bar(x_idx, ideal_arr, width, alpha=0.99, color=cor.grey1, hatch=None, label='ideal')
    rt_ax.bar(x_idx, stall_arr, width, bottom=ideal_arr, alpha=0.99, color=cor.grey2, hatch=None, label='stall')
    rt_ax.set_ylabel('Latency in cycle')
    rt_ax.minorticks_off()

    bars, labels = rt_ax.get_legend_handles_labels()
    
    plt.xlim(x_idx[0]-0.5, x_idx[-1]+0.5)
    rt_ax.set_xticks(x_idx)
    rt_ax.set_xticklabels(x_axis)
    plt.yscale("linear")
    rt_ax.legend(bars, labels, loc="lower right", ncol=1, frameon=True)
    
    print("rt_ax ylim: ", rt_ax.get_ylim())

    if arch_set == 'archs':
        rt_ax.set_ylim((0, 3700000))
        rt_ax.set_yticks((0, 1000000, 2000000, 3000000))
    elif 'w1' in arch_set:
        rt_ax.set_ylim((0, 31000000))
        rt_ax.set_yticks((0, 10000000, 20000000, 30000000))
    elif 'w2' in arch_set:
        rt_ax.set_ylim((0, 13000000))
        rt_ax.set_yticks((0, 5000000, 10000000))
    elif 'w4' in arch_set:
        rt_ax.set_ylim((0,   6200000))
        rt_ax.set_yticks((0, 2000000, 4000000, 6000000))
    

    fig.tight_layout()
    plt.savefig(output_path + f'/{nn_name}_{dtf_name}_{arch_set}_comparison_rt.pdf', bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)

    # bw plot
    fig, bw_ax = plt.subplots(figsize=(fig_w, fig_h))
    bw_ax.bar(x_idx, i_bw, width, alpha=0.99, color=cor.grey1, hatch=None, label='ireg')
    bw_ax.bar(x_idx, w_bw, width, bottom=i_bw, alpha=0.99, color=cor.grey2, hatch=None, label='wreg')
    bw_ax.bar(x_idx, o_bw, width, bottom=np.array(w_bw) + np.array(i_bw), alpha=0.99, color=cor.grey3, hatch=None, label='oreg')

    bw_ax.set_ylabel('Bandwidth (GB/s)')
    bw_ax.minorticks_off()

    bars, labels = bw_ax.get_legend_handles_labels()
    
    plt.xlim(x_idx[0]-0.5, x_idx[-1]+0.5)
    bw_ax.set_xticks(x_idx)
    bw_ax.set_xticklabels(x_axis)
    plt.yscale("linear")
    bw_ax.legend(bars, labels, loc="lower right", ncol=1, frameon=True)
    
    print("bw_ax ylim: ", bw_ax.get_ylim())

    if arch_set == 'archs':
        bw_ax.set_ylim((0, 50))
        bw_ax.set_yticks((0, 10, 20, 30, 40, 50))
    if 'w1' in arch_set:
        bw_ax.set_ylim((0, 3.1))
        bw_ax.set_yticks((0, 1, 2, 3))
    elif 'w2' in arch_set:
        bw_ax.set_ylim((0, 3.4))
        bw_ax.set_yticks((0, 1, 2, 3))
    elif 'w4' in arch_set:
        bw_ax.set_ylim((0, 4))
        bw_ax.set_yticks((0, 1, 2, 3, 4))
    

    fig.tight_layout()
    plt.savefig(output_path + f'/{nn_name}_{dtf_name}_{arch_set}_comparison_bw.pdf', bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)
    return ideal_arr, stall_arr, i_bw, w_bw, o_bw

def compare_all_arch_sets(arch_set_names_flat, arch_names_flat, ideal_, stall_, i_bw_, w_bw_, o_bw_):
    # TODO: Make pretty
    font = {'family':'Times New Roman', 'size': 5}
    matplotlib.rc('font', **font)
    my_dpi = 300
    fig_h = 1
    fig_w = 3.3115

    x_axis = arch_set_names_flat
    x_axis_real = arch_names_flat
    x_idx = np.arange(len(x_axis))
    x_idx_real = np.arange(len(x_axis_real))

    width = 0.1 # the width of the bars

    labels = x_axis

    x = np.arange(len(labels))  # the label locations
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ireg0 = [i[0] for i in i_bw_]
    wreg0 = [w[0] for w in w_bw_]
    rects1 = ax.bar(x - width - width/2, ireg0, width, bottom=None, alpha=0.99, color=cor.grey1, hatch=None, label='i16 ireg')
    rects1_2 = ax.bar(x - width - width/2, wreg0, width, bottom=ireg0, alpha=0.99, color=cor.mint1, hatch=None, label='i16 wreg')
    rects1_3 = ax.bar(x - width - width/2, [o[0] for o in o_bw_], width, bottom=np.array(ireg0) + np.array(wreg0), alpha=0.99, color=cor.blue1, hatch=None, label='i16 oreg')
    ireg1 = [i[1] for i in i_bw_]
    wreg1 = [w[1] for w in w_bw_]
    rects2 = ax.bar(x - width/2, ireg1, width, bottom=None, alpha=0.99, color=cor.grey2, hatch=None, label='i32 ireg')
    rects2_2 = ax.bar(x - width/2, wreg1, width, bottom=ireg1, alpha=0.99, color=cor.mint2, hatch=None, label='i32 wreg')
    rects2_3 = ax.bar(x - width/2, [o[1] for o in o_bw_], width, bottom=np.array(ireg1) + np.array(wreg1), alpha=0.99, color=cor.blue2, hatch=None, label='i32 oreg')
    ireg2 = [i[2] for i in i_bw_]
    wreg2 = [w[2] for w in w_bw_]
    rects3 = ax.bar(x + width/2, ireg2, width, bottom=None, alpha=0.99, color=cor.grey3, hatch=None, label='i64 ireg')
    rects3_2 = ax.bar(x + width/2, wreg2, width, bottom=ireg2, alpha=0.99, color=cor.mint3, hatch=None, label='i64 wreg')
    rects3_3 = ax.bar(x + width/2, [o[2] for o in o_bw_], width, bottom=np.array(ireg2) + np.array(wreg2), alpha=0.99, color=cor.blue3, hatch=None, label='i64 oreg')
    ireg3 = [i[3] for i in i_bw_]
    wreg3 = [w[3] for w in w_bw_]
    rects4 = ax.bar(x + width + width/2, ireg3, width, bottom=None, alpha=0.99, color=cor.grey4, hatch=None, label='i128 ireg')
    rects4_2 = ax.bar(x + width + width/2, wreg3, width, bottom=ireg3, alpha=0.99, color=cor.mint4, hatch=None, label='i128 wreg')
    rects4_3 = ax.bar(x + width + width/2, [o[3] for o in o_bw_], width, bottom=np.array(ireg3) + np.array(wreg3), alpha=0.99, color=cor.blue4, hatch=None, label='i128 oreg')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Bandwidth (GB/s)')
    # ax.set_xticks(x, labels)
    # ax.legend()

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    ax.minorticks_off()

    bars, labels = ax.get_legend_handles_labels()
    
    plt.xlim(x_idx[0]-0.5, x_idx[-1]+0.5)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_axis)
    plt.yscale("linear")
    ax.legend(bars, labels, loc="upper center", ncol=int(len(arch_names_flat)/len(arch_set_names_flat)), frameon=True)
    
    print("ax ylim: ", ax.get_ylim())

    # ax.set_ylim((0, 30000000))
    # ax.set_yticks((0, 10000000, 20000000, 30000000))

    fig.tight_layout()
    plt.savefig(output_path + f'/bw.pdf', bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)

    # bw plots
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ideal0 = [ideal[0] for ideal in ideal_]
    rects1 = ax.bar(x - width - width/2, ideal0, width, bottom=None, alpha=0.99, color=cor.grey1, hatch=None, label='i16 ideal')
    rects1_2 = ax.bar(x - width - width/2, [stall[0] for stall in stall_], width, bottom=ideal0, alpha=0.99, color=cor.mint1, hatch=None, label='i16 stall')
    ideal1 = [ideal[1] for ideal in ideal_]
    rects2 = ax.bar(x - width/2, ideal1, width, bottom=None, alpha=0.99, color=cor.grey2, hatch=None, label='i32 ideal')
    rects2_2 = ax.bar(x - width/2, [stall[1] for stall in stall_], width, bottom=ideal1, alpha=0.99, color=cor.mint2, hatch=None, label='i32 stall')
    ideal2 = [ideal[2] for ideal in ideal_]
    rects3 = ax.bar(x + width/2, ideal2, width, bottom=None, alpha=0.99, color=cor.grey3, hatch=None, label='i64 ideal')
    rects3_2 = ax.bar(x + width/2, [stall[2] for stall in stall_], width, bottom=ideal2, alpha=0.99, color=cor.mint3, hatch=None, label='i64 stall')
    ideal3 = [ideal[3] for ideal in ideal_]
    rects3 = ax.bar(x + width + width/2, ideal2, width, bottom=None, alpha=0.99, color=cor.grey4, hatch=None, label='i128 ideal')
    rects3_2 = ax.bar(x + width + width/2, [stall[3] for stall in stall_], width, bottom=ideal3, alpha=0.99, color=cor.mint4, hatch=None, label='i128 stall')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Latency in cycle')
    # ax.set_xticks(x, labels)
    # ax.legend()

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    ax.minorticks_off()

    bars, labels = ax.get_legend_handles_labels()
    
    plt.xlim(x_idx[0]-0.5, x_idx[-1]+0.5)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_axis)

    # ax.set_xticks(x_idx_real)
    # ax.set_xticklabels(x_axis_real, rotation=45)
    # print(x_axis_real)

    plt.yscale("log")
    ax.legend(bars, labels, loc="upper center", ncol=int(len(arch_names_flat)/len(arch_set_names_flat)), frameon=True)
    
    print("latency ax ylim: ", ax.get_ylim())

    ax.set_ylim((1, 15000000))
    ax.set_yticks((1, 100000, 1000000, 10000000, 100000000))

    fig.tight_layout()
    plt.savefig(output_path + f'/latency.pdf', bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)

if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    nn_path = pathlib.Path(args.nn_path).resolve()
    arch_set_file_path = pathlib.Path(args.arch_set_file_path).resolve()
    dtf_path = pathlib.Path(args.dtf_path).resolve()
    output_path = args.output_dir
    network_name = args.nn_path.split('workloads/')[1].split('_graph')[0]

    arch_set_file_name = args.arch_set_file_path.split('arch/')[1].split('.yml')[0]
    arch_set_names_flat, arch_names, nn_layer_names, dtf_names = parse_names_vec(arch_set_file_path, nn_path, dtf_path)
    print(utils.bcolors.UNDERLINE + f'============= Temporal-LUT simulation kick off for archset file {arch_set_file_name} =============' + utils.bcolors.ENDC)
    print(f'- arch set: {arch_set_names_flat}\n- arch: {arch_names}\n- dtf: {dtf_names}\n- nn layers: {nn_layer_names}')

    arch_names_flat = list(np.concatenate(arch_names))

    if args.no_update == False: # bypassing trace gen
        gen_all(arch_names_flat, nn_layer_names, dtf_names, output_path, network_name)

    print(utils.bcolors.HEADER + f'********** Generating network stats ***********'+ utils.bcolors.ENDC)
    for arch_name in arch_names_flat:
        for dtf_name in dtf_names:
            print(utils.bcolors.HEADER + f'{arch_name}/{dtf_name}' + utils.bcolors.ENDC)
            gen_network_stats(arch_name, nn_layer_names, dtf_name, output_path, network_name)

    # print(utils.bcolors.WARNING + f'********** Comparing dtfs ***********'+ utils.bcolors.ENDC)
    # for arch_name in arch_names:
    #     print(utils.bcolors.WARNING + f'{arch_name}' + utils.bcolors.ENDC)
    #     compare_dtf(arch_name, network_name, dtf_names, nn_layer_names, output_path)
    
    print(utils.bcolors.OKGREEN + f'********** Comparing archs ***********'+ utils.bcolors.ENDC)
    ideal_ = []
    stall_ = []
    i_bw_ = []
    w_bw_ = []
    o_bw_ = []
    for i in range(len(arch_set_names_flat)):
        print(utils.bcolors.OKGREEN + f'{arch_set_names_flat[i]}' + utils.bcolors.ENDC)
        ideal_arr, stall_arr, i_bw, w_bw, o_bw = compare_arch_set(arch_set_names_flat[i], arch_names[i], network_name, dtf_name, output_path)
        ideal_.append(ideal_arr)
        stall_.append(stall_arr)
        i_bw_.append(i_bw)
        w_bw_.append(w_bw)
        o_bw_.append(o_bw)

    print(utils.bcolors.OKCYAN + f'********** Comparing all archs ***********'+ utils.bcolors.ENDC)
    compare_all_arch_sets(arch_set_names_flat, arch_names_flat, ideal_, stall_, i_bw_, w_bw_, o_bw_)


