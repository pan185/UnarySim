import utils
import subprocess
import os.path
from pathlib import Path
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
from utils import cor
from tqdm import tqdm

_TRANCEGEN_DIR = os.environ['TRANCEGEN_DIR']

def construct_argparser():
    parser = argparse.ArgumentParser(description='Run Configuration')
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        help='Output Folder',
                        default=f'{_TRANCEGEN_DIR}/output_dir',
                        )
    parser.add_argument('-pp',
                        '--proj_top_level_file',
                        type=str,
                        help='Hardware Architecture Projection Def file Path',
                        default=f'{_TRANCEGEN_DIR}/configs/projection/sram_sensitivity.yml',
                        )
    parser.add_argument('--plot_only',
                        action='store_true',
                        help='Plot only',
                        default=False,
                        )

    return parser

def set_no_update(output_path, arch_name, dtf_name, network_name):
    # if first arch and first dataflow exist, no update
    path = output_path + f'/{arch_name}/{dtf_name}/{network_name}'
    p = Path(path)
    if p.exists() and p.is_dir():
        print(f'checking {path} exits!')
        return True
    else: return False

def project_all(arch_proj_top_level_path, dtf_top_level_names, dtf_names, output_path, network_names):
    arch_names = utils.parse_yaml(arch_proj_top_level_path)
    arch_proj_top_level_name = arch_proj_top_level_path.split('arch/')[1].split('.yml')[0]
    no_update = True
    
    for dtf in dtf_names:
        for network_name in network_names:
            for arch in arch_names:
                print(utils.bcolors.WARNING +f'checking {dtf} dataflow, {arch} arch, {network_name} network' + utils.bcolors.ENDC)
                no_update = no_update and set_no_update(output_path, arch, dtf, network_name)
    if no_update: print(utils.bcolors.WARNING + f'Skipping update!' + utils.bcolors.ENDC)
    else: print(utils.bcolors.FAIL + f'Needs updating. Regenerating all...' + utils.bcolors.ENDC)

    for dtf_top_level_name in dtf_top_level_names:
        for network_name in network_names:
            # version1: all layers
            in_arr = ['python3', f'{_TRANCEGEN_DIR}/tlut_systolic_perf_projection.py', 
                '-nn', f'{_TRANCEGEN_DIR}/configs/workloads/{network_name}_graph/layers.yaml', 
                '-ap', f'{_TRANCEGEN_DIR}/configs/arch/'+ arch_proj_top_level_name + '.yml',
                '-dtfs', f'{_TRANCEGEN_DIR}/configs/dataflow/'+dtf_top_level_name + '.yaml', 
                '-o', output_path]
            if no_update: in_arr.append('--no_update')
            # print(in_arr); exit()
            p = subprocess.Popen(in_arr)
            output, error = p.communicate()
            if output != None: print(output)

            # version2: conv only
            in_arr.append('--conv_only')
            in_arr.append('--no_update') # This absolutely does not need update
            p = subprocess.Popen(in_arr)
            output, error = p.communicate()
            if output != None: print(output)

def plot_percentage(filepath, arch_names, output_path, block):
    data = utils.parse_json(filepath)
    bsys_m_32_lat = data[f'8_{block}']['bsys']['lat']
    bsys_m_32_bw = data[f'8_{block}']['bsys']['bw']
    bsys_s_32_lat = data[f'4_{block}']['bsys']['lat']
    bsys_s_32_bw = data[f'4_{block}']['bsys']['bw']
    bsys_l_32_lat = data[f'16_{block}']['bsys']['lat']
    bsys_l_32_bw = data[f'16_{block}']['bsys']['bw']

    usys_m_32_lat = data[f'8_{block}']['usys']['lat']
    usys_m_32_bw = data[f'8_{block}']['usys']['bw']
    usys_s_32_lat = data[f'4_{block}']['usys']['lat']
    usys_s_32_bw = data[f'4_{block}']['usys']['bw']
    usys_l_32_lat = data[f'16_{block}']['usys']['lat']
    usys_l_32_bw = data[f'16_{block}']['usys']['bw']

    usys_lat = [usys_l_32_lat, [float('NaN')], usys_m_32_lat, [float('NaN')], usys_s_32_lat, [float('NaN')]]
    usys_lat = sum(usys_lat, [])
    usys_bw = [usys_l_32_bw, [float('NaN')], usys_m_32_bw, [float('NaN')], usys_s_32_bw, [float('NaN')]]
    usys_bw = sum(usys_bw, [])
    bsys_lat = [bsys_l_32_lat, [float('NaN')], bsys_m_32_lat, [float('NaN')], bsys_s_32_lat, [float('NaN')]]
    bsys_lat = sum(bsys_lat, [])
    bsys_bw = [bsys_l_32_bw, [float('NaN')], bsys_m_32_bw, [float('NaN')], bsys_s_32_bw, [float('NaN')]]
    bsys_bw = sum(bsys_bw, [])

    # TODO: Make pretty
    font = {'family':'Times New Roman', 'size': 5}
    matplotlib.rc('font', **font)
    my_dpi = 300
    fig_h = 1.2
    fig_w = 3.3115

    arch_names = arch_names+['']
    x_axis = arch_names*3
    x_idx = np.arange(len(x_axis))

    width = 0.2

    fig, rt_ax = plt.subplots(figsize=(fig_w, fig_h))
    
    ncol = 2
    rt_ax.plot(x_idx, usys_lat, '-s', color=cor.tlut_blue, ms=4, label='Unary systolic')
    rt_ax.plot(x_idx, bsys_lat, '-o', color=cor.tlut_nude, ms=4, label='Binary systolic')
    rt_ax.axhline(y=1, color='k', linestyle='--',linewidth=0.5)
    
    rt_ax.set_ylabel('Normalized latency')
    rt_ax.minorticks_off()

    bars, labels = rt_ax.get_legend_handles_labels()
    
    plt.xlim(x_idx[0]-0.5, x_idx[-1]+0.5)
    rt_ax.set_xticks(x_idx)
    rt_ax.set_xticklabels(x_axis, rotation = 45)
    plt.yscale("linear")
    rt_ax.legend(bars, labels, loc="upper center", ncol=ncol, frameon=True)

    xticks = rt_ax.xaxis.get_major_ticks()
    for i in range(len(x_axis)):
        if i % (len(arch_names)-1) == 0:
            xticks[i].label1.set_visible(False)
    
    print("rt_ax ylim: ", rt_ax.get_ylim())

    # Fine tuning limit and ticks
    max_val = max(max(usys_lat), max(bsys_lat))
    rt_ax.set_ylim((0, 1.1*max_val))
    # rt_ax.set_yticks((0, 1, 2, 3, 4))

    ax2 = rt_ax.twiny()
    plt.xlim(x_idx[0]-0.5, x_idx[-1]+0.5)
    x_idx2 = [1.5, 6.5, 11.5]
    ax2.set_xticks(x_idx2)
    ax2.set_xticklabels([f'16-bank, {block}B', f'8-bank, {block}B', f'4-bank, {block}B'])

    fig.tight_layout()
    id_str = filepath.split('projection/')[1].split('.json')[0]
    plt.savefig(output_path + f'/projection/{id_str}_lat' + '.pdf', bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)

    fig, bw_ax = plt.subplots(figsize=(fig_w, fig_h))
    
    ncol = 2
    bw_ax.plot(x_idx, usys_bw, '-s', color=cor.tlut_blue, ms=4, label='Unary systolic')
    bw_ax.plot(x_idx, bsys_bw, '-o', color=cor.tlut_nude, ms=4, label='Binary systolic')
    bw_ax.axhline(y=1, color='k', linestyle='--',linewidth=0.5)
    
    bw_ax.set_ylabel('Normalized bandwidth')
    bw_ax.minorticks_off()

    bars, labels = bw_ax.get_legend_handles_labels()
    
    plt.xlim(x_idx[0]-0.5, x_idx[-1]+0.5)
    bw_ax.set_xticks(x_idx)
    bw_ax.set_xticklabels(x_axis, rotation = 45)
    plt.yscale("linear")
    bw_ax.legend(bars, labels, loc="upper center", ncol=ncol, frameon=True)

    xticks = bw_ax.xaxis.get_major_ticks()
    for i in range(len(x_axis)):
        if i % (len(arch_names)-1) == 0:
            xticks[i].label1.set_visible(False)
    
    print("bw_ax ylim: ", bw_ax.get_ylim())

    # Fine tuning limit and ticks
    max_val = max(max(usys_bw), max(bsys_bw))
    bw_ax.set_ylim((0, 1.1*max_val))
    # rt_ax.set_yticks((0, 1, 2, 3, 4))

    ax2 = bw_ax.twiny()
    plt.xlim(x_idx[0]-0.5, x_idx[-1]+0.5)
    x_idx2 = [1.5, 6.5, 11.5]
    ax2.set_xticks(x_idx2)
    ax2.set_xticklabels([f'16-bank, {block}B', f'8-bank, {block}B', f'4-bank, {block}B'])

    fig.tight_layout()
    id_str = filepath.split('projection/')[1].split('.json')[0]
    plt.savefig(output_path + f'/projection/{id_str}_bw' + '.pdf', bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)


if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    projection_dict = utils.parse_yaml(args.proj_top_level_file)
    arch_proj_top_level_group_dicts = projection_dict['arch']

    for group_dict in arch_proj_top_level_group_dicts:
        arch_proj_top_level_names = group_dict['name']
        block = group_dict['block']

        # flatten dtf names
        dtf_names_hier = []
        dtf_top_level_names = projection_dict['dataflow']
        for i in dtf_top_level_names:
            dtf_names_hier.append(utils.parse_yaml(f'{_TRANCEGEN_DIR}/configs/dataflow/{i}.yaml'))
        dtf_names = sum(dtf_names_hier, [])
        # print(dtf_names); exit()

        network_names = projection_dict['workloads']
        output_path = args.output_dir

        pbar = tqdm(arch_proj_top_level_names)
        for name in pbar:
            arch_proj_top_level_path = f'{_TRANCEGEN_DIR}/configs/arch/'+ name + '.yml'
            if args.plot_only == False: project_all(arch_proj_top_level_path, dtf_top_level_names, dtf_names, output_path, network_names)
            pbar.set_description(f'Projecting {name}')

        # parsing for plotting
        for network_name in network_names:
            for dtf_name in dtf_names:
                # conv only
                projection_stats_file = utils.get_mem_sensitivity_stats_file_name(output_path, block, network_name, dtf_name, True)
                plot_percentage(filepath=projection_stats_file, 
                    arch_names=['16_16', '32_32', '64_64', '128_128'], 
                    output_path=output_path, block=block)
                
                # all layers
                projection_stats_file = utils.get_mem_sensitivity_stats_file_name(output_path, block, network_name, dtf_name, False)
                plot_percentage(filepath=projection_stats_file, 
                    arch_names=['16_16', '32_32', '64_64', '128_128'], 
                    output_path=output_path, block=block)