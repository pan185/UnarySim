import trace_gen
import argparse
import os
import pathlib
import utils
import subprocess

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
                        '--arch_path',
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


def parse_names_vec(arch_path, nn_path, dtf_path):
    arch_names = utils.parse_yaml(arch_path)
    nn_layer_names = utils.parse_yaml(nn_path)
    dtf_names = utils.parse_yaml(dtf_path)
    return arch_names, nn_layer_names, dtf_names

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

    

def compare_dtf(): 
    """
    This function reads two dataflow stats files across A different archs across L layers and also total network
    """
    pass

def compare_arch(): 
    """
    This function compares 2 arch sta
    """
    pass

if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    nn_path = pathlib.Path(args.nn_path).resolve()
    arch_path = pathlib.Path(args.arch_path).resolve()
    dtf_path = pathlib.Path(args.dtf_path).resolve()
    output_path = args.output_dir
    network_name = args.nn_path.split('workloads/')[1].split('_graph')[0]

    arch_names, nn_layer_names, dtf_names = parse_names_vec(arch_path, nn_path, dtf_path)

    if args.no_update == False: # bypassing trace gen
        gen_all(arch_names, nn_layer_names, dtf_names, output_path, network_name)

    print(utils.bcolors.HEADER + f'********** Generating network stats ***********'+ utils.bcolors.ENDC)
    for arch_name in arch_names:
        for dtf_name in dtf_names:
            print(utils.bcolors.HEADER + f'{arch_name}/{dtf_name}' + utils.bcolors.ENDC)
            gen_network_stats(arch_name, nn_layer_names, dtf_name, output_path, network_name)


