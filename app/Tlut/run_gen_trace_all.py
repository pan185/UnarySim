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

    return parser

def gen_all(arch_names, nn_layer_names, dtf_names, output_path):
    
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
                    '-pp', f'{_TRANCEGEN_DIR}/configs/workloads/convnet_graph/'+layer_n, 
                    '-ap', f'{_TRANCEGEN_DIR}/configs/arch/'+arch_n,
                    '-dp', f'{_TRANCEGEN_DIR}/configs/dataflow/'+dtf_n, 
                    '-o', output_path]
                # print(in_arr)
                p = subprocess.Popen(in_arr)
                output, error = p.communicate()
                if output != None: print(output)

                # # parsing output stats
                # data = utils.parse_json(output_path + '/' + arch + '/' + layer + '/' + dtf + '/stats.json')
                # print(data)

def parse_names_vec(arch_path, nn_path, dtf_path):
    arch_names = utils.parse_yaml(arch_path)
    nn_layer_names = utils.parse_yaml(nn_path)
    dtf_names = utils.parse_yaml(dtf_path)
    return arch_names, nn_layer_names, dtf_names

def gen_network_stats():
    """
    This function generates stats for 1 arch and 1 dtf
    """
    pass

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

    arch_names, nn_layer_names, dtf_names = parse_names_vec(arch_path, nn_path, dtf_path)
    gen_all(arch_names, nn_layer_names, dtf_names, output_path)


