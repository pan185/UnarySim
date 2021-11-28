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

def run_all(arch_path, nn_path, dtf_path, output_path):
    arch_names = utils.parse_yaml(arch_path)
    nn_layer_names = utils.parse_yaml(nn_path)
    dtf_names = utils.parse_yaml(dtf_path)
    # print(nn_layer_names, dtf_names)
    for arch in arch_names:
        print(utils.bcolors.OKGREEN + f'Processing {arch}' + utils.bcolors.ENDC)
        arch += '.yml'
        for layer in nn_layer_names:
            print(utils.bcolors.OKCYAN+ f'  Processing {layer}' + utils.bcolors.ENDC)
            layer += '.yaml'
            for dtf in dtf_names:
                print(utils.bcolors.OKBLUE +f'      Using {dtf} dataflow' + utils.bcolors.ENDC)
                dtf += '.yaml'
                in_arr = ['python3', f'{_TRANCEGEN_DIR}/trace_gen.py', 
                    '-pp', f'{_TRANCEGEN_DIR}/configs/workloads/convnet_graph/'+layer, 
                    '-ap', f'{_TRANCEGEN_DIR}/configs/arch/'+arch,
                    '-dp', f'{_TRANCEGEN_DIR}/configs/dataflow/'+dtf, 
                    '-o', output_path]
                # print(in_arr)
                p = subprocess.Popen(in_arr)
                output, error = p.communicate()
                if output != None: print(output)



if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    nn_path = pathlib.Path(args.nn_path).resolve()
    arch_path = pathlib.Path(args.arch_path).resolve()
    # arch_path = args.arch_path
    dtf_path = pathlib.Path(args.dtf_path).resolve()
    output_path = args.output_dir

    run_all(arch_path, nn_path, dtf_path, output_path)


