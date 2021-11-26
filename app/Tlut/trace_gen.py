"""
trace gen api
Compute: HW tlut kernel size: (T,A,C)
Memory: for each level
    - Name
    - Config
    - Num instance
Workload: 7-layer loop nest

"""
import argparse
import os
import pathlib
from tracegen_parse import Prob, Arch
import math
import utils
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
                        default=f'{_TRANCEGEN_DIR}/configs/arch/perfect_mem_128.yml',
                        )

    parser.add_argument('-pp',
                        '--prob_path',
                        type=str,
                        help='Problem Dimension Path',
                        default=f'{_TRANCEGEN_DIR}/configs/workloads/alexnet_graph/_outputs_input.2.yaml',
                        )

    parser.add_argument('--dataflow',
                    choices=['os', 'ws', 'is'],
                    help='Dataflow Config',
                    default='os'
                    )

    return parser

def cg_profile(prob, arch, output_dir, dataflow):
    """ Coarse grained profiling. 
        Default assuming naive output stationary dataflow.

    Args:
        prob: An object defines the layer dimension.
        arch: An object defines the hardware architecture dimension.
        output_dir: Path to output log.
        dataflow: dataflow configuration. 
    """
    #Note: only OS is supported for now
    assert(dataflow=='os')

    # tiling_factors: (input, weight) dimension tiling consts
    pqn = prob.prob_bound[prob.prob_name_idx_dict['P']]*prob.prob_bound[prob.prob_name_idx_dict['Q']]*prob.prob_bound[prob.prob_name_idx_dict['N']]
    k = prob.prob_bound[prob.prob_name_idx_dict['K']]
    tiling_factors = (math.ceil(k/arch.arithmetic['dimC']), 
        math.ceil(pqn/arch.arithmetic['dimA']))
    spatially_remapping = tiling_factors[0] * tiling_factors[1]
    temporal_mapping_times = spatially_remapping * prob.prob_bound[prob.prob_name_idx_dict['R']]*prob.prob_bound[prob.prob_name_idx_dict['S']]*prob.prob_bound[prob.prob_name_idx_dict['C']]

    # parse early termination 
    et = prob.prob.get('et_cycle', None)
    if et == None: single_pass_latency = arch.arithmetic['bwA']**2 # if no et cycle specificied, assume no et
    else: single_pass_latency = et
    latency = temporal_mapping_times * single_pass_latency
    print("latency = ", latency)
    total_compute = spatially_remapping * arch.arithmetic['dimA'] * arch.arithmetic['dimC']
    utilized_compute = pqn * k
    utilization = utilized_compute / total_compute * 100

    # parse coarse grained stats to json file
    output_base = pathlib.Path(output_path).resolve()
    output_dir = output_base / arch.config_str() / prob.config_str()
    print(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = 'stats'
    json_file = output_dir / f"{prefix}.json"
    status_dict = dict()
    status_dict['pe_cycle'] = latency
    status_dict['utilization'] = utilization
    utils.store_json(json_file, status_dict, indent=4)

    #****************gen trace*************
    hw_w = arch.arithmetic.dimC
    hw_i = arch.arithmetic.dimA

    # for n in range(prob.prob_bound[prob.prob_name_idx_dict['N']]):
    

def run_trace_gen(prob_path, arch_path, output_path, dataflow):
    prob = Prob(prob_path)
    arch = Arch(arch_path)

    print("Debugging")
    prob.print()
    arch.print()

    cg_profile(prob, arch, output_path, dataflow)
    gen(prob, arch, output_path, dataflow)


if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    prob_path = pathlib.Path(args.prob_path).resolve()
    arch_path = pathlib.Path(args.arch_path).resolve()
    output_path = args.output_dir

    run_trace_gen(prob_path=prob_path, arch_path=arch_path, output_path=output_path, dataflow=args.dataflow)
