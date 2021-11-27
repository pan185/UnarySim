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
import numpy as np
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
                        default=f'{_TRANCEGEN_DIR}/configs/workloads/convnet_graph/_conv1.yaml',
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
    K = prob.prob_bound[prob.prob_name_idx_dict['K']]
    tiling_factors = (math.ceil(K/arch.arithmetic['dimC']), 
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
    utilized_compute = pqn * K
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
    # output file path
    sram_read_trace_file = output_dir / 'sram_read.csv'
    rd_outfile = open(sram_read_trace_file, 'w')
    sram_write_trace_file = output_dir / 'sram_write.csv'
    wr_outfile = open(sram_write_trace_file, 'w')

    hw_w = arch.arithmetic['dimC']
    hw_i = arch.arithmetic['dimA']
    R = prob.prob_bound[prob.prob_name_idx_dict['R']]
    S = prob.prob_bound[prob.prob_name_idx_dict['S']]
    C = prob.prob_bound[prob.prob_name_idx_dict['C']]
    N = prob.prob_bound[prob.prob_name_idx_dict['N']]
    P = prob.prob_bound[prob.prob_name_idx_dict['P']]
    Q = prob.prob_bound[prob.prob_name_idx_dict['Q']]
    Wstride = prob.prob['Wstride']
    Hstride = prob.prob['Hstride']
    Wdilation = prob.prob['Wdilation']
    Hdilation = prob.prob['Hdilation']
    PAD = 0
    WGHT = K * R * S * C
    W = (P-1)*Wstride + R + 2 * PAD 
    H = (Q-1)*Hstride + S + 2 * PAD 
    INP = N * P * Q

    # address base (byte addressable) Note: different from uSystolic
    output_base=2000000 # output feature map base addr, in byte
    wght_base=1000000 # weight base addr, in byte
    input_base=0 # input feature map base addr, in byte

    cycle = 0
    i_pass = 0 # index of passes

    # load wght
    wght_addr = []
    if hw_w < WGHT: w_bound = hw_w
    else: w_bound = WGHT
    for i in range(w_bound):
        wght_addr.append(wght_base + i*R*S*C) # TODO: fix flex weight memory layout
    wght_rd = f'{cycle},' + utils.list_to_comma_separated_str_with_padding(wght_addr, hw_w)
    rd_outfile.write(wght_rd)

    # load input
    input_addr = []
    if hw_i < INP: 
        i_bound = hw_i
        if hw_i < P:
            p_bound = hw_i
            q_bound = 1
            n_bound = 1
        else: 
            p_bound = P
            if hw_i < P*Q:
                q_bound = int(hw_i/P)
                n_bound = 1
            else: 
                q_bound = Q
                n_bound = int(hw_i/(P*Q))
    else: 
        i_bound = INP
        p_bound = P # p subbound
        q_bound = Q # q subbound
        n_bound = N # n subbound
    print(f'Debugging input bounds\np:{p_bound}, q:{q_bound}, n:{n_bound}')
    # image to column address indexing
    input_layout = arch.storage[arch.mem_idx['InputBuffer']]['layout']
    print(f'Debugging input layout: {input_layout}')
    input_addr = []
    for p in range(p_bound):
        for q in range(q_bound):
            for n in range(n_bound):
                new_addr = input_base + utils.im2col_addr( #TODO: fix flex input memory layout
                    input_layout=input_layout,
                    patch_P=p, patch_Q=q, patch_N=n, pixel=i_pass, pad=PAD, R=R, S=S, C=C, N=N,
                    Wdilation=Wdilation, Hdilation=Hdilation, Wstride=Wstride, Hstride=Hstride)
                input_addr.append(new_addr)
    input_rd = f'{cycle},' + utils.list_to_comma_separated_str_with_padding(input_addr, hw_i)
    rd_outfile.write(input_rd)


def run_trace_gen(prob_path, arch_path, output_path, dataflow):
    prob = Prob(prob_path)
    arch = Arch(arch_path)

    print("Debugging")
    prob.print()
    arch.print()

    cg_profile(prob, arch, output_path, dataflow)
    # gen(prob, arch, output_path, dataflow)


if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    prob_path = pathlib.Path(args.prob_path).resolve()
    arch_path = pathlib.Path(args.arch_path).resolve()
    output_path = args.output_dir

    run_trace_gen(prob_path=prob_path, arch_path=arch_path, output_path=output_path, dataflow=args.dataflow)
