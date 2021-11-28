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
from tracegen_parse import Dataflow, Prob, Arch
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

    parser.add_argument('-dp',
                        '--dtf_path',
                        type=str,
                        help='Datfflow Path',
                        default=f'{_TRANCEGEN_DIR}/configs/dataflow/os_w_sta.yaml',
                        )

    parser.add_argument('--dataflow',
                    choices=['os', 'ws', 'is'],
                    help='Dataflow Config',
                    default='os'
                    )

    return parser

def cg_profile(prob, arch, dtf, output_dir):
    """ Coarse grained profiling. 
        Default assuming naive output stationary dataflow.

    Args:
        prob: An object defines the layer dimension.
        arch: An object defines the hardware architecture dimension.
        dtf: An object defines the dataflow mapping.
        output_dir: Path to output log.
    """
    #FIXME: only OS is supported for now
    assert(dtf.type=='OutputStationary')

    # tiling_factors: (weight, input) dimension tiling consts
    pqn = prob.prob_bound[prob.prob_name_idx_dict['P']]*prob.prob_bound[prob.prob_name_idx_dict['Q']]*prob.prob_bound[prob.prob_name_idx_dict['N']]
    K = prob.prob_bound[prob.prob_name_idx_dict['K']]
    tiling_factors = (math.ceil(K/arch.arithmetic['dimW']), 
        math.ceil(pqn/arch.arithmetic['dimI']))
    spatially_remapping = tiling_factors[0] * tiling_factors[1]
    temporal_mapping_times = spatially_remapping * prob.prob_bound[prob.prob_name_idx_dict['R']]*prob.prob_bound[prob.prob_name_idx_dict['S']]*prob.prob_bound[prob.prob_name_idx_dict['C']]

    # parse early termination 
    et = prob.prob.get('et_cycle', None)
    if et == None: single_pass_latency = arch.arithmetic['bwI']**2 # if no et cycle specificied, assume no et
    else: single_pass_latency = et
    latency = temporal_mapping_times * single_pass_latency
    print("latency = ", latency)
    total_compute = spatially_remapping * arch.arithmetic['dimI'] * arch.arithmetic['dimW']
    utilized_compute = pqn * K
    utilization = utilized_compute / total_compute * 100

    # parse coarse grained stats to json file
    output_base = pathlib.Path(output_path).resolve()
    output_dir = output_base / arch.config_str() / prob.config_str() / dtf.config_str()
    # print(output_dir)
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

    # Parse arch and prob dimensions
    hw_w = arch.arithmetic['dimW']
    hw_i = arch.arithmetic['dimI']
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

    # FIXME: fix memory granularity
    # address base (byte addressable) Note: different from uSystolic
    output_base=2000000 # output feature map base addr, in byte
    wght_base=1000000 # weight base addr, in byte
    input_base=0 # input feature map base addr, in byte

    # initialize checkpoint values
    cp_n = 0; cp_p = 0; cp_q = 0; i = 0;
    cp_k = 0 

    cycle = 0
    i_pass = 0 # index of passes
    iter_per_pass = R*S*C
    print(f'input tiling={tiling_factors[1]}, wght tiling={tiling_factors[0]}, RSC={iter_per_pass}')

    if dtf.tileStationary == 'W':
        # 2-layer for loop for IW tiles
        for w_tile in range(tiling_factors[0]):
            for i_tile in range(tiling_factors[1]):
                k_set = set()
                nqp_set = set()

                for i_pass in range(iter_per_pass):
                    # ***************load wght***************
                    wght_addr = []
                    _k =  0
                    while _k < hw_w:
                        wght_addr.append(wght_base + (cp_k + _k)*R*S*C + i_pass)
                        k_set.add(cp_k +_k)
                        _k += 1
                        if _k + cp_k >= K: break # Note: Unoptimized mapping, no packing for underutilized w vector

                    wght_rd = f'{cycle},' + utils.list_to_comma_separated_str_with_padding(wght_addr, hw_w)
                    rd_outfile.write(wght_rd)

                    # ***************load input***************
                    input_addr = []
                    # image to column address indexing
                    input_layout = arch.storage[arch.mem_idx['InputBuffer']]['layout']
                    # print(f'Debugging input layout: {input_layout}')

                    i = 0
                    while i < hw_i: # Spatially load across hw dimension
                        if i == 0:_p = cp_p; _q = cp_q; _n = cp_n;
                        if _p * _q * _n > P*Q*N: break
                        new_addr = input_base + utils.im2col_addr(
                            input_layout=input_layout,
                            patch_P=_p, patch_Q=_q, patch_N=_n, pixel=i_pass, pad=PAD, R=R, S=S, C=C, N=N,
                            Wdilation=Wdilation, Hdilation=Hdilation, Wstride=Wstride, Hstride=Hstride, W=W, H=H)
                        nqp_set.add((_n,_q,_p))
                        i += 1

                        # update _p, _q, _n (roll over if necessary)
                        _p += 1
                        if _p == P: 
                            _q += 1
                            _p = 0
                        if _q == Q:
                            _n += 1
                            _q = 0
                        if _n == N:
                            break

                        input_addr.append(new_addr)
                    input_rd = f'{cycle},' + utils.list_to_comma_separated_str_with_padding(input_addr, hw_i)
                    rd_outfile.write(input_rd)
                    
                    cycle += single_pass_latency
                print(f'Debugging sets: \nk={k_set}\nnqp={nqp_set}')
                # end for for 1 pass (R*S*C)

                # ***************write output***************
                output_addr = []
                output_layout = arch.storage[arch.mem_idx['OutputBuffer']]['layout']
                # print(f'Debugging output layout: {output_layout}')
                for nqp_tuple in nqp_set:
                    _n_tuple = nqp_tuple[0]; _q_tuple = nqp_tuple[1]; _p_tuple = nqp_tuple[2]
                    for _k_tuple in k_set:
                        if output_layout == 'NKPQ': output_addr.append(output_base + _n_tuple * K*P*Q + _k_tuple * P*Q + _p_tuple * Q + _q_tuple)
                        elif output_layout == 'NKQP': output_addr.append(output_base + _n_tuple * K*P*Q + _k_tuple * P*Q + _q_tuple * P + _p_tuple)
                        elif output_layout == 'NQPK': output_addr.append(output_base + _n_tuple * K*P*Q + _q_tuple * P*K + _p_tuple * K + _k_tuple)
                        elif output_layout == 'NPQK': output_addr.append(output_base + _n_tuple * K*P*Q + _p_tuple * Q*K + _q_tuple * K + _k_tuple)
                        else: print(f'Does not support output memory layout of {output_layout}'); exit()
                output_wr = f'{cycle},' + utils.list_to_comma_separated_str_with_padding(output_addr, hw_i*hw_w)
                wr_outfile.write(output_wr)

                # update checkpoint values for input
                cp_p = _p; cp_n = _n; cp_q = _q

            # end for for 1 inner streaming tile; By default a input tile

            # update checkpoint values for weight
            cp_k = _k
            cp_p = 0; cp_n = 0; cp_q = 0

    else: # I stationary streaming
        # 2-layer for loop for IW tiles
        for i_tile in range(tiling_factors[1]):
            for w_tile in range(tiling_factors[0]):
                k_set = set()
                nqp_set = set()

                for i_pass in range(iter_per_pass):
                    # ***************load wght***************
                    wght_addr = []
                    _k =  0
                    while _k < hw_w:
                        wght_addr.append(wght_base + (cp_k + _k)*R*S*C + i_pass)
                        k_set.add(cp_k +_k)
                        _k += 1
                        if _k + cp_k >= K: break # Note: Unoptimized mapping, no packing for underutilized w vector

                    wght_rd = f'{cycle},' + utils.list_to_comma_separated_str_with_padding(wght_addr, hw_w)
                    rd_outfile.write(wght_rd)

                    # ***************load input***************
                    input_addr = []
                    # image to column address indexing
                    input_layout = arch.storage[arch.mem_idx['InputBuffer']]['layout']
                    # print(f'Debugging input layout: {input_layout}')

                    i = 0
                    while i < hw_i: # Spatially load across hw dimension
                        if i == 0:_p = cp_p; _q = cp_q; _n = cp_n;
                        if _p * _q * _n > P*Q*N: break
                        new_addr = input_base + utils.im2col_addr(
                            input_layout=input_layout,
                            patch_P=_p, patch_Q=_q, patch_N=_n, pixel=i_pass, pad=PAD, R=R, S=S, C=C, N=N,
                            Wdilation=Wdilation, Hdilation=Hdilation, Wstride=Wstride, Hstride=Hstride, W=W, H=H)
                        nqp_set.add((_n,_q,_p))
                        i += 1

                        # update _p, _q, _n (roll over if necessary)
                        _p += 1
                        if _p == P: 
                            _q += 1
                            _p = 0
                        if _q == Q:
                            _n += 1
                            _q = 0
                        if _n == N:
                            break

                        input_addr.append(new_addr)
                    input_rd = f'{cycle},' + utils.list_to_comma_separated_str_with_padding(input_addr, hw_i)
                    rd_outfile.write(input_rd)
                    
                    cycle += single_pass_latency
                print(f'Debugging sets: \nk={k_set}\nnqp={nqp_set}')
                # end for for 1 pass (R*S*C)

                # ***************write output***************
                output_addr = []
                output_layout = arch.storage[arch.mem_idx['OutputBuffer']]['layout']
                # print(f'Debugging output layout: {output_layout}')
                for nqp_tuple in nqp_set:
                    _n_tuple = nqp_tuple[0]; _q_tuple = nqp_tuple[1]; _p_tuple = nqp_tuple[2]
                    for _k_tuple in k_set:
                        if output_layout == 'NKPQ': output_addr.append(output_base + _n_tuple * K*P*Q + _k_tuple * P*Q + _p_tuple * Q + _q_tuple)
                        elif output_layout == 'NKQP': output_addr.append(output_base + _n_tuple * K*P*Q + _k_tuple * P*Q + _q_tuple * P + _p_tuple)
                        elif output_layout == 'NQPK': output_addr.append(output_base + _n_tuple * K*P*Q + _q_tuple * P*K + _p_tuple * K + _k_tuple)
                        elif output_layout == 'NPQK': output_addr.append(output_base + _n_tuple * K*P*Q + _p_tuple * Q*K + _q_tuple * K + _k_tuple)
                        else: print(f'Does not support output memory layout of {output_layout}'); exit()
                output_wr = f'{cycle},' + utils.list_to_comma_separated_str_with_padding(output_addr, hw_i*hw_w)
                wr_outfile.write(output_wr)

                # update checkpoint values for weight
                cp_k = _k

            # end for for 1 inner streaming tile; In this case a weight tile

            # update checkpoint values for input
            cp_p = _p; cp_n = _n; cp_q = _q
            cp_k = 0

    

def run_trace_gen(prob_path, arch_path, dtf_path, output_path):
    prob = Prob(prob_path)
    arch = Arch(arch_path)
    dtf = Dataflow(dtf_path)

    print("Debugging")
    prob.print()
    arch.print()
    dtf.print()

    cg_profile(prob, arch, dtf, output_path)
    # gen(prob, arch, output_path, dataflow)


if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    prob_path = pathlib.Path(args.prob_path).resolve()
    arch_path = pathlib.Path(args.arch_path).resolve()
    dtf_path = pathlib.Path(args.dtf_path).resolve()
    output_path = args.output_dir

    run_trace_gen(prob_path=prob_path, arch_path=arch_path, dtf_path=dtf_path, output_path=output_path)
