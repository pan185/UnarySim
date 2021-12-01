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
import logging
import block_trace as bt
import contention_processing as cp

_TRANCEGEN_DIR = os.environ['TRANCEGEN_DIR']
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # capture everything

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

    return parser

def cg_profile(prob, arch, dtf, output_dir, nn_name):
    """ Coarse grained profiling and trace gen
        Default assuming naive output stationary dataflow.

    Args:
        prob: An object defines the layer dimension.
        arch: An object defines the hardware architecture dimension.
        dtf: An object defines the dataflow mapping.
        output_dir: Path to output log.
    
    Return:
        output_dir: output path to trace files
        latency: coarse grained latency
        utilization: hardware utilization under the defined dataflow
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
    total_compute = spatially_remapping * arch.arithmetic['dimI'] * arch.arithmetic['dimW']
    utilized_compute = pqn * K
    utilization = utilized_compute / total_compute * 100
    logger.debug(f"latency = {latency}, util={utilization}")

    # parse coarse grained stats to json file
    output_base = pathlib.Path(output_path).resolve()
    output_dir = output_base / arch.config_str() / dtf.config_str() / nn_name / prob.config_str()
    # print(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # prefix = 'ideal_stats'
    # json_file = output_dir / f"{prefix}.json"
    # status_dict = dict()
    # status_dict['pe_cycle'] = latency
    # status_dict['utilization'] = utilization
    # utils.store_json(json_file, status_dict, indent=4)

    #****************gen trace*************
    # output file path
    sram_read_trace_file_input = output_dir / 'sram_read_input.csv'
    rd_outfile_input = open(sram_read_trace_file_input, 'w')
    sram_read_trace_file_weight = output_dir / 'sram_read_weight.csv'
    rd_outfile_weight = open(sram_read_trace_file_weight, 'w')
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

    # address base
    # Note: address/trace is produced in block granularity to decouple trace gen and processing
    output_base = arch.storage[arch.mem_idx['OutputBuffer']]['base']
    wght_base = arch.storage[arch.mem_idx['WeightBuffer']]['base']
    input_base = arch.storage[arch.mem_idx['InputBuffer']]['base']

    # initialize checkpoint values
    cp_n = 0; cp_p = 0; cp_q = 0; i = 0;
    cp_k = 0 

    cycle = 0
    i_pass = 0 # index of passes
    iter_per_pass = R*S*C
    logger.debug(f'input tiling={tiling_factors[1]}, wght tiling={tiling_factors[0]}, RSC={iter_per_pass}')

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
                    rd_outfile_weight.write(wght_rd)

                    # ***************load input***************
                    input_addr = []
                    # image to column address indexing
                    input_layout = arch.storage[arch.mem_idx['InputBuffer']]['layout']
                    # print(f'Debugging input layout: {input_layout}')

                    i = 0
                    while i < hw_i: # Spatially load across hw dimension
                        if i == 0:_p = cp_p; _q = cp_q; _n = cp_n;
                        if _p > P: break
                        if _q > Q: break
                        if _n > N: break
                        new_addr = input_base + utils.im2col_addr(
                            input_layout=input_layout,
                            patch_P=_p, patch_Q=_q, patch_N=_n, pixel=i_pass, pad=PAD, R=R, S=S, C=C, N=N,
                            Wdilation=Wdilation, Hdilation=Hdilation, Wstride=Wstride, Hstride=Hstride, W=W, H=H)
                        nqp_set.add((_n,_q,_p))
                        i += 1
                        input_addr.append(new_addr)

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

                    # print(utils.bcolors.OKCYAN + f'{input_addr}' + utils.bcolors.ENDC)
                    input_rd = f'{cycle},' + utils.list_to_comma_separated_str_with_padding(input_addr, hw_i)
                    rd_outfile_input.write(input_rd)
                    
                    cycle += single_pass_latency
                # print(f'Debugging sets: \nk={k_set}\nnqp={nqp_set}')
                # logger.debug(f'Debugging sets: \nk={k_set}\nnqp={nqp_set}')
                # end for for 1 pass (R*S*C)

                # ***************write output***************
                output_addr = []
                output_layout = arch.storage[arch.mem_idx['OutputBuffer']]['layout']
                # print(f'Debugging output layout: {output_layout}')
                for nqp_tuple in nqp_set:
                    _n_tuple = nqp_tuple[0]; _q_tuple = nqp_tuple[1]; _p_tuple = nqp_tuple[2]
                    for _k_tuple in k_set:
                        if output_layout == 'NKPQ': o_addr = output_base + _n_tuple * K*P*Q + _k_tuple * P*Q + _p_tuple * Q + _q_tuple
                        elif output_layout == 'NKQP': o_addr = output_base + _n_tuple * K*P*Q + _k_tuple * P*Q + _q_tuple * P + _p_tuple
                        elif output_layout == 'NQPK': o_addr = output_base + _n_tuple * K*P*Q + _q_tuple * P*K + _p_tuple * K + _k_tuple
                        elif output_layout == 'NPQK': o_addr = output_base + _n_tuple * K*P*Q + _p_tuple * Q*K + _q_tuple * K + _k_tuple
                        else: print(f'Does not support output memory layout of {output_layout}'); exit()
                        output_addr.append(o_addr)
                # print(output_addr)
                output_wr = f'{cycle},' + utils.list_to_comma_separated_str_with_padding(output_addr, hw_i*hw_w)
                wr_outfile.write(output_wr)

                # update checkpoint values for input
                cp_p = _p; cp_n = _n; cp_q = _q

            # end for for 1 inner streaming tile; By default a input tile

            # update checkpoint values for weight
            cp_k += _k
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
                    rd_outfile_weight.write(wght_rd)

                    # ***************load input***************
                    input_addr = []
                    # image to column address indexing
                    input_layout = arch.storage[arch.mem_idx['InputBuffer']]['layout']
                    # print(f'Debugging input layout: {input_layout}')

                    i = 0
                    while i < hw_i: # Spatially load across hw dimension
                        if i == 0:_p = cp_p; _q = cp_q; _n = cp_n;
                        if _p > P: break
                        if _q > Q: break
                        if _n > N: break
                        new_addr = input_base + utils.im2col_addr(
                            input_layout=input_layout,
                            patch_P=_p, patch_Q=_q, patch_N=_n, pixel=i_pass, pad=PAD, R=R, S=S, C=C, N=N,
                            Wdilation=Wdilation, Hdilation=Hdilation, Wstride=Wstride, Hstride=Hstride, W=W, H=H)
                        nqp_set.add((_n,_q,_p))
                        i += 1
                        input_addr.append(new_addr)

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

                    # print(utils.bcolors.OKCYAN + f'{input_addr}' + utils.bcolors.ENDC)
                    input_rd = f'{cycle},' + utils.list_to_comma_separated_str_with_padding(input_addr, hw_i)
                    rd_outfile_input.write(input_rd)
                    
                    cycle += single_pass_latency
                # print(f'Debugging sets: \nk={k_set}\nnqp={nqp_set}')
                # logger.debug(f'Debugging sets: \nk={k_set}\nnqp={nqp_set}')
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
                cp_k += _k

            # end for for 1 inner streaming tile; In this case a weight tile

            # update checkpoint values for input
            cp_p = _p; cp_n = _n; cp_q = _q
            cp_k = 0

    return output_dir, latency, utilization

    

def run_trace_gen(prob_path, arch_path, dtf_path, output_path, nn_name):
    prob = Prob(prob_path)
    arch = Arch(arch_path)
    dtf = Dataflow(dtf_path)

    # print("Debugging")
    # prob.print()
    # arch.print()
    # dtf.print()

    out_dir, cg_lat, cg_util = cg_profile(prob, arch, dtf, output_path, nn_name)
    cp.profile(prob, arch, dtf, output_path, out_dir, cg_lat, cg_util)
    # return out_dir, cg_lat, cg_util
    
    


if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    # Setup logger
    module_name = pathlib.Path(__file__).stem
    utils.setup_logging(module_name, logger)

    prob_path = pathlib.Path(args.prob_path).resolve()
    arch_path = pathlib.Path(args.arch_path).resolve()
    dtf_path = pathlib.Path(args.dtf_path).resolve()
    output_path = args.output_dir

    nn_name = args.prob_path.split('workloads/')[1].split('_graph')[0]

    # out_dir, cg_lat, cg_uitl = 
    run_trace_gen(prob_path=prob_path, arch_path=arch_path, dtf_path=dtf_path, output_path=output_path, nn_name=nn_name)

