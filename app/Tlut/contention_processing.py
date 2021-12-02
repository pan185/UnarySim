import math
import block_trace
import utils
import argparse
import os
import utils
import pathlib
import logging
_TRANCEGEN_DIR = os.environ['TRANCEGEN_DIR']

def construct_argparser():
    parser = argparse.ArgumentParser(description='Run Configuration')
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        help='Output Folder',
                        default=f'{_TRANCEGEN_DIR}/output_dir',
                        )
    # parser.add_argument('-ap',
    #                     '--arch_path',
    #                     type=str,
    #                     help='Hardware Architecture Path',
    #                     default=f'{_TRANCEGEN_DIR}/configs/arch/perfect_mem_128.yml',
    #                     )

    # parser.add_argument('-pp',
    #                     '--prob_path',
    #                     type=str,
    #                     help='Problem Dimension Path',
    #                     default=f'{_TRANCEGEN_DIR}/configs/workloads/convnet_graph/_conv1.yaml',
    #                     )

    # parser.add_argument('-dp',
    #                     '--dtf_path',
    #                     type=str,
    #                     help='Datfflow Path',
    #                     default=f'{_TRANCEGEN_DIR}/configs/dataflow/os_w_sta.yaml',
    #                     )

    return parser

def profile(prob, arch, dtf, output_dir, out_dir, cg_lat, cg_util):
    print(utils.bcolors.OKBLUE + f'Contention processing on {arch.config_str()}/{prob.config_str()}/{dtf.config_str()}' + utils.bcolors.ENDC)
    # *******************Contention processing*******************
    # Code adapted from uSystolic-sim profiling.py

    # process frequency TODO: take freq from arch.yml
    freq = 400 #MHz
    period = 1.0 / freq # us

    # input read
    word_sz_bytes_input_rd = float(arch.storage[arch.mem_idx['InputBuffer']]['bw']/8)
    tot_word_ifmap_rd_sram, \
    max_word_ifmap_rd_sram, \
    tot_access_ifmap_rd_sram, \
    max_access_ifmap_rd_sram, \
    act_cycles_ifmap_rd_sram, \
    stall_cycles_ifmap_rd_sram, \
    ideal_start_cycle_ifmap_rd_sram, \
    ideal_end_cycle_ifmap_rd_sram = block_trace.sram_profiling(
        trace_file=out_dir / 'sram_read_input.csv', 
        word_sz_bytes=word_sz_bytes_input_rd,
        block_sz_bytes=16,
        bank=16,
        min_addr_word=arch.storage[arch.mem_idx['InputBuffer']]['base'],
        max_addr_word=arch.storage[arch.mem_idx['InputBuffer']]['base'] + arch.storage[arch.mem_idx['InputBuffer']]['entries'],
        access_buf=True)
    real_start_cycle_ifmap_rd_sram = ideal_start_cycle_ifmap_rd_sram
    real_end_cycle_ifmap_rd_sram = ideal_end_cycle_ifmap_rd_sram + stall_cycles_ifmap_rd_sram

    # weight read
    word_sz_bytes_weight_rd = float(arch.storage[arch.mem_idx['WeightBuffer']]['bw']/8)
    tot_word_filter_rd_sram, \
    max_word_filter_rd_sram, \
    tot_access_filter_rd_sram, \
    max_access_filter_rd_sram, \
    act_cycles_filter_rd_sram, \
    stall_cycles_filter_rd_sram, \
    ideal_start_cycle_filter_rd_sram, \
    ideal_end_cycle_filter_rd_sram = block_trace.sram_profiling(
        trace_file=out_dir / 'sram_read_weight.csv', 
        word_sz_bytes=word_sz_bytes_weight_rd,
        block_sz_bytes=16,
        bank=16,
        min_addr_word=arch.storage[arch.mem_idx['WeightBuffer']]['base'],
        max_addr_word=arch.storage[arch.mem_idx['WeightBuffer']]['base'] + arch.storage[arch.mem_idx['WeightBuffer']]['entries'],
        access_buf=True)
    real_start_cycle_filter_rd_sram = ideal_start_cycle_filter_rd_sram
    real_end_cycle_filter_rd_sram = ideal_end_cycle_filter_rd_sram + stall_cycles_filter_rd_sram
    
    # Output read
    # FIXME: updata later

    # Output write
    word_sz_bytes_output_wr = float(arch.storage[arch.mem_idx['OutputBuffer']]['bw']/8)
    tot_word_ofmap_wr_sram, \
    max_word_ofmap_wr_sram, \
    tot_access_ofmap_wr_sram, \
    max_access_ofmap_wr_sram, \
    act_cycles_ofmap_wr_sram, \
    stall_cycles_ofmap_wr_sram, \
    ideal_start_cycle_ofmap_wr_sram, \
    ideal_end_cycle_ofmap_wr_sram = block_trace.sram_profiling(
        trace_file=out_dir / 'sram_write.csv', 
        word_sz_bytes=word_sz_bytes_output_wr,
        block_sz_bytes=16,
        bank=16,
        min_addr_word=arch.storage[arch.mem_idx['OutputBuffer']]['base'],
        max_addr_word=arch.storage[arch.mem_idx['OutputBuffer']]['base'] + arch.storage[arch.mem_idx['OutputBuffer']]['entries'],
        access_buf=True)
    real_start_cycle_ofmap_wr_sram = ideal_start_cycle_ofmap_wr_sram
    real_end_cycle_ofmap_wr_sram = ideal_end_cycle_ofmap_wr_sram + stall_cycles_ofmap_wr_sram

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # run time calculation
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # ideal
    ideal_max_clk = max(ideal_end_cycle_ifmap_rd_sram, 
                        ideal_end_cycle_filter_rd_sram, 
                        # ideal_end_cycle_ofmap_rd_sram, 
                        ideal_end_cycle_ofmap_wr_sram)
    ideal_min_clk = min(ideal_start_cycle_ifmap_rd_sram, 
                        ideal_start_cycle_filter_rd_sram, 
                        # ideal_start_cycle_ofmap_rd_sram, 
                        ideal_start_cycle_ofmap_wr_sram)
    ideal_layer_cycle = ideal_max_clk - ideal_min_clk + 1
    ideal_layer_sec = ideal_layer_cycle * period / float(10**6) # period is in us
    ideal_layer_throughput = 1 / ideal_layer_sec
    # ideal_cycle_all += ideal_layer_cycle
    # ideal_sec_all += ideal_layer_sec

    # real
    real_max_clk =  ideal_max_clk + \
                        stall_cycles_filter_rd_sram + stall_cycles_ifmap_rd_sram + stall_cycles_ofmap_wr_sram
    print(utils.bcolors.OKBLUE + 
        f'Stall cycle: input={stall_cycles_ifmap_rd_sram}, weight={stall_cycles_filter_rd_sram}, output={stall_cycles_ofmap_wr_sram}' 
        + utils.bcolors.ENDC)
    real_min_clk =  ideal_min_clk
    real_layer_cycle = real_max_clk - real_min_clk + 1
    real_layer_sec = real_layer_cycle * period / float(10**6)
    real_layer_throughput = 1 / real_layer_sec
    # real_cycle_all += real_layer_cycle
    # real_sec_all += real_layer_sec

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # SRAM: bw,
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # input rd
    sram_bw_ideal_ifmap_rd  =   tot_word_ifmap_rd_sram  * word_sz_bytes_input_rd / float(2**30) / ideal_layer_sec
    sram_bw_real_ifmap_rd   =   tot_word_ifmap_rd_sram  * word_sz_bytes_input_rd / float(2**30) / real_layer_sec

    # weight wr
    sram_bw_ideal_filter_rd =   tot_word_filter_rd_sram  * word_sz_bytes_weight_rd / float(2**30) / ideal_layer_sec
    sram_bw_real_filter_rd  =   tot_word_filter_rd_sram  * word_sz_bytes_weight_rd / float(2**30) / real_layer_sec
    

    # those two situations will not happen simultaneously, if the sram for ofmap is large enough
    sram_bw_ideal_ofmap_wr  =   tot_word_ofmap_wr_sram   * word_sz_bytes_output_wr / float(2**30) / ideal_layer_sec
    sram_bw_real_ofmap_wr   =   tot_word_ofmap_wr_sram   * word_sz_bytes_output_wr / float(2**30) / real_layer_sec
    
    
    sram_bw_ideal_total = sram_bw_ideal_ifmap_rd + sram_bw_ideal_filter_rd  + sram_bw_ideal_ofmap_wr
    sram_bw_real_total = sram_bw_real_ifmap_rd + sram_bw_real_filter_rd + sram_bw_real_ofmap_wr
    
    # tot_word_ifmap_rd_sram_all  += tot_word_ifmap_rd_sram
    # tot_word_filter_rd_sram_all += tot_word_filter_rd_sram
    # tot_word_ofmap_wr_sram_all  += tot_word_ofmap_wr_sram

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # active cycle for dynamic power calculation
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # cycles for pe to be active: during ifmap streaming and ofmap streaming
    act_cycle_ifmap_rd = act_cycles_ifmap_rd_sram
    act_cycle_filter_rd = act_cycles_filter_rd_sram
    # act_cycle_ofmap_rd = act_cycles_ofmap_rd_sram
    act_cycle_ofmap_wr = act_cycles_ofmap_wr_sram

    dynamic_cycle_ireg = act_cycle_ifmap_rd
    dynamic_cycle_wreg = act_cycle_filter_rd
    dynamic_cycle_mac = max(act_cycle_ifmap_rd, act_cycle_filter_rd, act_cycle_ofmap_wr)

    # writing stats to json file
    prefix = 'stats'
    json_file = out_dir / f"{prefix}.json"
    status_dict = dict()

    status_dict = utils.construct_dict(status_dict, 
        cg_lat, cg_util, 
        ideal_layer_cycle, ideal_layer_sec, ideal_layer_throughput, 
        sram_bw_ideal_ifmap_rd, sram_bw_ideal_filter_rd, sram_bw_ideal_ofmap_wr, sram_bw_ideal_total,
        real_layer_cycle, real_layer_sec, real_layer_throughput, 
        sram_bw_real_ifmap_rd, sram_bw_real_filter_rd, sram_bw_real_ofmap_wr, sram_bw_real_total,
        dynamic_cycle_ireg, dynamic_cycle_wreg, dynamic_cycle_mac)

    utils.store_json(json_file, status_dict, indent=4)

def prune(input_list):
    l = []

    for e in input_list:
        e = e.strip() # remove the leading and trailing characters, here space
        if e != '' and e != ' ':
            l.append(e)

    return l

if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    profile()

