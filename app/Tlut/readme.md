# Temporal-LUT Architectural Simulator

This is directory for Temporal-LUT architectural simulator.

Architecture of configuration taken from CoSA: https://github.com/ucb-bar/cosa

## Dependencies

- pyyaml
- json

## Setup Instruction

Run `source setup_tlut.sh` to set up relavant paths.

## Directory Structure
<pre><code>├── <b>configs</b> (See section <i>config directory structure</i>)
├── <b>output_dir</b> (See section <i>output directory structure</i>)
├── block_trace.py
├── contention_processing.py
├── readme.md
├── run_gen_trace_all.py
├── tlut_systolic_perf_projection.py
├── trace_gen.py
├── tracegen_parse.py
├── utils.py</code></pre>

### Config Directory Structure
<pre><code>├── configs
│   ├── arch
│   │   ├── arch_tlut_systolic_projection.yml
│   │   ├── archs.yml
│   │   ├── [arch_name1].yml
│   │   ├── [arch_name2].yml
│   ├── dataflow
│   │   ├── dtfs.yml
│   │   ├── [dtf_name1].yml
│   │   ├── [dtf_name2].yml
│   ├── workloads
│   │   ├── convnet_graph
│   │   │   ├── layers.yml
│   │   │   ├── [layer_name1].yml
│   │   │   ├── [layer_name2].yml</code></pre>

### Output Directory Structure
<code><pre>├── output_dir
│   ├── <arch_name1>
│   │   ├── <dtf_name1>
│   │   │   ├── <network_name1>
│   │   │   │   ├── <layer_name1>
│   │   │   │   │   ├── sram_read_input.csv <b>(input read ideal trace)</b>
│   │   │   │   │   ├── sram_read_weight.csv <b>(weight read ideal trace)</b>
│   │   │   │   │   ├── sram_read_write.csv <b>(output write ideal trace)</b>
│   │   │   │   │   ├── stats.json <b>(per layer statistics)</b>
│   │   │   │   ├── <layer_name2>
│   │   │   │   │   ├── ...
│   ├── <arch_name2>
│   ├── <arch_name...>
│   ├── bw.pdf <b>(whole network bandwidth comparison across all architectures grouped in architecture sets)</b>
│   ├── <network_name>_<dtf_name>_<arch_set_name>comparison_bw.pdf <b>(whole network bandwidth comparison across one architecture set)</b>
│   ├── <network_name>_<dtf_name>_<arch_set_name>comparison_rt.pdf <b>(whole network latency comparison across one architecture set)</b>
│   ├── ...
│   ├── latency.pdf <b>(whole network latency comparison across all architectures grouped in architecture sets)</b></code></pre>
## Trace Generator

### One-shot profiling instructions:

1. Change `configs/arch/archs.yml` to define archtecture sets and include all the hardware configurations for profiling.
2. Change `configs/dataflow/dtfs.yaml` to include all the dataflow configurations for profiling.
3. Create a yaml file at `configs/workloads/<network name>/layers.yaml `that lists all the nn layers for profiling.
4. run run_gen_trace_all.py. For example, `python3 run_gen_trace_all.py`
5. (Optional:) Temporal-LUT vs systolic array performance projection by `python3 tlut_systolic_perf_projection.py`. 

### Contention-free Trace Generation

For now only the *output stationary dataflow* is supported.

### Contention Processing

Reference code from https://github.com/diwu1990/uSystolic-Sim/blob/main/simHw/block_trace.py

Taking SRAM contention into consideration and add additional stalls due to memory contention.