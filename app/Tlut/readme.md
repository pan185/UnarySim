# Temporal-LUT Architectural Simulator

This is directory for Temporal-LUT architectural simulator.

Architecture of configuration taken from CoSA: https://github.com/ucb-bar/cosa

## Dependencies

- pyyaml
- json

## Directory Structure
<pre><code>├── configs
│   ├── arch
│   │   ├── archs.yml
│   │   ├── <arch_name1>.yml
│   │   ├── <arch_name2>.yml
│   ├── dataflow
│   │   ├── dtfs.yml
│   │   ├── <dtf_name1>.yml
│   │   ├── <dtf_name2>.yml
│   ├── workloads
│   │   ├── convnet_graph
│   │   │   ├── layers.yml
│   │   │   ├── <layer_name1>.yml
│   │   │   ├── <layer_name2>.yml
├── <b>output_dir</b> (See section <i>output directory structure</i>)
├── block_trace.py
├── contention_processing.py
├── readme.md
├── run_gen_trace_all.py
├── trace_gen.py
├── tracegen_parse.py
├── utils.py</code></pre>

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
│   ├── latency.pdf <b>(whole network latency comparison across all architectures grouped in architecture sets)</b>
</code></pre>
## Trace Generator

### One-shot profiling instructions:

1. Change `configs/arch/archs.yml` to define archtecture sets and include all the hardware configurations for profiling.
2. Change `configs/dataflow/dtfs.yaml` to include all the dataflow configurations for profiling.
3. Create a yaml file at `configs/workloads/<network name>/layers.yaml `that lists all the nn layers for profiling.
4. run run_gen_trace_all.py. For example, `python3 run_gen_trace_all.py`

### Contention-free Trace Generation

For now only the *output stationary dataflow* is supported.

### Contention Processing

Reference code from https://github.com/diwu1990/uSystolic-Sim/blob/main/simHw/block_trace.py

Taking SRAM contention into consideration and add additional stalls due to memory contention.

<pre><code>However, I have another option.
I can use a code block that works very nicely 
but I can not <b>bold text</b> a particular part of the code
<a href="https://meta.stackexchange.com/q/243709">I want to stand out.</a>
Just to make it easier to locate.
Nor can I make the font smaller and get rid of that ugly scroll bar. It's just slightly too wide.</code></pre>