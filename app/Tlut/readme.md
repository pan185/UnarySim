This is directory for Temporal-LUT architectural simulator.

Architecture taken from CoSA: https://github.com/ucb-bar/cosa

# Dependencies

- pyyaml
- json

# Trace Generator

## One-shot profiling instructions:

1. Change `configs/arch/archs.yml` to include all the hardware configurations for profiling.
2. Change `configs/dataflow/dtfs.yaml` to include all the dataflow configurations for profiling.
3. Create a yaml file that lists all the nn layers for profiling
4. run run_gen_trace_all.py with -nn `<nn layer yml file path>`. For example, `python3 run_gen_trace_all.py -nn /home/zhewen/Repo/UnarySim/app/Tlut/configs/workloads/convnet_graph/layers.yaml`

## Contention-free Trace Generation

For now only the *output stationary dataflow* is supported.

## Contention Processing

Reference code from https://github.com/diwu1990/uSystolic-Sim/blob/main/simHw/block_trace.py

Taking SRAM contention into consideration and add additional stalls due to memory load and store.
