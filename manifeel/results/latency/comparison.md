# Inference latency: gr00t_n1.7_flow_matching vs. gr00t_n1.7_vlm_drifting

GPU: NVIDIA H200. Median over 100 calls with different ManiFeel observations (wrist 256x256 + EEF state + task text) -> one 16-step action chunk. End to end = the full policy call as in the eval server (pre-processing, backbone, action head, post-processing); backbone and action head are timed separately (CUDA-synchronized).

## Batch size 1 (one robot)

| Model | End to end | Backbone | Action head | Other | Speed-up end to end | Speed-up head |
|---|---|---|---|---|---|---|
| gr00t_n1.7_flow_matching | 85.4 ms | 29.2 ms | 53.7 ms | 2.6 ms | 1.00x | 1.00x |
| gr00t_n1.7_vlm_drifting | 40.3 ms | 24.9 ms | 8.5 ms | 6.8 ms | 2.12x | 6.29x |

## Batch size 50 (ManiFeel's parallel eval envs)

| Model | End to end | Backbone | Action head | Other | Speed-up end to end | Speed-up head |
|---|---|---|---|---|---|---|
| gr00t_n1.7_flow_matching | 213.0 ms | 104.4 ms | 55.2 ms | 53.3 ms | 1.00x | 1.00x |
| gr00t_n1.7_vlm_drifting | 184.1 ms | 122.1 ms | 9.9 ms | 52.2 ms | 1.16x | 5.58x |

## Parameters

| Model | Backbone | Action head |
|---|---|---|
| gr00t_n1.7_flow_matching | 1524 M | 1621 M |
| gr00t_n1.7_vlm_drifting | 1524 M | 386 M |

Control budget: 8 steps x 100 ms = 800 ms per call at 10 Hz.
