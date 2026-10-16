# Qwen3.5 DT inference probe

H2048/I512/E256/top-k8, BF16, single-node EP. This reuses the stable DT profile
and changes only H/I; it is not an autotune winner. Arithmetic stays M16/M64
(existing selector), N128/K64, three stages, with the original DT epilogues.

The production DT capacity limit remains 64 tokens per rank. To measure the
same kernel at 128/512, the diagnostic artifact sets
`--define WEFTMOE_DT_MAX_TOKENS=512`. The single-window guard remains mandatory:
set `WEFTMOE_STATIC_RING_TOKENS=512` and contribution depth 2. This does not
change the production SGLang selector or claim large-batch DT is preferable.

```bash
/opt/.venv/bin/python benchmarks/build_pf_main_timeline.py qwen_small_dt_20261016 \
  --profile configs/dt_qwen35_probe_20261016 --define WEFTMOE_DT_MAX_TOKENS=512
```

Compare through `benchmarks/torchrun_sglang_decode.py --weft-backend dt
--dp-attention --tokens 1 8 64 128 512 --fresh-routes --route-bias-std 0.5`.
Its untimed profiler rejects PF fallback; timing includes fused input staging
and the full inference dataflow, but excludes initialization and weight packing.
