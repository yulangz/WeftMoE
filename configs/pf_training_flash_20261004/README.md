# Flash PF training profile, 2026-10-04

H4096 / I2048 / E256 / top-k6, BF16, DP-local inputs, NVIDIA L20X.
One shared compiled profile (`e09cd7591438`) serves single-node EP8 and
two-node EP16. This is a **PF training profile**, not a replacement DT profile.
The original default libraries and other model profiles are unchanged.

| EP | B / D | Prepared capacities in the table |
|---|---|---|
| 8 | 2048 / 2 | 10175, 25820 |
| 16 | 3072 / 3 | 10390, 25964 |

These capacities are the maxima of the measured variable-token traces near
8192 and 20480 tokens/rank, not nominal token buckets or supported-size limits.
The table key uses the exact prepared capacity; do not round another workload
to one of these entries. Search for the intended shape/trace with
`tools/autotune_training.py --config configs/pf_training_autotune.json`.

The selected FWD uses FC1/FC2 N256/N256, K128, 2/2 static stages; BWD retains
dU/dX N256, dW M128/N256, K64, activation stages 4 and weight-gradient stages 3.
Both additional prefill-entry thresholds are zero: this library directly runs
the selected PF arithmetic, with no DT entry selected by the small-token path.

Build using the existing CUDA/CUTLASS/NCCL environment:

```bash
export NVCCFLAGS="-lineinfo -DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1 -I${PWD}/configs/pf_training_flash_20261004"
python build_native.py lib --output build/flash_pf/libweftmoe.so --jobs 6
python tools/build_tensor_runtime.py --library build/flash_pf/libweftmoe.so
```

Set `WEFTMOE_TENSOR_EXTENSION` to the extension path printed by the adapter
builder, and `WEFTMOE_AUTOTUNE_TABLE` to this directory's `table.json`.
The latter contains `training=true` entries; it performs lookup only, not
runtime compilation, benchmarking, or per-call ring-layout changes.

Search and validation evidence are recorded in
[the training report](../../docs/static_forward_v4_training_record.md).
The B2304/D2 network candidate stalled during repeated confirmation and was
excluded; this is not a claim that its unresolved wait condition was fixed.
The reported winner is the best validated candidate in this staged search,
not a proof of a global optimum or of performance on other model dimensions.
