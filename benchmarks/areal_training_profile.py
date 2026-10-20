"""Bounded PP1 training diagnostics; profiling runs are not performance samples.

The replay runner installs this instrumentation explicitly. No production engine
hook or extra GPU synchronization is required in the timed method wrappers.
"""

from collections import defaultdict
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist


def instrument_engine():
    """Instrument PP1 replay without modifying AReaL's production schedule."""
    import areal.engine.megatron_engine as module

    original = module.MegatronEngine.forward_backward_batch

    def measured(engine, *args, forward_only=False, **kwargs):
        if forward_only:
            return original(engine, *args, forward_only=True, **kwargs)
        diagnostic = TrainingProfile(engine)
        factory = module.get_forward_backward_func

        def profiled_factory():
            schedule = factory()

            def profiled_schedule(*args, **kwargs):
                forward = kwargs["forward_step_func"]

                def step(*args, **kwargs):
                    diagnostic.microbatch()
                    return forward(*args, **kwargs)

                kwargs["forward_step_func"] = step
                return schedule(*args, **kwargs)

            return profiled_schedule

        module.get_forward_backward_func = profiled_factory
        try:
            return original(engine, *args, forward_only=False, **kwargs)
        finally:
            module.get_forward_backward_func = factory
            diagnostic.finish()

    module.MegatronEngine.forward_backward_batch = measured


class TrainingProfile:
    def __init__(self, engine):
        self.directory = Path(os.environ["AREAL_MEGATRON_PROFILE_DIR"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank = dist.get_rank()
        self.step = getattr(engine, "_diagnostic_training_step", 0)
        engine._diagnostic_training_step = self.step + 1
        self.mb = -1
        self.profiler = None
        self.elapsed = defaultdict(list)
        self.runtime_setup = None
        self.originals = []
        self.routes = []
        self.started = time.perf_counter()
        if self.step == 0:
            from megatron.core import parallel_state as mpu

            if mpu.get_tensor_model_parallel_rank() == 0:
                # The diagnostic hook is called from forward_backward_batch,
                # directly under train_batch. Save CPU training inputs only,
                # allowing identical-batch replay without another rollout.
                batch = sys._getframe(2).f_locals["input_"]
                torch.save(batch, self.directory / f"batch.rank{self.rank}.pt")
        adapter = getattr(engine, "_weftmoe_adapter", None)
        if adapter is not None:
            self._wrap(adapter, "get_runtime", "runtime_prepare")
            self._wrap(adapter, "pack", "weight_stack")
            from weftmoe.runtime import TensorRuntime

            self._wrap(TensorRuntime, "forward", "native_forward")
            self._wrap(TensorRuntime, "backward", "native_backward")
            if hasattr(TensorRuntime, "forward_async"):
                self._wrap(TensorRuntime, "forward_async", "native_forward")
                self._wrap(TensorRuntime, "backward_async", "native_backward")

    def _wrap(self, target, method, label):
        original = getattr(target, method)
        self.originals.append((target, method, original))

        def measured(*args, **kwargs):
            if label == "native_forward" and self.step == 0 and self.mb == 2:
                # Retain only routing metadata, never model weights/checkpoints.
                self.routes.append(args[2].detach())
            scope = torch.profiler.record_function("weft/" + label) if self.profiler else nullcontext()
            with scope:
                begin = time.perf_counter()
                result = original(*args, **kwargs)
                self.elapsed[label].append((self.mb, time.perf_counter() - begin))
                if label == "runtime_prepare" and self.runtime_setup is None:
                    if hasattr(result.native, "setup_seconds"):
                        self.runtime_setup = result.native.setup_seconds()
                        if self.rank == 0:
                            print("WEFT_SETUP_JSON " + json.dumps({
                                "step": self.step, "seconds": self.runtime_setup,
                                "usage": result.native.usage(),
                            }), flush=True)
                return result

        setattr(target, method, measured)

    def microbatch(self):
        self.mb += 1
        if self.rank == 0 and self.step == 0:
            if self.mb == 2:
                self.profile_label = "warm"
                self.profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False, with_stack=False,
                )
                self.profiler.start()
            elif self.mb == 3:
                self._stop()

    def _stop(self):
        if self.profiler is None:
            return
        self.profiler.stop()
        prefix = self.directory / f"step{self.step}.rank{self.rank}.{self.profile_label}"
        self.profiler.export_chrome_trace(str(prefix) + ".trace.json")
        self.profiler = None

    def finish(self):
        self._stop()
        for target, method, original in reversed(self.originals):
            setattr(target, method, original)
        prefix = self.directory / f"step{self.step}.rank{self.rank}"
        summary = {
            "rank": self.rank, "step": self.step, "microbatches": self.mb + 1,
            "wall_seconds": time.perf_counter() - self.started,
            "host_call_seconds": dict(self.elapsed),
            "runtime_setup_seconds": self.runtime_setup,
        }
        prefix.with_suffix(prefix.suffix + ".timings.json").write_text(json.dumps(summary))
        if self.routes:
            torch.save([r.cpu() for r in self.routes], str(prefix) + ".routes.pt")
