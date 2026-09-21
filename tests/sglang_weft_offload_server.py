"""Standalone full-model AWEX residency test server, without a training writer.

Pass normal SGLang server arguments and AREAL_USE_WEFTMOE=1. Register the same
AWEX scheduler hooks without a MetaServer background worker, then exercise the
HTTP release/resume endpoints. Requires the patched AReaL checkout on PYTHONPATH.
"""

import sys

from areal.engine.awex.memory_saver import patch_tms_hook_mode

patch_tms_hook_mode()


def scheduler_process(*args, **kwargs):
    import torch
    from areal.engine.awex.sglang_plugin import register_awex_plugin
    import weftmoe.sglang as adapter
    from weftmoe.sglang import install

    install()
    for name in ("suspend", "resume"):
        original = getattr(adapter, name)

        def traced(runner, original=original, name=name):
            original(runner)
            print(
                "RESIDENCY",
                torch.distributed.get_rank(),
                name,
                "allocated",
                torch.cuda.memory_allocated(),
                "reserved",
                torch.cuda.memory_reserved(),
                "free",
                torch.cuda.mem_get_info()[0],
                flush=True,
            )

        setattr(adapter, name, traced)
    register_awex_plugin()
    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)


if __name__ == "__main__":
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args

    launch_server(
        prepare_server_args(sys.argv[1:]),
        run_scheduler_process_func=scheduler_process,
    )
