"""Start SGLang HTTP serving with the WeftMoE adapter installed in each worker."""

import sys


def scheduler_process(*args, **kwargs):
    from sglang.srt.managers.scheduler import run_scheduler_process
    from weftmoe.sglang import install

    install()
    return run_scheduler_process(*args, **kwargs)


if __name__ == "__main__":
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args

    args = prepare_server_args(["--moe-runner-backend", "triton", "--moe-a2a-backend", "none", *sys.argv[1:]])
    launch_server(args, run_scheduler_process_func=scheduler_process)
