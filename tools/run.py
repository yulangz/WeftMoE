"""Unified repository entry point for JIT compilation, benchmarking and autotuning.

This executes generated benchmark inputs through the native rank launchers.
It is not a tensor operator binding. Call run() once during preparation, before
starting application work or CUDA Graph capture.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from weftmoe import ModelSpec, RuntimeSettings
from autotune_common import (
    DEFAULT_COMPILE_JOBS, DEFAULT_FINAL_ROUNDS, configure_preparation,
    resolve_parallel_settings, validate_execution_topology, write_result,
)


def run(model: ModelSpec, *, phase: str = 'forward', tokens: int = 8192,
        autotune: bool = False, nodes: int | None = None, gpus_per_node: int | None = None,
        config: str | Path | None = None, system: dict | None = None,
        jobs: int = DEFAULT_COMPILE_JOBS, output: str | Path | None = None,
        generate_only: bool = False) -> dict:
    """Compile and benchmark one configuration, optionally selecting it by tuning.

    Tuning measures a bucket representative. The final benchmark always uses
    the requested token count. Explicit config values take precedence over the
    untuned defaults. Result and log paths are recorded in the returned object.
    """
    if phase not in ('forward', 'backward'):
        raise ValueError('phase must be forward or backward')
    if type(tokens) is not int or tokens <= 0:
        raise ValueError('tokens must be a positive integer')
    if type(jobs) is not int or not 1 <= jobs <= 32:
        raise ValueError('jobs must be an integer in [1, 32]')
    module = importlib.import_module(f'autotune_{phase}')
    settings = RuntimeSettings.from_environment()
    system = dict(system or {})
    if nodes is not None:
        system["nodes"] = nodes
    if gpus_per_node is not None:
        system["gpus_per_node"] = gpus_per_node
    topology = resolve_parallel_settings(system)
    nodes = topology["nodes"]
    topology_name = 'single_node' if nodes == 1 else 'multi_node'
    explicit_config = config or settings.autotune_config
    config_path = Path(explicit_config or ROOT / 'configs' /
                       f'{phase}_h200_{topology_name}_autotune.json').resolve()
    workload = {'tokens_per_rank': tokens}
    plan = module.load_plan(config_path, model=asdict(model), system=system,
                            benchmark=workload)
    configure_preparation(plan, phase, exact_tokens=not autotune)
    plan['jit']['jobs'] = jobs
    candidates = module.expand_candidates(plan)
    candidate = candidates[0]
    output = Path(output or settings.tune_directory(phase) /
                  f'run-{uuid.uuid4().hex}.json').resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(phase=phase, mode='autotune' if autotune else 'benchmark',
                  requested_tokens=tokens, topology=topology, output=str(output))
    if generate_only:
        result.update(status='generated', jit=plan['jit'],
                      benchmark=plan['benchmark'], selection_policy=plan['selection'],
                      candidates=candidates if autotune else [candidate])
        write_result(output, result)
        return result
    validate_execution_topology(plan['benchmark'])
    log_dir = output.parent / (output.stem + '-logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    if autotune:
        tune_path = output.with_name(output.stem + '-tune.json')
        system_path = log_dir / 'system.json'
        write_result(system_path, system)
        arguments = ['--config', str(config_path), '--system-config', str(system_path),
                     '--tokens', str(tokens), '--jobs', str(jobs),
                     '--output', str(tune_path), '--skip-final']
        for key, value in asdict(model).items():
            arguments += ['--' + key.replace('_', '-'), str(value)]
        if module.main(arguments) != 0:
            raise RuntimeError(f'autotune failed; see {tune_path}')
        tuning = json.loads(tune_path.read_text())
        selected = tuning['selected']
        candidate = dict(name=selected['name'], config=selected['config'])
        binary = Path(selected['binary'])
        result['tuning_result'] = str(tune_path)
    else:
        environment = (module._sanitized_environment() if phase == 'forward'
                       else module.sanitized_environment())
        compiler_class = (module.ForwardJitCompiler if phase == 'forward'
                          else module.BackwardJitCompiler)
        compiler = compiler_class(ROOT, plan['jit'], environment,
                                  nccl_lib_dir=plan['benchmark'].get('nccl_lib_dir'))
        built = module.compile_candidate(candidate, compiler, log_dir, command_jobs=jobs)
        result['build'] = built
        if built['returncode'] != 0:
            write_result(output, result | {'status': 'build_failed'})
            raise RuntimeError(f'compilation failed; see {output}')
        binary = Path(built['binary'])
    environment = (module._sanitized_environment() if phase == 'forward'
                   else module.sanitized_environment())
    # Bucket size selects parameters, not the shape of this final measurement.
    benchmark = dict(plan['benchmark'], tokens_per_rank=tokens)
    measured = module.benchmark_candidate(candidate, binary, benchmark, log_dir,
                                          environment, rounds=DEFAULT_FINAL_ROUNDS,
                                          suffix='run')
    success = measured['returncode'] == 0 and measured['record'] is not None
    result.update(status='complete' if success else 'benchmark_failed',
                  config=candidate['config'], binary=str(binary), benchmark=measured)
    write_result(output, result)
    if not success:
        raise RuntimeError(f'benchmark failed; see {output}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['forward', 'backward'], default='forward')
    for name in ('hidden', 'intermediate', 'experts', 'top-k'):
        parser.add_argument('--' + name, type=int, required=True)
    parser.add_argument('--tokens', type=int, default=8192)
    parser.add_argument('--autotune', action='store_true')
    parser.add_argument('--nodes', type=int)
    parser.add_argument('--gpus-per-node', type=int)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--system-config', type=Path)
    parser.add_argument('--remote')
    parser.add_argument('--port', type=int)
    parser.add_argument('--jobs', type=int, default=DEFAULT_COMPILE_JOBS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--generate-only', action='store_true')
    args = vars(parser.parse_args())
    model = ModelSpec(**{key: args.pop(key) for key in ('hidden', 'intermediate', 'experts', 'top_k')})
    path = args.pop('system_config')
    system = json.loads(path.read_text()) if path else {}
    for key in ('remote', 'port'):
        value = args.pop(key)
        if value is not None:
            system[key] = value
    result = run(model, system=system, **args)
    print(f"{result['status']}: {result['output']}")


if __name__ == '__main__':
    main()
