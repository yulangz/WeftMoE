#!/usr/bin/env python3
"""Run the two-host Megatron baseline over a verified NCCL IB/RoCE path.

The launcher fails closed unless NCCL reports NET/IB and GDRDMA, no NCCL
NET/Socket payload transport is selected, and configured HCA counters advance.
The SSH reverse tunnel is used only for torch elastic rendezvous.
"""

import argparse
import hashlib
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nccl_library import configure_nccl_library
from run_multinode_megatron_socket import (
    benchmark_arguments,
    extract_record,
    remote_cleanup,
    terminate,
    torchrun_arguments,
)


PRIVATE_IPV4 = re.compile(
    r"(?<![0-9])(?:10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|"
    r"192\.168\.[0-9]{1,3}\.[0-9]{1,3}|"
    r"172\.(?:1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3})(?![0-9])")
COUNTER_NAMES = ("port_xmit_data", "port_rcv_data")


def redact(text):
    return PRIVATE_IPV4.sub("PRIVATE_IPV4", text)


def normalize_evidence_line(line):
    line = redact(line)
    return re.sub(r"^[^: ]+:[0-9]+:[0-9]+", "HOST:PID:TID", line)


def parse_hcas(raw):
    hcas = []
    for item in raw.split(","):
        name = item.strip().split(":", 1)[0]
        if name and name not in hcas:
            hcas.append(name)
    if not hcas:
        raise ValueError("--ib-hca did not contain an HCA")
    return hcas


def local_counter_snapshot(hcas, gid_index):
    result = {}
    for hca in hcas:
        base = Path("/sys/class/infiniband") / hca / "ports" / "1"
        if not base.is_dir():
            raise RuntimeError(f"configured HCA is absent: {hca}")
        state = (base / "state").read_text(encoding="utf-8").strip()
        if "ACTIVE" not in state:
            raise RuntimeError(f"configured HCA is not active: {hca} state={state}")
        counters = {}
        for name in COUNTER_NAMES:
            # IB port data counters count four-octet words.
            words = int((base / "counters" / name).read_text(
                encoding="utf-8").strip())
            counters[name.removeprefix("port_") + "_bytes"] = words * 4
        gid_base = base / "gid_attrs"
        result[hca] = {
            **counters,
            "rate": (base / "rate").read_text(encoding="utf-8").strip(),
            "gid_type": (gid_base / "types" / str(gid_index)).read_text(
                encoding="utf-8").strip(),
            "netdev": (gid_base / "ndevs" / str(gid_index)).read_text(
                encoding="utf-8").strip(),
        }
    return result


def remote_counter_snapshot(args, hcas):
    code = (
        "import json; from pathlib import Path; "
        f"names={hcas!r}; gid={args.ib_gid_index!r}; out={{}}; "
        "exec(\"for name in names:\\n"
        " base=Path('/sys/class/infiniband')/name/'ports'/'1'\\n"
        " state=(base/'state').read_text().strip()\\n"
        " assert 'ACTIVE' in state, (name,state)\\n"
        " out[name]={'xmit_data_bytes':int((base/'counters'/'port_xmit_data').read_text().strip())*4,"
        "'rcv_data_bytes':int((base/'counters'/'port_rcv_data').read_text().strip())*4,"
        "'rate':(base/'rate').read_text().strip(),"
        "'gid_type':(base/'gid_attrs'/'types'/str(gid)).read_text().strip(),"
        "'netdev':(base/'gid_attrs'/'ndevs'/str(gid)).read_text().strip()}\"); "
        "print(json.dumps(out,sort_keys=True))")
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-p", str(args.port), args.remote,
         shlex.join([args.remote_python, "-c", code])],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=20.0, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "remote HCA counter snapshot failed: " + redact(completed.stderr))
    return json.loads(completed.stdout)


def counter_delta(before, after):
    result = {}
    for hca in sorted(before):
        if hca not in after:
            raise RuntimeError(f"HCA disappeared during benchmark: {hca}")
        result[hca] = {
            "tx_bytes": after[hca]["xmit_data_bytes"] -
                        before[hca]["xmit_data_bytes"],
            "rx_bytes": after[hca]["rcv_data_bytes"] -
                        before[hca]["rcv_data_bytes"],
            "rate": after[hca]["rate"],
            "gid_type": after[hca]["gid_type"],
            "netdev": after[hca]["netdev"],
        }
        if result[hca]["tx_bytes"] < 0 or result[hca]["rx_bytes"] < 0:
            raise RuntimeError(f"HCA counter moved backwards: {hca}")
    return result


def log_evidence(text):
    categories = {"version": [], "selection": [], "gdrdma": []}
    for line in text.splitlines():
        category = None
        if "NCCL version" in line:
            category = "version"
        elif "NET/IB : Using" in line or "Using network IB" in line:
            category = "selection"
        elif "GDRDMA" in line:
            category = "gdrdma"
        if category is not None:
            sanitized = normalize_evidence_line(line)
            if sanitized not in categories[category]:
                categories[category].append(sanitized)
    limits = {"version": 2, "selection": 4, "gdrdma": 16}
    return [line for category in categories
            for line in categories[category][:limits[category]]]


def validate_rdma_logs(local_text, remote_text):
    combined = local_text + "\n" + remote_text
    evidence = log_evidence(combined)
    ib_lines = [line for line in evidence if "NET/IB" in line]
    gdr_lines = [line for line in evidence if "GDRDMA" in line]
    socket_lines = [normalize_evidence_line(line)
                    for line in combined.splitlines()
                    if "Using network Socket" in line or "NET/Socket" in line]
    if not ib_lines:
        raise RuntimeError("NCCL did not report NET/IB; RDMA result rejected")
    if not gdr_lines:
        raise RuntimeError("NCCL did not report GDRDMA; RDMA result rejected")
    if socket_lines:
        raise RuntimeError(
            "NCCL selected a Socket payload transport; RDMA result rejected: " +
            " | ".join(socket_lines[:8]))
    return {
        "net_ib_observed": True,
        "gdrdma_observed": True,
        "socket_payload_fallback_observed": False,
        "key_log_lines": evidence,
        "local_log_sha256": hashlib.sha256(
            local_text.encode("utf-8", errors="replace")).hexdigest(),
        "remote_log_sha256": hashlib.sha256(
            remote_text.encode("utf-8", errors="replace")).hexdigest(),
    }


def write_evidence(directory, local_text, remote_text, evidence):
    if not directory:
        return
    root = Path(directory)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    (root / "local-nccl.log").write_text(redact(local_text), encoding="utf-8")
    (root / "remote-nccl.log").write_text(redact(remote_text), encoding="utf-8")
    (root / "rdma-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--master-port", type=int, default=29631)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--torchrun", default="/opt/.venv/bin/torchrun")
    parser.add_argument("--remote-torchrun", default="/opt/.venv/bin/torchrun")
    parser.add_argument("--remote-python", default="python3")
    parser.add_argument("--nccl-lib-dir", required=True)
    parser.add_argument("--socket-ifname", required=True,
                        help="NCCL bootstrap/OOB interface, not payload rail")
    parser.add_argument("--ib-hca", required=True,
                        help="comma-separated NCCL HCA:port allowlist")
    parser.add_argument("--ib-gid-index", type=int, required=True)
    parser.add_argument("--ib-address-family", default="AF_INET")
    parser.add_argument("--ib-roce-version", default="2")
    parser.add_argument("--net-gdr-level", default="SYS")
    parser.add_argument("--ib-pci-relaxed-ordering", default="1")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--evidence-dir")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--gate-mode", choices=("fixed-random", "unit"),
                        default="fixed-random")
    parser.add_argument("--expert-body", choices=("identity", "te-grouped-mlp"),
                        default="te-grouped-mlp")
    args = parser.parse_args()
    if args.gpus_per_node <= 0:
        parser.error("--gpus-per-node must be positive")
    if args.ib_gid_index < 0:
        parser.error("--ib-gid-index must be non-negative")
    hcas = parse_hcas(args.ib_hca)

    launch_id = secrets.token_hex(16)
    environment = os.environ.copy()
    library = configure_nccl_library(environment, args.nccl_lib_dir)
    rdma_environment = {
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "NET,INIT",
        "NCCL_NET": "IB",
        "NCCL_IB_DISABLE": "0",
        "NCCL_IB_HCA": args.ib_hca,
        "NCCL_IB_GID_INDEX": str(args.ib_gid_index),
        "NCCL_IB_ADDR_FAMILY": args.ib_address_family,
        "NCCL_IB_ROCE_VERSION_NUM": args.ib_roce_version,
        "NCCL_NET_GDR_LEVEL": args.net_gdr_level,
        "NCCL_IB_PCI_RELAXED_ORDERING": args.ib_pci_relaxed_ordering,
        "NCCL_SOCKET_IFNAME": args.socket_ifname,
        "MEGAMOE_MULTINODE_LAUNCH_ID": launch_id,
    }
    environment.update(rdma_environment)
    remote_overrides = {
        key: environment[key] for key in
        ("LD_LIBRARY_PATH", "MEGAMOE_NCCL_LIBRARY", *rdma_environment.keys())
    }
    before_local = local_counter_snapshot(hcas, args.ib_gid_index)
    before_remote = remote_counter_snapshot(args, hcas)
    launch_begin = time.monotonic()
    local_process = None
    remote_process = None
    local_text = ""
    remote_text = ""
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as local_log, \
            tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as remote_log:
        try:
            local_process = subprocess.Popen(
                torchrun_arguments(args, 0),
                cwd=Path(__file__).resolve().parents[1], env=environment,
                stdout=local_log, stderr=subprocess.STDOUT, text=True)
            time.sleep(0.5)
            remote_command = [
                "env", *(f"{key}={value}"
                         for key, value in remote_overrides.items()),
                *torchrun_arguments(args, 1),
            ]
            remote_process = subprocess.Popen(
                ["ssh", "-o", "BatchMode=yes",
                 "-o", "ExitOnForwardFailure=yes",
                 "-R", f"{args.master_port}:127.0.0.1:{args.master_port}",
                 "-p", str(args.port), args.remote,
                 "cd " + shlex.quote(str(Path.cwd())) + " && exec " +
                 shlex.join(remote_command)],
                stdout=remote_log, stderr=subprocess.STDOUT, text=True)
            deadline = time.monotonic() + args.timeout
            while (local_process.poll() is None or remote_process.poll() is None):
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(
                        "two-host Megatron RDMA benchmark", args.timeout)
                time.sleep(0.1)
            if local_process.returncode != 0 or remote_process.returncode != 0:
                raise RuntimeError(
                    f"Megatron launch failed local={local_process.returncode} "
                    f"remote={remote_process.returncode}")
        except BaseException as error:
            terminate(local_process)
            terminate(remote_process)
            remote_cleanup(args, launch_id, "TERM")
            time.sleep(0.25)
            remote_cleanup(args, launch_id, "KILL")
            local_log.seek(0)
            remote_log.seek(0)
            local_text = local_log.read()
            remote_text = remote_log.read()
            evidence = log_evidence(local_text + "\n" + remote_text)
            write_evidence(args.evidence_dir, local_text, remote_text,
                           {"key_log_lines": evidence})
            raise RuntimeError(
                f"{error}; local_tail={redact(local_text[-6000:])}; "
                f"remote_tail={redact(remote_text[-6000:])}") from None
        finally:
            terminate(local_process)
            terminate(remote_process)
        local_log.seek(0)
        remote_log.seek(0)
        local_text = local_log.read()
        remote_text = remote_log.read()

    after_local = local_counter_snapshot(hcas, args.ib_gid_index)
    after_remote = remote_counter_snapshot(args, hcas)
    launch_seconds = time.monotonic() - launch_begin
    local_delta = counter_delta(before_local, after_local)
    remote_delta = counter_delta(before_remote, after_remote)
    evidence = validate_rdma_logs(local_text, remote_text)
    active_rails = sorted({
        hca for hca in hcas
        if local_delta[hca]["tx_bytes"] + local_delta[hca]["rx_bytes"] > 0 and
           remote_delta[hca]["tx_bytes"] + remote_delta[hca]["rx_bytes"] > 0
    })
    if set(active_rails) != set(hcas):
        raise RuntimeError(
            "not every configured HCA counter advanced on both hosts: "
            f"expected={sorted(hcas)} actual={active_rails}")
    evidence["active_hcas_with_bidirectional_host_evidence"] = active_rails
    evidence["local_hca_counter_delta"] = local_delta
    evidence["remote_hca_counter_delta"] = remote_delta
    evidence["launcher_elapsed_seconds"] = launch_seconds
    evidence["aggregate_hca_counter_throughput_gbytes_per_second"] = {
        "local_tx": sum(item["tx_bytes"] for item in local_delta.values()) /
                    launch_seconds / 1.0e9,
        "local_rx": sum(item["rx_bytes"] for item in local_delta.values()) /
                    launch_seconds / 1.0e9,
        "remote_tx": sum(item["tx_bytes"] for item in remote_delta.values()) /
                     launch_seconds / 1.0e9,
        "remote_rx": sum(item["rx_bytes"] for item in remote_delta.values()) /
                     launch_seconds / 1.0e9,
    }
    write_evidence(args.evidence_dir, local_text, remote_text, evidence)

    record = extract_record(local_text)
    if record.get("world") != 2 * args.gpus_per_node:
        raise RuntimeError(
            f"Megatron world mismatch expected={2 * args.gpus_per_node} "
            f"actual={record.get('world')}")
    if record.get("nccl_runtime_version") != "2.29.7":
        raise RuntimeError(
            f"unexpected NCCL runtime {record.get('nccl_runtime_version')}")
    record["network_backend"] = "nccl-rdma-roce-gdr"
    record["network_environment"] = {
        key: value for key, value in rdma_environment.items()
        if key != "MEGAMOE_MULTINODE_LAUNCH_ID"
    }
    record["explicit_nccl_library_path"] = library
    record["rdma_evidence"] = evidence
    print("MULTINODE_MEGATRON_RDMA_JSON " + json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        raise SystemExit(1)
