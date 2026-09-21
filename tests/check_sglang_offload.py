"""Exercise real AWEX CUDA graph/KV offload and compare greedy generations."""

import argparse
import json
import subprocess
import time

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args()

    def post(endpoint, payload):
        response = requests.post(args.url + endpoint, json=payload, timeout=600)
        response.raise_for_status()
        return response.json()

    def memory():
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
            text=True,
        ).strip()

    prompt = {
        "text": "Calculate 2 + 2. Answer:",
        "sampling_params": {"temperature": 0, "max_new_tokens": 32},
        "return_logprob": True,
    }
    reference = post("/generate", prompt)
    expected = reference["meta_info"]["output_token_logprobs"]
    expected_ids = [entry[1] for entry in expected]
    print("reference", json.dumps(reference["text"]), expected_ids, flush=True)
    for cycle in range(args.cycles):
        print("before", cycle, memory(), flush=True)
        begin = time.monotonic()
        post("/release_memory_occupation", {"tags": ["kv_cache", "cuda_graph"]})
        print("released", cycle, memory(), flush=True)
        post("/resume_memory_occupation", {"tags": ["kv_cache", "cuda_graph"]})
        print(
            "resumed", cycle, "seconds", time.monotonic() - begin, memory(), flush=True
        )
        result = post("/generate", prompt)
        ids = [entry[1] for entry in result["meta_info"]["output_token_logprobs"]]
        assert ids == expected_ids, (cycle, ids, expected_ids)
        print("AWEX_GRAPH_OFFLOAD_PARITY_PASS", cycle, flush=True)


if __name__ == "__main__":
    main()
