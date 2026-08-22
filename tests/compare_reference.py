import json
import sys

from reference.notify_dispatch_reference import (
    ReferenceConfig,
    ReferenceTopology,
    build_reference_all,
)


MASK64 = (1 << 64) - 1
RANDOM_SEEDS = [
    0x5EED1234, 0x10203040, 0xDEADBEEF, 0xC001D00D, 0x01234567,
    0x89ABCDEF, 0x31415926, 0x27182818, 0x0BADF00D, 0xFEEDFACE,
    0x13579BDF, 0x2468ACE0, 0xA5A5A5A5, 0x5A5A5A5A, 0x11223344,
    0x55667788, 0x99AABBCC, 0xDDEEFF00, 0x42424242, 0x77777777,
    0xABCDEF01, 0x10FEDCBA, 0x600DBEEF, 0x7F4A7C15, 0xF00DCAFE,
]


def make_config(nodes, gpus, experts, top_k, mtile):
    world = nodes * gpus
    return ReferenceConfig(
        experts, top_k, mtile,
        ReferenceTopology(
            world, nodes,
            [node for node in range(nodes) for _ in range(gpus)],
            [local for _ in range(nodes) for local in range(gpus)],
            [expert % world for expert in range(experts)],
        ),
    )


def seeded_inputs(world_size, experts, top_k, seed):
    def next_value():
        nonlocal seed
        seed = (seed * 6364136223846793005 + 1442695040888963407) & MASK64
        return seed

    routes = []
    for _rank in range(world_size):
        tokens = []
        for _token in range(next_value() % 6):
            candidates = list(range(experts))
            selected_experts = []
            for slot in range(top_k):
                selected = slot + next_value() % (experts - slot)
                candidates[slot], candidates[selected] = candidates[selected], candidates[slot]
                selected_experts.append(candidates[slot])
            tokens.append(selected_experts)
        routes.append(tokens)
    return routes


def expected_fixtures():
    targeted = make_config(2, 2, 6, 3, 4)
    targeted = ReferenceConfig(
        targeted.num_experts, targeted.top_k, targeted.m_tile_rows,
        ReferenceTopology(4, 2, [0, 0, 1, 1], [0, 1, 0, 1],
                          [0, 1, 2, 3, 2, 3]),
    )
    targeted_routes = [
        [[2, 4, 3], [0, 2, 5]], [], [], []
    ]

    many_config = make_config(73, 1, 73, 2, 8)
    many_routes = [[] for _ in range(73)]
    many_routes[0] = [[0, 72]]

    skew_config = make_config(2, 2, 8, 2, 4)
    skew_routes = [
        [[0, 1], [0, 2], [0, 3]],
        [[0, 4], [0, 5]],
        [],
        [[0, 6]],
    ]
    fixtures = {
        "targeted": build_reference_all(targeted, targeted_routes,
                                          "cross-targeted", 3),
    }
    random_config = make_config(2, 2, 11, 4, 3)
    for index, seed in enumerate(RANDOM_SEEDS):
        suffix = f"{index:02d}"
        random_routes = seeded_inputs(4, 11, 4, seed)
        first = build_reference_all(
            random_config, random_routes, f"cross-random-{suffix}", 100 + index)
        second = build_reference_all(
            random_config, random_routes, f"cross-random-{suffix}", 100 + index)
        first_bytes = json.dumps(first, separators=(",", ":"), ensure_ascii=True)
        second_bytes = json.dumps(second, separators=(",", ":"), ensure_ascii=True)
        if first_bytes != second_bytes:
            raise AssertionError(
                f"Python byte-for-byte repeat mismatch for seed=0x{seed:08x}")
        fixtures[f"seeded_{suffix}"] = first
    fixtures["many_nodes"] = build_reference_all(
        many_config, many_routes, "cross-nodes", 23)
    fixtures["skew_empty"] = build_reference_all(
        skew_config, skew_routes, "cross-skew", 29)
    return fixtures


def first_difference(actual, expected, path="$" ):
    if type(actual) is not type(expected):
        return f"{path}: type {type(actual).__name__} != {type(expected).__name__}"
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            return f"{path}: keys {actual.keys()} != {expected.keys()}"
        for key in actual:
            difference = first_difference(actual[key], expected[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(actual, list):
        if len(actual) != len(expected):
            return f"{path}: length {len(actual)} != {len(expected)}"
        for index, (lhs, rhs) in enumerate(zip(actual, expected)):
            difference = first_difference(lhs, rhs, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if actual != expected:
        return f"{path}: {actual!r} != {expected!r}"
    return None


def main(path):
    with open(path, "r", encoding="utf-8") as stream:
        actual = json.load(stream)
    expected = expected_fixtures()
    difference = first_difference(actual, expected)
    if difference:
        raise AssertionError("C++/Python metadata mismatch: " + difference)
    handle_count = sum(len(handles) for handles in actual.values())
    print(f"[PASS] C++/Python full-field metadata cross-check: "
          f"{len(actual)} fixtures, {handle_count} rank-local handles")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: compare_reference.py CPP_METADATA_JSON")
    main(sys.argv[1])
