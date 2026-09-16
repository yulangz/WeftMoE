"""Exercise source lifetime under delayed claims and unordered RX completion."""

import random
import unittest


class DispatchReceiveTicketsTest(unittest.TestCase):
    def exercise(self, depth, packet_rows, part_rows, seed):
        rng = random.Random(seed)
        parts = (packet_rows + part_rows - 1) // part_rows
        durable = 0
        for _ in range(4):
            sizes = [rng.randrange(packet_rows + 1) for _ in range(rng.randrange(1, 40))]
            sent = ticket = 0
            claimed = durable
            payload, readers, seen = {}, {}, {}
            pending, completed, acknowledgements = [], [0] * depth, set()
            # Completion and credit delivery are independent. In particular,
            # an idle consumer must not make its source slot reusable early.
            credits = []
            for _step in range(100000):
                if len(acknowledgements) == len(sizes):
                    break
                operation = rng.randrange(4)
                if operation == 0 and sent < len(sizes):
                    sequence = durable + sent
                    slot = sequence % depth
                    if slot not in payload:
                        payload[slot] = (sequence, sizes[sent])
                        sent += 1
                elif operation == 1:
                    sequence = durable + ticket // parts
                    slot = sequence % depth
                    if len(pending) < 32 and payload.get(slot, (None,))[0] == sequence:
                        pending.append((ticket, sequence, slot))
                    if pending and (len(pending) == 32 or rng.random() < .8):
                        observed, sequence, slot = pending.pop(rng.randrange(len(pending)))
                        # A losing, arbitrarily delayed CAS must not read a header.
                        if observed != ticket:
                            continue
                        ticket += 1
                        part = observed % parts
                        self.assertEqual(payload[slot][0], sequence)
                        begin = min(part * part_rows, payload[slot][1])
                        end = min(begin + part_rows, payload[slot][1])
                        self.assertNotIn((sequence, part), readers)
                        self.assertNotIn((sequence, part), seen)
                        readers[sequence, part] = (slot, begin, end)
                        if part + 1 == parts:
                            claimed = max(claimed, sequence + 1)
                elif operation == 2 and readers:
                    key = rng.choice(list(readers))
                    slot = readers[key][0]
                    if completed[slot] % parts == parts - 1 and len(credits) == 4:
                        continue
                    slot, begin, end = readers.pop(key)
                    sequence, _part = key
                    self.assertEqual(payload[slot][0], sequence,
                                     "slot reused while a TMA reader was live")
                    seen[key] = (begin, end)
                    completed[slot] += 1
                    if completed[slot] % parts == 0:
                        self.assertFalse(any(key[0] == sequence for key in readers))
                        rows = [row for part in range(parts)
                                for row in range(*seen[sequence, part])]
                        self.assertEqual(rows, list(range(sizes[sequence - durable])))
                        self.assertNotIn((sequence, slot), credits)
                        credits.append((sequence, slot))
                elif operation == 3 and credits:
                    # Different contexts may deliver credits in any order.
                    sequence, slot = credits.pop(rng.randrange(len(credits)))
                    self.assertEqual(payload[slot][0], sequence)
                    self.assertFalse(any(key[0] == sequence for key in readers))
                    self.assertNotIn(sequence, acknowledgements)
                    acknowledgements.add(sequence)
                    del payload[slot]
            else:
                self.fail("receive protocol failed to drain")
            self.assertEqual(ticket, len(sizes) * parts)
            self.assertEqual(claimed, durable + len(sizes))
            self.assertFalse(readers)
            self.assertFalse(credits)
            self.assertFalse(payload)
            durable = claimed

    def test_reuse_short_packets_empty_packets_and_delayed_claims(self):
        for depth in (4, 8, 16):
            for packet_rows in (7, 64, 128):
                for part_rows in (3, 16, 32, 128):
                    for seed in range(8):
                        with self.subTest(depth=depth, packet_rows=packet_rows,
                                          part_rows=part_rows, seed=seed):
                            self.exercise(depth, packet_rows, part_rows, seed)


if __name__ == "__main__":
    unittest.main()
