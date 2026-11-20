"""Read physical mlx5 port counters on each member of the node's RoCE bonds."""

import ctypes
import fcntl
from pathlib import Path
import socket
import struct


def _ethtool(sock, name, request, extra=0):
    payload = ctypes.create_string_buffer(request, len(request) + extra)
    ifreq = struct.pack("@16sP", name.encode(), ctypes.addressof(payload)).ljust(40, b"\x00")
    fcntl.ioctl(sock.fileno(), 0x8946, ifreq)
    return bytes(payload)


class PhysicalCounters:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ports = {}
        for bond in sorted(Path("/sys/class/net").glob("bond[0-9]*/bonding/slaves")):
            for port in bond.read_text().split():
                info = _ethtool(self.sock, port, struct.pack("=IIQI", 0x37, 0, 2, 0))
                (count,) = struct.unpack("=I", info[-4:])
                strings = _ethtool(self.sock, port, struct.pack("=III", 0x1B, 1, count), count * 32)[12:]
                names = [
                    strings[i : i + 32].split(b"\x00", 1)[0].decode() for i in range(0, len(strings), 32)
                ]
                self.ports[port] = count, names.index("tx_bytes_phy"), names.index("rx_bytes_phy")
        if not self.ports:
            raise RuntimeError("No physical RoCE bond members were found")

    def read(self):
        counters = {}
        for port, (count, tx, rx) in self.ports.items():
            payload = _ethtool(self.sock, port, struct.pack("=II", 0x1D, count), count * 8)
            counters[port] = [struct.unpack_from("=Q", payload, 8 + i * 8)[0] for i in (tx, rx)]
        return counters
