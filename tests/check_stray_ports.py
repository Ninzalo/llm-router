#!/usr/bin/env python3
"""Fail if any 18xxx test listener survived the suite.

Portable on purpose: uses a TCP connect probe (works on Linux and macOS), so it
does not depend on lsof/ss/netstat, which are not all present on every runner.
The test suites only ever bind 18xxx ports, so scanning that range is complete.
"""
import socket
import sys


def leaked_18xxx() -> list[int]:
    found = []
    for port in range(18000, 19000):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.05)
        try:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                found.append(port)
        except OSError:
            pass
        finally:
            s.close()
    return found


def main() -> int:
    leaked = leaked_18xxx()
    if leaked:
        print("::error::18xxx test listeners survived the suite:", leaked)
        return 1
    print("No stray 18xxx test listeners.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
