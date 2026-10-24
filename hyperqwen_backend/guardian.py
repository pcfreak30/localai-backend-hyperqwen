"""Standalone stdlib process guardian: adapter death must not orphan HQ workers.

Runtime starts this process as a new session/group leader. The native launcher
and workers inherit its group; no unsafe threaded preexec_fn is used.
"""
import os
import signal
import subprocess
import sys
import time


def main(parent, command):
    if os.getpgrp() != os.getpid():
        raise RuntimeError('guardian requires its own dedicated process group')
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    # Cover adapter death between Popen and guardian initialization.
    if os.getppid() != parent:
        return 1
    child = subprocess.Popen(command)
    try:
        while not stopping and os.getppid() == parent and child.poll() is None:
            time.sleep(0.1)
    finally:
        # Ignore our forwarded signal, let HQ shut down, then remove every group
        # member even if the launcher exited before its workers. SIGKILL also
        # ends this guardian; the live adapter reaps it, or PID 1 adopts it.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        os.killpg(os.getpgrp(), signal.SIGTERM)
        try:
            child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            pass
        os.killpg(os.getpgrp(), signal.SIGKILL)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(int(sys.argv[1]), sys.argv[2:]))
