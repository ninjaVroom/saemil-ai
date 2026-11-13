"""Stand-in CLI for tests: `python fake_cli.py MODE [ARG...]`. The prompt arrives on stdin only."""
import json
import os
import sys
import time

mode = sys.argv[1]
prompt = sys.stdin.read()
if mode == "echo":
    sys.stdout.write(json.dumps({"argv": sys.argv[1:], "stdin": prompt, "cwd": os.getcwd(),
                                 "cwd_files": os.listdir(".")}))
elif mode == "slow":  # holds the CLI slot for a moment, then answers
    time.sleep(1.5)
    sys.stdout.write("slow but fine")
elif mode == "sleep":  # sleeps past the test timeout; the marker proves whether it was killed
    time.sleep(3.0)
    open(os.environ["FAKE_CLI_MARKER"], "w").close()
    sys.stdout.write("too late")
elif mode == "fail":
    sys.stdout.write("partial")
    sys.exit(3)
elif mode == "empty":
    sys.stdout.write("  \n")
elif mode == "big":  # exactly N bytes, then exit
    sys.stdout.write("x" * int(sys.argv[2]))
elif mode == "spew":  # never stops writing; must be killed for size well before the timeout
    while True:
        sys.stdout.write("y" * 4096)
        sys.stdout.flush()
