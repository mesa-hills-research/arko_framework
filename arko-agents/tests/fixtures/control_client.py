"""Portable request/reply client used by the terminal integration fixtures.

One writer per stream, complete UTF-8 JSON lines, flush after each request.
The reader retains partial lines; empty reads mean no new bytes yet.
"""
import os, sys, time, json


class Replies:
    def __init__(self):
        self.file = open(os.environ["ARKO_INPUT_REPLY_PATH"], "rb", buffering=0)
        self.pending = b""

    def readline(self):
        while b"\n" not in self.pending:
            chunk = self.file.read(8192)
            if chunk:
                self.pending += chunk
            else:
                time.sleep(0.02)
        line, self.pending = self.pending.split(b"\n", 1)
        return line.decode("utf-8")
