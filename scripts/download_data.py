# -*- coding: utf-8 -*-
"""Threaded ranged downloader for MSD Task05 Prostate (validates each part size)."""
import os
import threading
import time
import urllib.request

URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "Task05_Prostate.tar")
PART_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "parts")
SIZE = 239839232
N = 16
CHUNK = (SIZE + N - 1) // N
os.makedirs(PART_DIR, exist_ok=True)

lock = threading.Lock()
done_bytes = [0]


def worker(i):
    s = i * CHUNK
    e = min(s + CHUNK, SIZE) - 1
    want = e - s + 1
    part = os.path.join(PART_DIR, f"p{i:02d}")
    for attempt in range(8):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={s}-{e}"})
            got = 0
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    got += len(b)
                    with lock:
                        done_bytes[0] += len(b)
            if os.path.getsize(part) == want:
                return
            print(f"[p{i:02d}] size {os.path.getsize(part)} != {want}, retrying", flush=True)
        except Exception as ex:
            print(f"[p{i:02d}] attempt {attempt}: {ex}", flush=True)
            time.sleep(3)
    raise RuntimeError(f"part {i} failed")


threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(N)]
t0 = time.time()
for t in threads:
    t.start()
while any(t.is_alive() for t in threads):
    time.sleep(10)
    el = time.time() - t0
    print(f"progress: {done_bytes[0]}/{SIZE} ({100*done_bytes[0]/SIZE:.1f}%), {done_bytes[0]/el/1e6:.2f} MB/s", flush=True)
for t in threads:
    t.join()

with open(OUT, "wb") as out:
    for i in range(N):
        with open(os.path.join(PART_DIR, f"p{i:02d}"), "rb") as f:
            while True:
                b = f.read(1 << 22)
                if not b:
                    break
                out.write(b)
assert os.path.getsize(OUT) == SIZE, os.path.getsize(OUT)
print("DONE", os.path.getsize(OUT), "bytes in", round(time.time() - t0), "s", flush=True)
