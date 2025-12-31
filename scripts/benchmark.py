import time
from statistics import mean

def benchmark(pipeline, questions):
    times = []
    for q in questions:
        start = time.perf_counter()
        pipeline.ask(q)
        times.append(time.perf_counter() - start)

    print(f"Avg latency: {mean(times):.2f}s")
    print(f"Min: {min(times):.2f}s | Max: {max(times):.2f}s")
