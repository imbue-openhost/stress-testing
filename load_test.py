#!/usr/bin/env python3
"""
Load test for llama.cpp /infill endpoint.

Simulates multiple concurrent users typing code and requesting completions.
Ramps up concurrency and reports latency/throughput at each level.

Usage:
    python load_test.py --endpoint https://code-completion.example.com --token YOUR_TOKEN
    python load_test.py --endpoint https://code-completion.example.com --token YOUR_TOKEN --max-users 20 --requests-per-user 10
"""

import argparse
import asyncio
import json
import ssl
import statistics
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import aiohttp

# Varied prompts to avoid cache hits giving unrealistic results
PROMPTS = [
    {"input_prefix": "def fibonacci(n):\n    ", "input_suffix": "\n\nprint(fibonacci(10))"},
    {"input_prefix": "class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n        ", "input_suffix": "\n\n    def pop(self):\n        return self.items.pop()"},
    {"input_prefix": "import os\nimport sys\n\ndef read_config(path):\n    ", "input_suffix": "\n\nconfig = read_config('settings.json')"},
    {"input_prefix": "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    ", "input_suffix": "\n\nprint(merge_sort([3, 1, 4, 1, 5, 9]))"},
    {"input_prefix": "from dataclasses import dataclass\n\n@dataclass\nclass User:\n    name: str\n    email: str\n\n    def ", "input_suffix": "\n\nuser = User('Alice', 'alice@example.com')"},
    {"input_prefix": "async def fetch_data(url):\n    ", "input_suffix": "\n\nasyncio.run(fetch_data('https://api.example.com'))"},
    {"input_prefix": "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    ", "input_suffix": "\n\nresult = binary_search([1, 3, 5, 7, 9], 5)"},
    {"input_prefix": "import json\nfrom pathlib import Path\n\ndef save_results(data, output_dir):\n    ", "input_suffix": "\n\nsave_results({'score': 95}, '/tmp/results')"},
]


@dataclass
class RequestResult:
    latency: float
    status: int
    tokens_predicted: int
    error: str | None = None


@dataclass
class LevelResult:
    concurrency: int
    results: list[RequestResult] = field(default_factory=list)

    @property
    def successes(self) -> list[RequestResult]:
        return [r for r in self.results if r.error is None and r.status == 200]

    @property
    def failures(self) -> list[RequestResult]:
        return [r for r in self.results if r.error is not None or r.status != 200]

    def summary(self) -> dict:
        latencies = [r.latency for r in self.successes]
        tokens = [r.tokens_predicted for r in self.successes]
        if not latencies:
            return {"concurrency": self.concurrency, "error": "all requests failed"}
        return {
            "concurrency": self.concurrency,
            "total_requests": len(self.results),
            "successes": len(self.successes),
            "failures": len(self.failures),
            "latency_p50": round(statistics.median(latencies), 2),
            "latency_p90": round(sorted(latencies)[int(len(latencies) * 0.9)], 2),
            "latency_p99": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
            "latency_min": round(min(latencies), 2),
            "latency_max": round(max(latencies), 2),
            "latency_mean": round(statistics.mean(latencies), 2),
            "avg_tokens": round(statistics.mean(tokens), 1) if tokens else 0,
            "throughput_rps": round(len(self.successes) / sum(latencies) * len(latencies), 2) if latencies else 0,
        }


async def send_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    token: str,
    prompt_idx: int,
    max_tokens: int,
) -> RequestResult:
    prompt = PROMPTS[prompt_idx % len(PROMPTS)]
    body = {
        "input_prefix": prompt["input_prefix"],
        "input_suffix": prompt["input_suffix"],
        "n_predict": max_tokens,
        "stream": False,
        "cache_prompt": True,
    }
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = endpoint.rstrip("/") + "/infill"
    start = time.monotonic()
    try:
        async with session.post(url, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            data = await resp.json()
            elapsed = time.monotonic() - start
            return RequestResult(
                latency=round(elapsed, 3),
                status=resp.status,
                tokens_predicted=data.get("tokens_predicted", 0),
            )
    except Exception as e:
        elapsed = time.monotonic() - start
        return RequestResult(latency=round(elapsed, 3), status=0, tokens_predicted=0, error=str(e))


async def simulate_user(
    session: aiohttp.ClientSession,
    endpoint: str,
    token: str,
    user_id: int,
    num_requests: int,
    max_tokens: int,
    think_time: float,
) -> list[RequestResult]:
    """Simulate a user making requests with pauses between them."""
    results = []
    for i in range(num_requests):
        result = await send_request(session, endpoint, token, user_id * num_requests + i, max_tokens)
        results.append(result)
        if i < num_requests - 1:
            await asyncio.sleep(think_time)
    return results


async def run_level(
    endpoint: str,
    token: str,
    concurrency: int,
    requests_per_user: int,
    max_tokens: int,
    think_time: float,
) -> LevelResult:
    """Run a load test at a specific concurrency level."""
    ssl_ctx = ssl.create_default_context()
    connector = aiohttp.TCPConnector(ssl=ssl_ctx, limit=concurrency + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            simulate_user(session, endpoint, token, i, requests_per_user, max_tokens, think_time)
            for i in range(concurrency)
        ]
        all_results = await asyncio.gather(*tasks)

    level = LevelResult(concurrency=concurrency)
    for user_results in all_results:
        level.results.extend(user_results)
    return level


async def run_load_test(
    endpoint: str,
    token: str,
    max_users: int,
    requests_per_user: int,
    max_tokens: int,
    think_time: float,
    step: int,
):
    levels = list(range(1, max_users + 1, step))
    if levels[-1] != max_users:
        levels.append(max_users)

    print(f"Load test: {endpoint}")
    print(f"  Max users: {max_users}, Requests/user: {requests_per_user}, Max tokens: {max_tokens}")
    print(f"  Think time: {think_time}s, Step: {step}")
    print()
    print(f"{'Users':>5} | {'Reqs':>5} | {'OK':>4} | {'Fail':>4} | {'p50':>7} | {'p90':>7} | {'p99':>7} | {'Mean':>7} | {'Max':>7} | {'Tok':>5} | {'RPS':>6}")
    print("-" * 90)

    all_summaries = []
    for n in levels:
        level = await run_level(endpoint, token, n, requests_per_user, max_tokens, think_time)
        s = level.summary()
        all_summaries.append(s)

        if "error" in s:
            print(f"{n:>5} | {'ALL FAILED':>50}")
        else:
            print(
                f"{s['concurrency']:>5} | "
                f"{s['total_requests']:>5} | "
                f"{s['successes']:>4} | "
                f"{s['failures']:>4} | "
                f"{s['latency_p50']:>6}s | "
                f"{s['latency_p90']:>6}s | "
                f"{s['latency_p99']:>6}s | "
                f"{s['latency_mean']:>6}s | "
                f"{s['latency_max']:>6}s | "
                f"{s['avg_tokens']:>5} | "
                f"{s['throughput_rps']:>5}"
            )

        # Brief pause between levels
        if n != levels[-1]:
            await asyncio.sleep(2)

    print()
    print("Summary:")
    for s in all_summaries:
        if "error" not in s:
            usable = "YES" if s["latency_p90"] < 3.0 else ("MARGINAL" if s["latency_p90"] < 5.0 else "NO")
            print(f"  {s['concurrency']} concurrent users: p90={s['latency_p90']}s, mean={s['latency_mean']}s -> {usable}")


def main():
    parser = argparse.ArgumentParser(description="Load test a llama.cpp /infill endpoint")
    parser.add_argument("--endpoint", required=True, help="Server URL")
    parser.add_argument("--token", default="", help="API bearer token")
    parser.add_argument("--max-users", type=int, default=10, help="Max concurrent users to test (default: 10)")
    parser.add_argument("--requests-per-user", type=int, default=5, help="Requests per user per level (default: 5)")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens per completion (default: 64)")
    parser.add_argument("--think-time", type=float, default=2.0, help="Seconds between requests per user (default: 2.0)")
    parser.add_argument("--step", type=int, default=2, help="Increment users by this many per level (default: 2)")
    args = parser.parse_args()

    asyncio.run(run_load_test(
        endpoint=args.endpoint,
        token=args.token,
        max_users=args.max_users,
        requests_per_user=args.requests_per_user,
        max_tokens=args.max_tokens,
        think_time=args.think_time,
        step=args.step,
    ))


if __name__ == "__main__":
    main()
