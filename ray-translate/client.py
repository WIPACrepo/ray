#!/usr/bin/env python3
import requests
import argparse
import asyncio
import aiohttp
import aiofiles

async def post_request(session: aiohttp.ClientSession, url: str, payload: dict) -> dict:
    """Send a single POST request and return the result."""
    async with session.post(url, json=payload) as response:
        response.raise_for_status()
        return await response.json()

async def process_chunk(session: aiohttp.ClientSession, url: str, chunk: list[dict]) -> list[dict]:
    """Process a chunk of payloads concurrently using a TaskGroup."""
    results = []
    async with asyncio.TaskGroup() as tg:
        tasks = [
            tg.create_task(post_request(session, url, payload)) for payload in chunk
        ]

    results = [task.result() for task in tasks]
    return results

async def chunked_post_requests(
    url: str,
    payloads: list[dict],
    chunk_size: int = 10,
) -> list[dict]:
    """
    Send POST requests in chunks, collecting all results into a list.

    Args:
        url:        The endpoint to POST to.
        payloads:   List of request bodies to send.
        chunk_size: How many requests to fire concurrently per chunk.

    Returns:
        Flat list of response dicts in the same order as `payloads`.
    """
    all_results: list[dict] = []
    chunks = [payloads[i:i + chunk_size] for i in range(0, len(payloads), chunk_size)]

    async with aiohttp.ClientSession() as session:
        for chunk in chunks:
            chunk_results = await process_chunk(session, url, chunk)
            all_results.extend(chunk_results)

    return all_results

async def main(args):
    async with aiofiles.open(args.filename) as f:
        url = "https://raytest.icecube.aq/predict"
        payloads = []

        token_chunk = []
        async for line in f:
            for word in line.split():
                token_chunk.append(word)
                print (len(token_chunk))
                if len(token_chunk) >= round(args.token_size * 0.9, 0) and "." in word:
                    tokens = " ".join(map(str, token_chunk))
                    token_chunk = []
                    payloads.append({"fr": args.fr, "to": args.to, "content": tokens})

        if len(token_chunk) != 0:
            tokens = " ".join(map(str, token_chunk))
            token_chunk = []
            payloads.append({"fr": args.fr, "to": args.to, "content": tokens})

        print(payloads)
        results = await chunked_post_requests(url, payloads, chunk_size=args.chunk_size)

        print(f"Total responses: {len(results)}")
        for r in results:
            print(r)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("filename")
    parser.add_argument("--fr",
                        type=str,
                        default="English")
    parser.add_argument("--to",
                        type=str,
                        default="French")
    parser.add_argument("--token_size",
                        type=int,
                        default=0,
                        help="how many tokens in a chunk")
    parser.add_argument("--chunk_size",
                        type=int,
                        default=1,
                        help="how many max token chunks to process at a time")

    args = parser.parse_args()

    asyncio.run(main(args))