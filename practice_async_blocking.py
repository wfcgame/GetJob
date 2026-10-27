"""Compare blocking and cooperative waits inside asyncio coroutines."""

import asyncio
import time


async def blocking_task() -> str:
    """Block the event-loop thread for one second."""
    time.sleep(1)
    return "blocking task completed"


async def non_blocking_task() -> str:
    """Yield control to the event loop while waiting for one second."""
    await asyncio.sleep(1)
    return "non-blocking task completed"


async def main() -> None:
    start = time.perf_counter()
    blocking_results = await asyncio.gather(*(blocking_task() for _ in range(3)))
    blocking_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    non_blocking_results = await asyncio.gather(
        *(non_blocking_task() for _ in range(3))
    )
    non_blocking_elapsed = time.perf_counter() - start

    print(f"Blocking results: {blocking_results}")
    print(f"Blocking elapsed: {blocking_elapsed:.3f} seconds")
    print(f"Non-blocking results: {non_blocking_results}")
    print(f"Non-blocking elapsed: {non_blocking_elapsed:.3f} seconds")
    print(f"Blocking / non-blocking time: {blocking_elapsed / non_blocking_elapsed:.2f}x")
    print()
    print("time.sleep() blocks the event-loop thread, so other tasks cannot run.")
    print("Even with gather(), these three blocking waits run sequentially (~3 seconds).")
    print("await asyncio.sleep() yields control, allowing the three waits to overlap (~1 second).")
    print("Declaring a function async does not make blocking operations non-blocking.")


if __name__ == "__main__":
    asyncio.run(main())
