import asyncio
import functools


def run_async(fn):
    """Lets an async test function run under plain pytest, with no pytest-asyncio
    dependency -- just wraps it in asyncio.run()."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        asyncio.run(fn(*args, **kwargs))

    return wrapper
