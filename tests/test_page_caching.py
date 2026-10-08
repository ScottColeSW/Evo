"""2026-10-08: the pages carry Cache-Control: no-cache, so a browser always revalidates them. Without it a browser kept an old index.html for hours and showed an old map."""
import asyncio

from aiohttp.test_utils import TestClient, TestServer

from backend.app import create_app


def test_every_page_tells_the_browser_to_revalidate():
    async def run():
        async with TestClient(TestServer(create_app())) as client:
            seen = {}
            for path in ("/", "/debug", "/about.html", "/books.html"):
                r = await client.get(path)
                seen[path] = (r.status, r.headers.get("Cache-Control"))
            return seen

    seen = asyncio.run(run())
    assert all(status == 200 and cache == "no-cache" for status, cache in seen.values()), seen
