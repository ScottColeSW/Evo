import argparse

from aiohttp import web

from backend.app import create_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"Evo-LLM-Evolution2Civ starting on http://{args.host}:{args.port}")
    web.run_app(create_app(), host=args.host, port=args.port)
