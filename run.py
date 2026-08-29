from aiohttp import web

from backend.app import create_app

if __name__ == "__main__":
    print("Evo-LLM-Evolution2Civ starting on http://localhost:8765")
    web.run_app(create_app(), host="localhost", port=8765)
