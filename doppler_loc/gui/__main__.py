"""Start the GUI: uv run python -m doppler_loc.gui [--port 8765]"""

import argparse

import uvicorn

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--port", type=int, default=8765)
args = parser.parse_args()
print(f"open http://127.0.0.1:{args.port}")
uvicorn.run("doppler_loc.gui.server:app", host="127.0.0.1", port=args.port, log_level="warning")
