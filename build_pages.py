"""Generate a static GitHub Pages build that runs the app in the browser.

GitHub Pages serves static files only, so it cannot run Python. stlite works
around that by running Streamlit inside WebAssembly (Pyodide) in the visitor's
browser -- no server, no install, just a URL.

The trade-offs are real and are surfaced in the page itself: everything is
downloaded and parsed client-side, so start-up takes a while and very large
logs will be slow or exhaust browser memory. For heavy work, run the app
locally or on Streamlit Community Cloud.

Usage:  python build_pages.py   ->  writes docs/index.html
"""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "docs"
STLITE = "1.8.1"

# Pure-Python wheels resolved by micropip at load time. pandas and numpy ship
# with Pyodide itself. streamlit-folium is deliberately absent: it is a custom
# component and cannot work here, which is why the app renders its map as HTML.
REQUIREMENTS = ["plotly", "folium", "branca"]

APP_FILES = ["app.py"] + sorted(
    str(p.relative_to(ROOT)) for p in (ROOT / "boatviz").glob("*.py"))

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Boat Log Explorer</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 32 32%22><text y=%2226%22 font-size=%2226%22>⛵</text></svg>"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/browser@{stlite}/build/stlite.css"/>
<style>
  html, body {{ margin:0; height:100%; }}
  #boot {{
    position:fixed; inset:0; display:grid; place-content:center; gap:14px;
    font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    text-align:center; padding:32px; background:#fbfbfa; color:#37352f; z-index:9;
  }}
  #boot h1 {{ font-size:20px; margin:0; }}
  #boot p {{ margin:0; max-width:44ch; color:#6b6a65; }}
  #boot .bar {{ width:220px; height:4px; border-radius:2px; background:#e1e0d9;
                overflow:hidden; margin:4px auto 0; }}
  #boot .bar i {{ display:block; width:40%; height:100%; background:#2a78d6;
                  animation:slide 1.1s ease-in-out infinite; }}
  @keyframes slide {{ 0%{{transform:translateX(-100%)}} 100%{{transform:translateX(250%)}} }}
  @media (prefers-color-scheme: dark) {{
    #boot {{ background:#0e1117; color:#e6e6e3; }}
    #boot p {{ color:#9b9a94; }}
    #boot .bar {{ background:#2c2c2a; }}
  }}
</style>
</head>
<body>
<div id="boot">
  <h1>⛵ Boat Log Explorer</h1>
  <p>Starting Python in your browser. The first load downloads about 30 MB of
     runtime and takes roughly a minute; afterwards it is cached.</p>
  <div class="bar"><i></i></div>
</div>
<div id="root"></div>
<script type="module">
import {{ mount }} from "https://cdn.jsdelivr.net/npm/@stlite/browser@{stlite}/build/stlite.js";
mount({{
  requirements: {requirements},
  entrypoint: "app.py",
  files: {files},
  streamlitConfig: {{ "client.toolbarMode": "minimal" }},
}}, document.getElementById("root"));

// The boot card sits above the app until Streamlit paints something.
const boot = document.getElementById("boot");
const root = document.getElementById("root");
new MutationObserver((_m, obs) => {{
  if (root.querySelector('[data-testid="stAppViewContainer"]')) {{
    boot.style.transition = "opacity .4s"; boot.style.opacity = "0";
    setTimeout(() => boot.remove(), 400); obs.disconnect();
  }}
}}).observe(root, {{ childList: true, subtree: true }});
</script>
</body>
</html>
"""


def build():
    files = {rel: {"data": (ROOT / rel).read_text(encoding="utf-8")}
             for rel in APP_FILES}

    OUT.mkdir(exist_ok=True)
    (OUT / ".nojekyll").write_text("")

    # Sample logs are copied beside the page and referenced by URL rather than
    # inlined, so index.html stays small and the browser can cache the CSVs
    # independently of the code.
    sample_dir = OUT / "sample_logs"
    sample_dir.mkdir(exist_ok=True)
    for csv in sorted((ROOT / "sample_logs").glob("*.csv")):
        shutil.copy2(csv, sample_dir / csv.name)
        files[f"sample_logs/{csv.name}"] = {"url": f"./sample_logs/{csv.name}"}

    (OUT / "index.html").write_text(
        PAGE.format(requirements=json.dumps(REQUIREMENTS),
                    files=json.dumps(files), stlite=STLITE),
        encoding="utf-8")

    kb = (OUT / "index.html").stat().st_size / 1024
    mb = sum(f.stat().st_size for f in sample_dir.glob("*.csv")) / 1e6
    print(f"docs/index.html      {kb:6.0f} KB  ({len(APP_FILES)} python files inlined)")
    print(f"docs/sample_logs/    {mb:6.1f} MB  "
          f"({len(list(sample_dir.glob('*.csv')))} logs, fetched on demand)")


if __name__ == "__main__":
    build()
