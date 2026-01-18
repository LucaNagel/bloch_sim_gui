import http.server
import socketserver
import os
import shutil
import glob
import sys
import webbrowser

# Configuration
PORT = 8000
DEV_DIR = "_dev"
WEB_DIR = "web"
DIST_DIR = "dist"


def build_site():
    print(f"Building site into '{DEV_DIR}'...")

    # 1. Clean/Create Dev Directory
    if os.path.exists(DEV_DIR):
        shutil.rmtree(DEV_DIR)
    os.makedirs(DEV_DIR)

    # 2. Build HTML (Concat Partials)
    print("  Compiling index.html...")
    partials_dir = os.path.join(WEB_DIR, "partials")
    parts = ["header.html", "home.html", "rf_explorer.html", "footer.html"]
    content = ""
    for part in parts:
        path = os.path.join(partials_dir, part)
        if os.path.exists(path):
            with open(path, "r") as f:
                content += f.read() + "\n"
        else:
            print(f"  [WARN] Partial {part} not found")

    with open(os.path.join(DEV_DIR, "index.html"), "w") as f:
        f.write(content)

    # 3. Copy Static Assets
    print("  Copying static assets...")
    shutil.copytree(os.path.join(WEB_DIR, "static"), os.path.join(DEV_DIR, "static"))

    # 4. Handle Wheel File (Optional)
    # Check if a .whl exists in dist/ (e.g. from a previous build)
    wheel_files = glob.glob(os.path.join(DIST_DIR, "*.whl"))
    js_path = os.path.join(DEV_DIR, "static", "js", "app.js")

    with open(js_path, "r") as f:
        js_content = f.read()

    if wheel_files:
        wheel_path = wheel_files[0]
        wheel_name = os.path.basename(wheel_path)
        print(f"  Found local wheel: {wheel_name}")
        shutil.copy(wheel_path, os.path.join(DEV_DIR, wheel_name))

        # Update JS to use this wheel
        js_content = js_content.replace("WHEEL_FILE_PLACEHOLDER", wheel_name)
    else:
        print("  No wheel file found in dist/. App will run in MOCK MODE.")
        # We leave the placeholder or set it to something harmless
        # The JS has a try/catch block that handles missing wheels by enabling mocks.

    with open(js_path, "w") as f:
        f.write(js_content)


def run_server():
    os.chdir(DEV_DIR)
    Handler = http.server.SimpleHTTPRequestHandler
    # Allow WASM MIME type (important for some browsers)
    Handler.extensions_map[".wasm"] = "application/wasm"

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}"
        print(f"\nServing at {url}")
        print("Press Ctrl+C to stop")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping server...")


if __name__ == "__main__":
    build_site()
    run_server()
