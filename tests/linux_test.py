"""Linux paths, reversible removal, engine startup and a real HTTP startup (no downloads/models)."""
import http.server
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
TMP = tempfile.TemporaryDirectory(prefix="survival-linux-test-")
os.environ.update(SURVIVAL_LLM_APP_DIR=str(Path(TMP.name) / "settings"),
                  SURVIVAL_LLM_LIBRARY=str(Path(TMP.name) / "library"),
                  SURVIVAL_LLM_FILES=str(Path(TMP.name) / "files"))
import server


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux integration checks")
class LinuxTests(unittest.TestCase):
    def test_data_paths(self):
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, {"HOME": home}, clear=True):
            expected = os.path.join(home, ".local", "share", "survival-llm")
            self.assertEqual(server.search.app_dir(), expected)
            self.assertEqual(server.search.default_library(), os.path.join(expected, "Library"))
            with patch.dict(os.environ, {"XDG_DATA_HOME": home + "/data"}):
                self.assertEqual(server.search.app_dir(), home + "/data/survival-llm")
            with patch.dict(os.environ, {"XDG_DATA_HOME": "relative"}):
                self.assertEqual(server.search.app_dir(), expected)
            os.mkdir(home + "/.survival-llm")
            self.assertEqual(server.search.app_dir(), home + "/.survival-llm")
            with patch.dict(os.environ, {"SURVIVAL_LLM_APP_DIR": home + "/custom"}):
                self.assertEqual(server.search.app_dir(), home + "/custom")
        self.assertEqual(server.APP_DIR, server.search.app_dir())

    def test_trash_never_deletes_on_failure(self):
        path = Path(TMP.name) / "keep.txt"
        path.write_text("keep me", encoding="utf-8")
        trash = types.SimpleNamespace(send2trash=lambda _: (_ for _ in ()).throw(OSError("no trash")))
        with patch.dict(sys.modules, {"send2trash": trash}):
            with self.assertRaises(OSError):
                server.to_trash(str(path))
        self.assertEqual(path.read_text(), "keep me")
        with patch.dict(sys.modules, {"send2trash": None}):
            with self.assertRaises(OSError):
                server.to_trash(str(path))
        self.assertTrue(path.exists())

    def test_ollama_from_path(self):
        with patch.object(server.shutil, "which", return_value=sys.executable):
            self.assertEqual(server.ollama_locations(), (None, sys.executable))
            with patch.object(server.subprocess, "Popen") as popen:
                server.ollama_started_at[0] = 0
                server.start_ollama()
                self.assertEqual(popen.call_args.args[0], [sys.executable, "serve"])
                self.assertTrue(popen.call_args.kwargs["start_new_session"])
                self.assertNotIn("creationflags", popen.call_args.kwargs)

    def test_server_startup(self):
        class Ollama(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                data = {"/api/tags": {"models": [{"name": "granite4.2:8b"}]},
                        "/api/version": {"version": "0.40.11"}, "/api/ps": {"models": []}}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(data.get(self.path, {})).encode())

            def log_message(self, *args):
                pass

        import urllib.request
        with tempfile.TemporaryDirectory() as folder:
            fake = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Ollama)
            threading.Thread(target=fake.serve_forever, daemon=True).start()
            env = {**os.environ, "SURVIVAL_LLM_APP_DIR": folder,
                   "OLLAMA_HOST": f"127.0.0.1:{fake.server_port}",
                   "SURVIVAL_LLM_UPDATES": Path(folder, "absent-update.json").as_uri(),
                   "SURVIVAL_LLM_MANIFEST": Path(folder, "absent-library.json").as_uri()}
            proc = subprocess.Popen([sys.executable, str(ROOT / "app/server.py"), "--no-browser", "--port", "0"],
                                    env=env, cwd="/", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                state = Path(folder, "server.json")
                deadline = time.monotonic() + 10
                while not state.exists() and proc.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(state.exists(), "server did not start")
                port = json.loads(state.read_text())["port"]
                base = f"http://127.0.0.1:{port}"
                with urllib.request.urlopen(base) as response:
                    page = response.read().decode()
                token = re.search(r'<meta name="token" content="([^"]+)">', page)[1]
                request = urllib.request.Request(base + "/api/status", headers={"X-Token": token})
                with urllib.request.urlopen(request) as response:
                    status = json.load(response)
                self.assertEqual(status["machine"]["os"], "Linux")
                self.assertTrue(status["ollama"])
                self.assertEqual(status["default_model"], "granite4.2:8b")
                self.assertTrue(status["catalog"])
                with urllib.request.urlopen(base + "/static/app.js") as response:
                    self.assertEqual(response.status, 200)
            finally:
                proc.terminate()
                out, err = proc.communicate(timeout=10)
                fake.shutdown()
                fake.server_close()
            self.assertEqual(proc.returncode, 0, out + err)
            self.assertFalse(state.exists(), "shutdown should clear the saved instance")


if __name__ == "__main__":
    unittest.main()
