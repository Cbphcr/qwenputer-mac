"""A local web page for talking to Qwenputer: the model (MacChat, default backend and chip
implementation) loads once; each question is one conversation with a freshly loaded firmware,
streamed to the page as it decodes (Server-Sent Events).

    .venv/bin/python -m macport.web /path/to/Qwen3.5-4B-Qwenputer [--port 8765]

then open http://localhost:8765. Standard library only (http.server); one generation at a time.
The stream tells Qwen's text from the circuit's (its echoes, e.g. the tool response) and shows
the runs of chip tokens between them. The model runs on the main thread (MLX's streams belong
to the thread that made them); the HTTP server's threads hand it jobs and stream its events.
"""

import argparse
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from chipformer.rv import bridge as BR
from chipformer.rv import elf, ref
from chipformer.rv import spec as S
from chipformer.rv.chat import CALC_TOOL, replay
from macport.chat import MacChat

PAGE = Path(__file__).with_name("web.html")


class App:
    def __init__(self, args):
        t0 = time.time()
        self.chat = MacChat(args.model, chip=args.chip, impl=args.impl, qwen_bits=args.qwen_bits)
        self.firmware = args.firmware
        self.config = {"impl": args.impl, "chip": args.chip, "qwen_bits": args.qwen_bits,
                       "firmware": Path(args.firmware).name,
                       "load_seconds": round(time.time() - t0, 1)}
        self.busy = threading.Lock()
        self.jobs = queue.Queue()

    def run_jobs(self) -> None:
        """The model's thread: each job is (converse's arguments, its event queue)."""
        while True:
            args, events = self.jobs.get()
            try:
                self.converse(*args, events.put)
            except Exception as e:  # report it on the page
                events.put({"k": "error", "error": f"{type(e).__name__}: {e}"})
            events.put(None)

    def machine(self):
        m = ref.Machine.from_program(elf.program(self.firmware))
        BR.park(m)
        return m

    def prompt_ids(self, message: str, tools: bool) -> list[int]:
        return self.chat.prompt_ids([{"role": "user", "content": message}],
                                    tools=(CALC_TOOL,) if tools else ())

    def converse(self, message: str, tools: bool, max_new: int, verify: bool, send) -> None:
        """One conversation, its tokens streamed through ``send(event)``."""
        chat, v0 = self.chat, self.chat.v0
        ids = self.prompt_ids(message, tools)
        lock = BR.Session(chat.T, self.machine())
        lock.history(ids)
        send({"k": "start", "prompt_tokens": len(ids)})
        text, sent, run = [], "", {"n": 0, "tail": [], "t": 0.0}

        def flush_run():
            if run["n"]:
                send({"k": "c", "n": run["n"], "tail": run["tail"], "end": True})
                run.update(n=0, tail=[])

        def on_token(t, owned):
            nonlocal sent
            if t >= v0:
                run["n"] += 1
                run["tail"] = (run["tail"] + [S.name(t - v0)])[-6:]
                now = time.time()
                if now - run["t"] > 0.05:
                    run["t"] = now
                    send({"k": "c", "n": run["n"], "tail": run["tail"]})
                return
            flush_run()
            text.append(t)
            s = chat.tok.decode(text, skip_special_tokens=False)
            if s.endswith("�"):  # an incomplete UTF-8 sequence: wait for the rest
                return
            if len(s) > len(sent):
                send({"k": "t", "s": s[len(sent):], "o": owned})
            sent = s

        r = chat.generate(ids, self.machine(), max_new, lock=lock, on_token=on_token)
        flush_run()
        done = {k: r[k] for k in ("text_tokens", "chip_tokens", "seconds", "prefill_seconds",
                                  "chip_seconds", "text_seconds", "lock_error")}
        done["lock_error"] = None if r["lock_error"] is None else str(r["lock_error"])
        send({"k": "done", **done})
        if verify:
            send({"k": "verify", **replay(chat, ids, r["tokens"], self.firmware)})


def handler(app: App):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                b = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            elif self.path == "/info":
                self._json(app.config)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path == "/prompt":
                q = self._body()
                ids = app.prompt_ids(q.get("message", ""), q.get("tools", True))
                return self._json({"tokens": len(ids),
                                   "text": app.chat.tok.decode(ids, skip_special_tokens=False)})
            if self.path != "/chat":
                return self._json({"error": "not found"}, 404)
            q = self._body()
            if not app.busy.acquire(blocking=False):
                return self._json({"error": "busy: one conversation at a time"}, 409)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                events = queue.Queue()
                app.jobs.put(((str(q.get("message", "")), bool(q.get("tools", True)),
                               min(int(q.get("max_new", 600)), 4000),
                               bool(q.get("verify", False))), events))
                gone = False
                while (ev := events.get()) is not None:  # (drain it even if the page left)
                    if gone:
                        continue
                    try:
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        gone = True
            finally:
                app.busy.release()

    return H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--firmware", default="upstream/experimental/riscv/tests/data/rv/calc-bridge.elf")
    ap.add_argument("--chip", default="neural", choices=["neural", "circuit", "verify"])
    ap.add_argument("--impl", default="mlx-freerun")
    ap.add_argument("--qwen-bits", type=int, default=None, choices=[4, 8])
    args = ap.parse_args()
    app = App(args)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[loaded in {app.config['load_seconds']}s] http://localhost:{args.port}", flush=True)
    app.run_jobs()


if __name__ == "__main__":
    main()
