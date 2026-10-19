"""Seminar demo server.  Usage:  python app.py --artifacts artifacts   ->  http://127.0.0.1:8000
Needs only: torch, numpy, scipy (same as the notebook). No web framework.
Matches transformer_embedding_from_scratch_improved.ipynb (title-prefixed index, stop-word BM25, z-score hybrid).

DEBUGGING BUILT IN (read the terminal output):
  * every startup stage is printed with a timestamp; if startup stalls, a stack trace is printed every 20 s showing the stuck line
  * after start-up the server tests ITSELF over 127.0.0.1 and localhost and prints the result
  * every connection / request / response is printed; requests stuck for >8 s trigger a thread-stack dump
  * http://127.0.0.1:8000/test   -> tiny diagnostic page (works even if the main page does not)
  * http://127.0.0.1:8000/debug  -> JSON status;   /ping -> "ok";   /slow?s=12 -> fake slow request to try the watchdog
"""
import sys, time, faulthandler
T0 = time.time(); LOG = []
def log(msg):
    line = f"[{time.time() - T0:6.1f}s] {msg}"; LOG.append(line); del LOG[:-300]; print(line, flush=True)
faulthandler.enable(); faulthandler.dump_traceback_later(20, repeat=True)        # while starting up: every 20 s print where we are stuck
log(f"python {sys.version.split()[0]} on {sys.platform}; importing numpy / scipy / torch (first import can take 10-60 s) ...")
import argparse, json, math, os, threading, webbrowser, urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np;                           log(f"numpy {np.__version__} imported")
import torch, torch.nn.functional as F;       log(f"torch {torch.__version__} imported")
from scipy.sparse import csr_matrix;          log("scipy imported")
from lib import (
    BPETokenizer, Encoder, pretokenize,
    pad_batch, mean_pool, PAD, N_SPECIAL
);                                            log("lib.py imported")

# --- BM25 identical to the notebook's (ignores stop-words). Defined here so lib.py needs no change. ---
STOP = set("a an the of in on at to for from by with and or is are was were be been being what which who whom whose when where why how "
           "did does do has have had that this these those it its as into than then also".split())
bm_tokens = lambda t: [w for w in pretokenize(t) if w not in STOP]

class BM25:
    def __init__(s, docs, k1=1.5, b=0.75):
        toks = [bm_tokens(d) for d in docs]; s.vocab = {}; r, c, v = [], [], []; dl = np.array([max(1, len(t)) for t in toks]); df = Counter()
        for i, t in enumerate(toks):
            for w, n_ in Counter(t).items(): j = s.vocab.setdefault(w, len(s.vocab)); r.append(i); c.append(j); v.append(n_); df[j] += 1
        N = len(docs); idf = np.zeros(len(s.vocab))
        for j, d in df.items(): idf[j] = math.log(1 + (N - d + .5) / (d + .5))
        r, c, v = np.array(r), np.array(c), np.array(v, dtype=float)
        s.W = csr_matrix((idf[c] * v * (k1 + 1) / (v + k1 * (1 - b + b * dl[r] / dl.mean())), (r, c)), shape=(N, len(s.vocab)))
    def score(s, queries):
        r, c = [], []
        for i, q in enumerate(queries):
            for w in set(bm_tokens(q)):
                if w in s.vocab: r.append(i); c.append(s.vocab[w])
        Q = csr_matrix((np.ones(len(r)), (r, c)), shape=(len(queries), len(s.vocab))); return (Q @ s.W.T).toarray()

ap = argparse.ArgumentParser(); ap.add_argument("--artifacts", default="artifacts"); ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this computer only; 0.0.0.0 = whole network")
ap.add_argument("--no-browser", action="store_true", help="do not auto-open a browser tab; open the printed URL yourself")
args = ap.parse_args(); A = args.artifacts; HERE = os.path.dirname(os.path.abspath(__file__))
dev = "cuda" if torch.cuda.is_available() else "cpu"
log(f"device {dev}; torch threads {torch.get_num_threads()}; artifacts folder: {os.path.abspath(A)}")
for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy"):
    if os.environ.get(k): log(f"note: environment variable {k}={os.environ[k]}")

NEED = ["tokenizer.json", "encoder.pt", "corpus.json", "sent_emb.npy", "tsne_idx.npy", "tsne_xy.npy", "metrics.json", "curves.json", "vocab_freq.json"]
SIZES = {f: (os.path.getsize(f"{A}/{f}") if os.path.exists(f"{A}/{f}") else None) for f in NEED}
for f, n in SIZES.items(): log(f"  artifact {f:<16} " + ("MISSING" if n is None else f"{n / 1e6:8.2f} MB"))
if any(n is None for n in SIZES.values()):
    raise SystemExit("Some artifact files are missing (incomplete unzip / upload?). Copy ALL files from one notebook run into the artifacts folder.")

def timed(name, fn):
    t = time.time(); r = fn(); log(f"loaded {name} ({time.time() - t:.1f}s)"); return r
tok = timed("tokenizer", lambda: BPETokenizer.load(f"{A}/tokenizer.json"))
ck = timed("encoder.pt", lambda: torch.load(f"{A}/encoder.pt", map_location=dev)); c = ck["cfg"]
enc = Encoder(c["vocab"], c["d"], c["L"], c["h"], c["dff"], c["max_len"]).to(dev); enc.load_state_dict(ck["state"]); enc.eval()
C = timed("corpus.json", lambda: json.load(open(f"{A}/corpus.json")))
S = timed("sent_emb.npy", lambda: np.load(f"{A}/sent_emb.npy").astype(np.float32))
T_IDX, T_XY = np.load(f"{A}/tsne_idx.npy"), np.load(f"{A}/tsne_xy.npy")
METRICS, CURVES = json.load(open(f"{A}/metrics.json")), json.load(open(f"{A}/curves.json"))
FREQ = np.array(json.load(open(f"{A}/vocab_freq.json"))["vocab_freq"])
if S.shape[0] != len(C["sentences"]): log(f"WARNING: sent_emb has {S.shape[0]} rows but corpus has {len(C['sentences'])} sentences -> artifacts come from DIFFERENT runs")
if S.shape[1] != c["d"]: log(f"WARNING: sent_emb dimension {S.shape[1]} != model dimension {c['d']} -> artifacts come from DIFFERENT runs")
title_of = lambda i: C["para_title"][C["sent_para"][i]]
titled = [f"{title_of(i).replace('_', ' ')}. {s}" for i, s in enumerate(C["sentences"])]     # same text the notebook indexed
bm25 = timed("BM25 index", lambda: BM25(titled))
BM25_W = METRICS.get("bm25_weight", 0.5)                                                      # weight chosen by the notebook's sweep
z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
top_titles = [t for t, _ in Counter(title_of(i) for i in T_IDX).most_common(10)]
row_of = {int(i): r for r, i in enumerate(T_IDX)}
log(f"ready: {len(C['sentences']):,} sentences, vocab {len(tok.vocab):,}, device {dev}, bm25 weight {BM25_W}")

@torch.no_grad()
def embed(texts):
    ids = pad_batch([tok.encode(t, c["max_len"]) for t in texts]).to(dev); h, _ = enc(ids)
    return F.normalize(mean_pool(h, ids != PAD), dim=-1).cpu().numpy()

def info():
    f = np.sort(FREQ[N_SPECIAL:])[::-1]; f = f[f > 0]; step = max(1, len(f) // 400)
    xs = T_XY - T_XY.min(0); xs = xs / xs.max(0)
    cat = [top_titles.index(title_of(i)) if title_of(i) in top_titles else -1 for i in T_IDX]
    return dict(metrics=METRICS, curves=CURVES, zipf=[[int(i + 1), int(f[i])] for i in range(0, len(f), step)], layers=c["L"], heads=c["h"],
                map=dict(xy=np.round(xs, 4).tolist(), cat=cat, titles=top_titles))

def search(q, k, mode):
    e = embed([q])[0]; dense = S @ e; bm = bm25.score([q])[0]
    sc = {"dense": dense, "bm25": bm, "hybrid": z(dense) + BM25_W * z(bm)}[mode]              # z-score fusion, as in the notebook
    top = np.argsort(-sc)[:k]; sd = S[T_IDX] @ e; nb = np.argsort(-sd)[:5]; w = np.exp((sd[nb] - sd[nb].max()) * 20)
    pos = (T_XY[nb] * w[:, None]).sum(0) / w.sum(); lo, hi = T_XY.min(0), (T_XY - T_XY.min(0)).max(0)
    return dict(results=[dict(sentence=C["sentences"][i], title=title_of(i), score=float(sc[i]), dense=float(dense[i]), bm25=float(bm[i]),
                              paragraph=C["paragraphs"][C["sent_para"][i]], row=row_of.get(int(i), -1)) for i in top],
                query_xy=((pos - lo) / hi).round(4).tolist(), near_rows=nb.tolist())

@torch.no_grad()
def attention(text, layer):
    ids = torch.tensor([tok.encode(text, 32)], device=dev); _, att = enc(ids, return_attn=True)
    return dict(tokens=[tok.vocab[i] for i in ids[0].tolist()], attn=np.round(att[layer][0].cpu().numpy(), 3).tolist())

def neighbors(word):
    w = word.strip().lower()
    if w not in tok.tok2id: return dict(error=f"'{w}' is not a single token in our vocabulary; it splits into {tok.tokenize(w)}. Try a more common word.")
    E = F.normalize(enc.tok_emb.weight.detach().float(), dim=-1); s = E @ E[tok.tok2id[w]]; s[:N_SPECIAL] = -1
    top = s.topk(11); return dict(neighbors=[[tok.vocab[j], round(v, 3)] for v, j in zip(top.values.tolist(), top.indices.tolist()) if tok.vocab[j] != w][:10])

# warm-up: run every heavy step once now, so a hang shows up in the start-up log instead of as a page that "loads forever"
timed("warm-up embedding", lambda: embed(["warm up"]))
timed("warm-up search", lambda: search("which team won?", 1, "hybrid"))
INFO = timed("page data (/api/info)", info)

ROUTES = {
 "/api/tokenize": lambda d: dict(pieces=tok.tokenize(d["text"]), ids=tok.encode(d["text"], add_special=False), words=len(pretokenize(d["text"]))),
 "/api/similarity": lambda d: (lambda v: dict(cosine=float(v[0] @ v[1]), a=np.round(v[0][:64], 3).tolist(), b=np.round(v[1][:64], 3).tolist()))(embed([d["a"], d["b"]])),
 "/api/search": lambda d: search(d["query"], int(d.get("k", 5)), d.get("mode", "dense")),
 "/api/attention": lambda d: attention(d["text"], int(d.get("layer", -1))),
 "/api/neighbors": lambda d: neighbors(d["word"]),
}

TEST_HTML = """<!doctype html><meta charset="utf-8"><title>diagnostic</title><body style="font:16px sans-serif;padding:20px">
<h2>Server diagnostic page</h2><p>If you can read this, the browser reaches the server. Results of each call:</p><pre id="o">running...</pre><script>
const o=document.getElementById('o');o.textContent='';const out=s=>o.textContent+=s+'\\n';
async function t(name,fn){const t0=performance.now();try{const r=await fn();out('OK   '+name+'   '+(performance.now()-t0).toFixed(0)+' ms   '+r)}catch(e){out('FAIL '+name+'   '+e)}}
(async()=>{out('page loaded from '+location.href);
await t('GET /ping',async()=>(await fetch('/ping')).text());
await t('GET /debug',async()=>{const d=await (await fetch('/debug')).json();return 'device '+d.device+', uptime '+d.uptime_s+' s'});
await t('GET /api/info',async()=>{const d=await (await fetch('/api/info')).json();return Object.keys(d).join(',')});
await t('POST /api/search',async()=>{const d=await (await fetch('/api/search',{method:'POST',body:JSON.stringify({query:'Which team won Super Bowl 50?',k:1,mode:'hybrid'})})).json();return d.error||d.results[0].sentence.slice(0,70)});
await t('GET / (main page)',async()=>{const r=await fetch('/');return r.status+', '+(await r.text()).length+' bytes'});
out('done')})();</script>"""

INFLIGHT, LOCK, NREQ = {}, threading.Lock(), [0]

def tracked(fn):                                                  # log + time every request, and remember which are still running
    def w(self):
        t = time.time(); key = object(); NREQ[0] += 1
        with LOCK: INFLIGHT[key] = (t, f"{self.command} {self.path}")
        log(f"-> {self.command} {self.path}   (from {self.client_address[0]}:{self.client_address[1]})")
        try: return fn(self)
        finally:
            with LOCK: INFLIGHT.pop(key, None)
            log(f"<- {self.command} {self.path}   done in {time.time() - t:.2f}s")
    return w

def debug_info():
    with LOCK: running = [(round(time.time() - t, 1), d) for t, d in INFLIGHT.values()]
    return dict(python=sys.version.split()[0], torch=torch.__version__, device=dev, torch_threads=torch.get_num_threads(), uptime_s=round(time.time() - T0, 1),
                requests_served=NREQ[0], in_flight=running, artifact_sizes_mb={k: round(v / 1e6, 2) for k, v in SIZES.items()},
                sentences=len(C["sentences"]), index_html_exists=os.path.exists(f"{HERE}/index.html"), recent_log=LOG[-40:])

class H(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json", code=200):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    @tracked
    def do_GET(self):
        try:
            p = self.path.split("?")[0]
            if p == "/ping": return self._send(b"ok", "text/plain")
            if p == "/debug": return self._send(debug_info())
            if p == "/test": return self._send(TEST_HTML.encode(), "text/html; charset=utf-8")
            if p == "/slow": time.sleep(float(self.path.split("s=")[-1]) if "s=" in self.path else 12); return self._send(b"slow done", "text/plain")
            if p == "/api/info": return self._send(INFO)
            if p in ("/", "/index.html"): return self._send(open(f"{HERE}/index.html", "rb").read(), "text/html; charset=utf-8")
            self._send({"error": "not found"}, code=404)
        except Exception as e:
            log(f"ERROR on {self.path}: {e!r}"); self._send({"error": str(e)}, code=500)
    @tracked
    def do_POST(self):
        try:
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"); self._send(ROUTES[self.path](d))
        except Exception as e:
            log(f"ERROR on {self.path}: {e!r}"); self._send({"error": str(e)}, code=400)
    def log_message(self, *a): pass

class Server(ThreadingHTTPServer):
    allow_reuse_address = os.name != "nt"          # Windows: SO_REUSEADDR would let a 2nd copy share the port with an old stuck copy
    request_queue_size = 128
    def get_request(self):
        r = super().get_request(); log(f"connection accepted from {r[1][0]}:{r[1][1]}"); return r   # browsers also open idle "pre-connect" sockets: normal
    def handle_error(self, request, client_address): log(f"connection error from {client_address}: {sys.exc_info()[1]!r}")

def watchdog():
    last = time.time()
    while True:
        time.sleep(5)
        with LOCK: stuck = [(round(time.time() - t, 1), d) for t, d in INFLIGHT.values() if time.time() - t > 8]
        if stuck: log(f"WARNING: request(s) running for more than 8 s: {stuck}  -> stack of every thread follows"); faulthandler.dump_traceback(all_threads=True)
        if time.time() - last > 60: last = time.time(); log(f"alive; {NREQ[0]} requests served so far")

def self_test(port):                                              # can this computer talk to its own server? (bypasses any proxy settings)
    time.sleep(0.5); op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for host in ("127.0.0.1", "localhost"):
        t = time.time()
        try: r = op.open(f"http://{host}:{port}/ping", timeout=8).read(); log(f"SELF-TEST http://{host}:{port}/ping -> {r!r} in {time.time() - t:.2f}s")
        except Exception as e: log(f"SELF-TEST http://{host}:{port}/ping FAILED after {time.time() - t:.1f}s: {e!r}")

if __name__ == "__main__":
    try: srv = Server((args.host, args.port), H)
    except OSError as e: raise SystemExit(f"Cannot use port {args.port} ({e}). Another program (maybe an older app.py) is using it. Try: python app.py --artifacts {A} --port {args.port + 1}")
    faulthandler.cancel_dump_traceback_later()
    url = f"http://127.0.0.1:{args.port}"
    log("=" * 64); log(f"SERVER LISTENING on {args.host}:{args.port}.  Open:  {url}/test   (diagnostic page)   then   {url}/   (main page)"); log("=" * 64)
    threading.Thread(target=watchdog, daemon=True).start(); threading.Thread(target=self_test, args=(args.port,), daemon=True).start()
    if not os.path.exists(f"{HERE}/index.html"): log(f"WARNING: {HERE}/index.html not found; the main page cannot load")
    if not args.no_browser: threading.Timer(1.5, lambda: (log("opening browser tab ..."), webbrowser.open(url + "/test"))).start()
    srv.serve_forever()
