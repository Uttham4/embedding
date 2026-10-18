"""Seminar demo server.  Usage:  python app.py --artifacts artifacts   ->  http://localhost:8000
Needs only: torch, numpy, scipy (same as the notebook). No web framework."""
import argparse, json, os, threading, webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, torch, torch.nn.functional as F
from OneDrive.Desktop.scratch.demo.lib import BPETokenizer, Encoder, BM25, pretokenize, pad_batch, mean_pool, PAD, N_SPECIAL

ap = argparse.ArgumentParser(); ap.add_argument("--artifacts", default="artifacts"); ap.add_argument("--port", type=int, default=8000)
args = ap.parse_args(); A = args.artifacts; HERE = os.path.dirname(os.path.abspath(__file__))
dev = "cuda" if torch.cuda.is_available() else "cpu"

tok = BPETokenizer.load(f"{A}/tokenizer.json")
ck = torch.load(f"{A}/encoder.pt", map_location=dev); c = ck["cfg"]
enc = Encoder(c["vocab"], c["d"], c["L"], c["h"], c["dff"], c["max_len"]).to(dev); enc.load_state_dict(ck["state"]); enc.eval()
C = json.load(open(f"{A}/corpus.json")); S = np.load(f"{A}/sent_emb.npy").astype(np.float32)
T_IDX, T_XY = np.load(f"{A}/tsne_idx.npy"), np.load(f"{A}/tsne_xy.npy")
METRICS, CURVES = json.load(open(f"{A}/metrics.json")), json.load(open(f"{A}/curves.json"))
FREQ = np.array(json.load(open(f"{A}/vocab_freq.json"))["vocab_freq"])
bm25 = BM25(C["sentences"]); title_of = lambda i: C["para_title"][C["sent_para"][i]]
top_titles = [t for t, _ in Counter(title_of(i) for i in T_IDX).most_common(10)]
row_of = {int(i): r for r, i in enumerate(T_IDX)}
print(f"ready: {len(C['sentences']):,} sentences, vocab {len(tok.vocab):,}, device {dev}")

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
    sc = {"dense": dense, "bm25": bm, "hybrid": dense + 0.5 * bm / (bm.max() + 1e-9)}[mode]
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

ROUTES = {
 "/api/tokenize": lambda d: dict(pieces=tok.tokenize(d["text"]), ids=tok.encode(d["text"], add_special=False), words=len(pretokenize(d["text"]))),
 "/api/similarity": lambda d: (lambda v: dict(cosine=float(v[0] @ v[1]), a=np.round(v[0][:64], 3).tolist(), b=np.round(v[1][:64], 3).tolist()))(embed([d["a"], d["b"]])),
 "/api/search": lambda d: search(d["query"], int(d.get("k", 5)), d.get("mode", "dense")),
 "/api/attention": lambda d: attention(d["text"], int(d.get("layer", -1))),
 "/api/neighbors": lambda d: neighbors(d["word"]),
}
INFO = None

class H(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json", code=200):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        global INFO
        if self.path == "/api/info": INFO = INFO or info(); return self._send(INFO)
        if self.path in ("/", "/index.html"): return self._send(open(f"{HERE}/index.html", "rb").read(), "text/html; charset=utf-8")
        self._send({"error": "not found"}, code=404)
    def do_POST(self):
        try:
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"); self._send(ROUTES[self.path](d))
        except Exception as e: self._send({"error": str(e)}, code=400)
    def log_message(self, *a): pass

if __name__ == "__main__":
    url = f"http://localhost:{args.port}"; print("open", url); threading.Timer(1, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()
