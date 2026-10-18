# Auto-extracted from the notebook so the demo uses exactly the same code.
import re, json, math, heapq, unicodedata
from collections import Counter, defaultdict
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.sparse import csr_matrix

PRETOK = re.compile(r"\w+|[^\w\s]", re.UNICODE)
def normalize(t): return unicodedata.normalize("NFKC", t).lower()
def pretokenize(t): return PRETOK.findall(normalize(t))
SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]; PAD, UNK, CLS, SEP, MASK = range(5); N_SPECIAL = 5

class BPETokenizer:
    def __init__(self, vocab, merges):
        self.vocab, self.merges = vocab, merges
        self.tok2id = {t: i for i, t in enumerate(vocab)}
        self.ranks = {tuple(m): i for i, m in enumerate(merges)}
        self.cache = {}

    @classmethod
    def train(cls, word_counts, vocab_size, min_char=50, min_word=3, verbose=True):
        char_cnt = Counter()
        for w, c in word_counts.items():
            for ch in w: char_cnt[ch] += c
        alphabet = sorted(ch for ch, c in char_cnt.items() if c >= min_char)
        aset = set(alphabet)
        vocab = list(SPECIALS) + alphabet + ["##" + c for c in alphabet]; seen = set(vocab)
        words = [(w, c) for w, c in word_counts.items() if c >= min_word and all(ch in aset for ch in w)]
        W = [[w[0]] + ["##" + ch for ch in w[1:]] for w, _ in words]; Fq = [c for _, c in words]
        pc, where = defaultdict(int), defaultdict(set)
        for i, syms in enumerate(W):
            for p in zip(syms, syms[1:]): pc[p] += Fq[i]; where[p].add(i)
        heap = [(-c, p) for p, c in pc.items()]; heapq.heapify(heap); merges = []
        while len(vocab) < vocab_size and heap:
            negc, pair = heapq.heappop(heap); cur = pc.get(pair, 0)
            if cur != -negc:                                   # stale heap entry -> refresh
                if cur > 0: heapq.heappush(heap, (-cur, pair))
                continue
            if cur < 2: break
            a, b = pair; merged = a + b[2:]; changed = set()
            for i in list(where[pair]):
                syms = W[i]; new = []; j = 0
                while j < len(syms):
                    if j < len(syms) - 1 and syms[j] == a and syms[j + 1] == b: new.append(merged); j += 2
                    else: new.append(syms[j]); j += 1
                if len(new) == len(syms): continue
                for p in zip(syms, syms[1:]): pc[p] -= Fq[i]
                for p in zip(new, new[1:]): pc[p] += Fq[i]; where[p].add(i); changed.add(p)
                W[i] = new
            pc.pop(pair, None); where.pop(pair, None)
            for p in changed:
                if pc[p] > 0: heapq.heappush(heap, (-pc[p], p))
            merges.append(pair)
            if merged not in seen: seen.add(merged); vocab.append(merged)
            if verbose and len(merges) % 2000 == 0: print(f"  {len(merges):>6} merges, vocab {len(vocab)}, last: {a} + {b} -> {merged}")
        return cls(vocab, merges)

    def _word(self, w):
        syms = [w[0]] + ["##" + c for c in w[1:]]
        while len(syms) > 1:
            best, br = None, 1e18
            for i in range(len(syms) - 1):
                r = self.ranks.get((syms[i], syms[i + 1]))
                if r is not None and r < br: br, best = r, i
            if best is None: break
            syms[best:best + 2] = [syms[best] + syms[best + 1][2:]]
        ids = [self.tok2id.get(s, UNK) for s in syms]
        ids = [UNK] if UNK in ids else ids                      # any unknown char -> whole word [UNK] (like BERT)
        self.cache[w] = ids; return ids

    def encode_words(self, text):
        out = []; cache = self.cache
        for w in pretokenize(text): out.extend(cache.get(w) or self._word(w))
        return out

    def encode(self, text, max_len=None, add_special=True):
        ids = []
        for piece in re.split(r"(\[MASK\])", text):             # allow a literal [MASK] in the text
            ids.extend([MASK] if piece == "[MASK]" else self.encode_words(piece))
        if add_special:
            ids = ids[: (max_len - 2) if max_len else None]; ids = [CLS] + ids + [SEP]
        return ids

    def tokenize(self, text): return [self.vocab[i] for i in self.encode(text, add_special=False)]
    def decode(self, ids): return " ".join(self.vocab[i] for i in ids)
    def save(self, path): json.dump({"vocab": self.vocab, "merges": self.merges}, open(path, "w"))
    @classmethod
    def load(cls, path): d = json.load(open(path)); return cls(d["vocab"], [tuple(m) for m in d["merges"]])


class Attention(nn.Module):
    def __init__(s, d, h, p):
        super().__init__(); s.h, s.dh, s.p = h, d // h, p
        s.qkv, s.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
    def forward(s, x, keep, return_attn=False):                 # keep: (B,T) bool, True = real token
        B, T, _ = x.shape
        q, k, v = s.qkv(x).view(B, T, 3, s.h, s.dh).permute(2, 0, 3, 1, 4)
        if return_attn:                                         # explicit formula
            sc = (q @ k.transpose(-2, -1)) / math.sqrt(s.dh)
            sc = sc.masked_fill(~keep[:, None, None, :], float("-inf"))
            a = sc.softmax(-1); o = F.dropout(a, s.p, s.training) @ v
        else:                                                   # same maths, fused kernel
            a = None; o = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :], dropout_p=s.p if s.training else 0.0)
        return s.out(o.transpose(1, 2).reshape(B, T, -1)), a

class Block(nn.Module):
    def __init__(s, d, h, dff, p):
        super().__init__(); s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.attn = Attention(d, h, p); s.ff = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Linear(dff, d)); s.drop = nn.Dropout(p)
    def forward(s, x, keep, return_attn=False):
        a_out, a = s.attn(s.ln1(x), keep, return_attn); x = x + s.drop(a_out)
        return x + s.drop(s.ff(s.ln2(x))), a

class Encoder(nn.Module):
    def __init__(s, vocab, d, L, h, dff, max_len, p=0.1):
        super().__init__(); s.cfg = dict(vocab=vocab, d=d, L=L, h=h, dff=dff, max_len=max_len, p=p)
        s.tok_emb, s.pos_emb = nn.Embedding(vocab, d, padding_idx=PAD), nn.Embedding(max_len, d)
        s.ln_emb, s.drop = nn.LayerNorm(d), nn.Dropout(p)
        s.blocks = nn.ModuleList(Block(d, h, dff, p) for _ in range(L)); s.ln_f = nn.LayerNorm(d)
        s.apply(s._init)
        for b in s.blocks:                                      # GPT-2 style: shrink residual-branch outputs
            for lin in (b.attn.out, b.ff[2]): nn.init.normal_(lin.weight, std=0.02 / math.sqrt(2 * L))
    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
    def forward(s, ids, return_attn=False):
        keep = ids != PAD; x = s.tok_emb(ids) + s.pos_emb(torch.arange(ids.shape[1], device=ids.device))
        x = s.drop(s.ln_emb(x)); attns = []
        for b in s.blocks:
            x, a = b(x, keep, return_attn); attns.append(a)
        return s.ln_f(x), attns


def mean_pool(h, keep): k = keep.unsqueeze(-1).to(h.dtype); return (h * k).sum(1) / k.sum(1).clamp(min=1)
def pad_batch(seqs):
    T = max(map(len, seqs)); out = torch.full((len(seqs), T), PAD, dtype=torch.long)
    for i, s in enumerate(seqs): out[i, :len(s)] = torch.tensor(s)
    return out

class BM25:
    def __init__(s, docs, k1=1.5, b=0.75):
        toks = [pretokenize(d) for d in docs]; s.vocab = {}; r, c, v = [], [], []; dl = np.array([len(t) for t in toks]); df = Counter()
        for i, t in enumerate(toks):
            for w, n_ in Counter(t).items(): j = s.vocab.setdefault(w, len(s.vocab)); r.append(i); c.append(j); v.append(n_); df[j] += 1
        N = len(docs); idf = np.zeros(len(s.vocab))
        for j, d in df.items(): idf[j] = math.log(1 + (N - d + .5) / (d + .5))
        r, c, v = np.array(r), np.array(c), np.array(v, dtype=float)
        s.W = csr_matrix((idf[c] * v * (k1 + 1) / (v + k1 * (1 - b + b * dl[r] / dl.mean())), (r, c)), shape=(N, len(s.vocab)))
    def score(s, queries):
        r, c = [], []
        for i, q in enumerate(queries):
            for w in set(pretokenize(q)):
                if w in s.vocab: r.append(i); c.append(s.vocab[w])
        Q = csr_matrix((np.ones(len(r)), (r, c)), shape=(len(queries), len(s.vocab))); return (Q @ s.W.T).toarray()

