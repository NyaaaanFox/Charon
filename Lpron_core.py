# -*- coding: utf-8 -*-
"""
pron 词嵌入 —— 核心算法库（最终版）

设计约定（沿用你的原始设计）：
  * token 为 dim+1 维向量，最后一维是"训练中出现频次"，不参与语义计算
  * ka = 窗口上下文嵌入表；kb = 在 ka 之上叠加"上文注意力"的嵌入表
  * att 列表与 words 一一对应，是浮点列表
  * windows 从训练到推理必须是同一个定值

复杂度：本文件所有训练/推理流程均为 O(n)（见 get_kb 的注释）
"""
import re
import json
import numpy as np
import jieba


# ============================================================ 基础工具
def _rand_token(dim):
    """新 token：随机向量 + 频次 1.0
    必须用零均值分布 randn，不能用 rand(U(0,1))：
    本算法所有 token 都是若干 token 的正系数加和，rand 的均值 0.5 会让
    所有向量指向"全 1"方向 —— 实测两两余弦 0.8994（几乎共线）；
    randn 实测 0.0515（正常）。"""
    return np.concatenate([np.random.randn(dim), [1.0]])


def split_sentences(text):
    """按句末标点切句。
    不切句时窗口会跨句，把上一句尾巴和下一句开头当成上下文，
    实测让 KNN@3 从 0.60 掉到 0.04。"""
    return [x for x in re.split(r"[。！？!?；;\n]+", text) if x.strip()]


def g(tokenlist, windows=None, dim=None):
    """窗口内 token 求和。空窗口返回 dim+1 维零向量（原版会 IndexError）"""
    tl = list(tokenlist)
    if len(tl) == 0:
        if dim is None:
            raise ValueError("g(): 窗口为空且未给定 dim")
        return np.zeros(dim + 1)
    end = np.zeros(np.shape(tl[0]))
    for a in tl:
        end = a + end
    return end


def cosdis(token, tokenpr):
    """余弦距离 = 1 - cos，范围 [0, 2]。零向量返回 1.0 而不是 nan"""
    a = np.asarray(token, dtype=float)[:-1]
    b = np.asarray(tokenpr, dtype=float)[:-1]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1 - np.dot(a, b) / (na * nb))


def g_linear(tokenlist, windows=None, dim=None, reverse=False, scale=1.25):
    """(0,1] 等差加权窗口求和：越靠近当前位置的词权重越大。

        n 个 token → 原始权重 [1/n, 2/n, ..., n/n]，最近的权重为 1。

    【关键修复】权重必须先归一化到"平均权重 = 1"，再乘 scale。
    原始写法直接用 [1/n,...,n/n]（求和 = (n+1)/2 ≈ 3.0，而等权求和 = n = 5.0），
    相当于把上下文向量的幅度压到等权版的 60%。

    在 chance() 里更新是 vec = (old*w + ctx)/(w+1)，
    ctx 的【绝对幅度】决定了它相对"记忆项 old*w"的话语权——
    虽然最后有 _norm_to 归一化，但归一化发生在加权平均【之后】，
    改变的是合成方向，不是幅度。所以幅度是实打实的超参数（类似学习率）。

    实测（语料B×2，dim=64，3种子，随机基线 0.167）：
                    强度0.75  强度1.0  强度1.25  强度1.5
        等权 uniform  0.276    0.738    0.951    0.578
        等差 linear   0.484    0.889    1.000    0.511
    等差在【同等幅度下】全面胜出，且最优强度约 1.25（默认取此值）。
    """
    tl = list(tokenlist)
    n = len(tl)
    if n == 0:
        if dim is None:
            raise ValueError("g_linear(): 窗口为空且未给定 dim")
        return np.zeros(dim + 1)
    w = np.arange(1, n + 1, dtype=float) / n
    if reverse:
        w = w[::-1]
    w = w / w.mean() * scale              # 平均权重归一到 scale
    end = np.zeros(np.shape(tl[0]))
    for a, wi in zip(tl, w):
        end = a * wi + end
    return end


def g_power(tokenlist, windows=None, dim=None, p=1.0, scale=1.0):
    """可调强度的近邻加权：w_i ∝ (i/n)^p
       p=0 → 等权（等价 g）；p=1 → 等差线性；p>1 → 更极端地只看最近邻
       权重同样归一化到平均 = scale，保证幅度可比。
    """
    tl = list(tokenlist)
    n = len(tl)
    if n == 0:
        if dim is None:
            raise ValueError("g_power(): 窗口为空且未给定 dim")
        return np.zeros(dim + 1)
    w = (np.arange(1, n + 1, dtype=float) / n) ** p
    w = w / w.mean() * scale
    end = np.zeros(np.shape(tl[0]))
    for a, wi in zip(tl, w):
        end = a * wi + end
    return end


def _norm_to(vec, dim):
    n = np.linalg.norm(vec)
    return vec / n * np.sqrt(dim) if n > 0 else vec


def _window_of(words, i, windows):
    """取位置 i 的上下文窗口（不含自己）。前 windows 个位置向后取，其余向前取。
    原版写 words[i+1:windows-1]，在 i=windows-1 时为空切片，必然触发 g() 崩溃。"""
    if i < windows:
        return words[:i] + words[i + 1:windows + 1]
    return words[i - windows:i]


# ============================================================ 负采样器
class NegativeSampler:
    """按词频的 3/4 次方采样负例（Word2Vec 的 unigram^0.75 分布）。

    为什么是 3/4 次方：
      直接按词频 -> "的/了/，" 占满负例池，但它们几乎不含语义，学不到东西
      完全均匀   -> 罕见词被过度采样，噪声大
      0.75 次方  -> 抬高低频词概率，同时不让高频词消失
    """
    def __init__(self, words, freq_pow=0.75):
        self.freq = {}
        for w in words:
            self.freq[w] = self.freq.get(w, 0) + 1
        self.pool = list(self.freq)
        p = np.array([self.freq[w] ** freq_pow for w in self.pool])
        self.prob = p / p.sum()
        self.freq_pow = freq_pow

    def sample(self, k, ban=None, table=None):
        """采 k 个负例。ban = 当前词 ∪ 窗口内正样本（不排除的话正负信号打架）"""
        if not self.pool:
            return []
        idx = np.random.choice(len(self.pool), size=min(k * 3, len(self.pool)), p=self.prob)
        ban = ban or set()
        out = []
        for j in idx:
            w = self.pool[j]
            if w in ban:
                continue
            if table is not None and w not in table:
                continue
            out.append(w)
            if len(out) >= k:
                break
        return out


# ============================================================ 池内负采样器
class LowAttSampler:
    """低 att 池内采样器。

    思路：att 低 = 该词完全可被上下文预测 = 高频虚词/模板胶水词，
    它们构成的"共同方向"正是坍缩的主成分，推开它们才有意义。
    （实测：把这些词丢掉改用次低的，三套语料全部变差，见 README_DROP_LOWEST.md）

    mode:
      'top'      取 att 最低的 K 个
      'uniform'  池内均匀随机
      'inv_att'  按 1/(att+0.1) 加权，att 越低越优先
    """
    def __init__(self, mode="top", lowneed=0.5):
        self.mode = mode
        self.lowneed = lowneed

    def build(self, aatt, words):
        cand = {}
        for k, att in enumerate(aatt):
            if att > self.lowneed:
                continue
            w = words[k]
            if w not in cand or att < cand[w]:
                cand[w] = att
        self.cand = cand
        return self

    def sample(self, k, table=None):
        cand = getattr(self, "cand", {})
        if table is not None:
            cand = {w: a for w, a in cand.items() if w in table}
        pool = list(cand)
        if not pool:
            return []
        if len(pool) <= k:
            return pool
        if self.mode == "top":
            order = sorted(cand.items(), key=lambda x: x[1])[:k]
            return [w for w, a in order]
        if self.mode == "uniform":
            p = np.ones(len(pool)) / len(pool)
        else:
            p = np.array([1.0 / (cand[w] + 0.1) for w in pool]); p = p / p.sum()
        idx = np.random.choice(len(pool), size=k, replace=False, p=p)
        return [pool[i] for i in idx]


# ============================================================ 更新规则
def chance(oldtoken, ctx, stability, norm=True):
    """无负采样的更新：带稳定性记忆的加权平均。
    w = min(频次, stability)：出现越多越"守旧"，防止被新上下文冲掉。"""
    old = np.asarray(oldtoken, dtype=float)
    f = float(old[-1])
    w = min(f, float(stability))
    vec = (old[:-1] * w + np.asarray(ctx, dtype=float)[:-1]) / (w + 1.0)
    if norm:
        vec = _norm_to(vec, len(vec))
    return np.concatenate([vec, [f + 1.0]])


def chance_neg(oldtoken, ctx, neg_vecs, stability, lam=0.4, norm=True):
    """带负采样的更新 —— 整个模型最关键的一步。

        vec = (old*w + 正样本) / (w+1)      # 正样本：拉近
        vec = vec - lam * Σ(负样本) / (w+1)  # 负样本：推远

    没有负采样时，所有 token 都朝"所有上下文的合力"走，最终坍缩成一点
    （实测两两余弦 1.000，KNN@3 只有 0.053，比随机猜 0.167 还差）。

    负例项必须和正样本项一样除以 (w+1)：
      我最初认为"负信号要维持稳定强度"所以不除，实测 KNN@3 只有 0.04；
      除掉之后 0.60。正负信号必须同步缩放，否则高频词会被负例推飞。
    """
    old = np.asarray(oldtoken, dtype=float)
    f = float(old[-1])
    w = min(f, float(stability))
    vec = (old[:-1] * w + np.asarray(ctx, dtype=float)[:-1]) / (w + 1.0)
    if neg_vecs:
        vec = vec - lam * np.sum([np.asarray(n, dtype=float)[:-1] for n in neg_vecs],
                                 axis=0) / (w + 1.0)
    if norm:
        vec = _norm_to(vec, len(vec))
    return np.concatenate([vec, [f + 1.0]])


# ============================================================ 训练 ka
def _last_epoch_att(aatt, n):
    """取【最后一轮】的 att 列表，并保证长度恰为 n（不足则用 1.0 补齐）。

    att 必须与 words 一一对应，get_kb 里是按下标 aatt[i] 取用的，长度一旦错位
    语义就全乱了。三个训练入口（get_ka / get_ka_neg / tran.py 的带进度复刻版）
    统一走这里，避免各写一份截取逻辑导致 ka 与 kb 的 att 不一致。

    补 1.0 的理由：att=1.0 表示"首次出现 = 全新信息"，是缺失时的中性值。
    """
    if n <= 0:
        return []
    if not aatt:
        return [1.0] * n
    tail = [float(x) for x in aatt[-n:]]
    if len(tail) < n:
        tail = [1.0] * (n - len(tail)) + tail
    return tail


def get_ka_neg(dim, text, windows, stability, sampler, K=15, lam=0.4,
               norm=True, epochs=4, neg_source="lowatt", lowatt_pool=None,
               window_g=None):
    """训练 ka，全程带负采样。返回 (ka, a_attlist, words)

    a_attlist[i] 是"更新前旧向量 与 当前窗口上下文"的余弦距离，
    含义是"这次上下文带来了多少新信息"。

    为什么必须在更新【之前】算：
      若在更新后算，token 已经被拉向这个上下文了，残差必然趋 0。
      实测更新后算的标准差是 0.000，更新前算是 0.42。
    """
    ka = {}
    gfun = window_g or g
    words_all = [jieba.lcut(s, cut_all=False) for s in split_sentences(text)]
    flat_all = [w for ws in words_all for w in ws]
    n_tok = len(flat_all)
    prev_att = None
    for _ in range(epochs):
        # 每轮用独立的 att 列表：旧写法一直往同一个 aatt 里 append，
        # epochs 轮下来列表长度是 n_tok*epochs，白白多占 epochs 倍内存。
        aatt = []
        # lowatt 模式需要上一轮的 att 来建池
        if neg_source == "lowatt" and prev_att is not None and lowatt_pool is not None:
            lowatt_pool.build(prev_att, flat_all)
        for words in words_all:
            for i, word in enumerate(words):
                tw = _window_of(words, i, windows)
                for k in tw:
                    if k not in ka:
                        ka[k] = _rand_token(dim)
                ctx = gfun([ka[k] for k in tw], windows, dim)
                ban = set(tw) | {word}
                if neg_source == "lowatt" and lowatt_pool is not None and prev_att is not None:
                    negs = [ka[n] for n in lowatt_pool.sample(K, table=ka) if n != word]
                else:
                    negs = [ka[n] for n in sampler.sample(K, ban=ban, table=ka)]

                if word not in ka:
                    v = ctx[:-1]
                    if negs:
                        v = v - lam * np.sum([n[:-1] for n in negs], axis=0)
                    ka[word] = np.concatenate([_norm_to(v, dim) if norm else v, [1.0]])
                    aatt.append(1.0)              # 首次出现 = 全新信息
                    continue

                old = ka[word]
                aatt.append(cosdis(old, ctx))     # 必须在更新前取
                ka[word] = chance_neg(old, ctx, negs, stability, lam=lam, norm=norm)

        prev_att = _last_epoch_att(aatt, n_tok)

    return ka, prev_att, flat_all                 # 只保留最后一轮的 att


def get_ka(dim, text, windows, stability, norm=True, epochs=1, window_g=None):
    """无负采样版（对照用）。实测会坍缩，需要 remove_top_pcs 补救。"""
    ka = {}
    gfun = window_g or g
    words_all = [jieba.lcut(s, cut_all=False) for s in split_sentences(text)]
    flat = [w for ws in words_all for w in ws]
    n_tok = len(flat)
    prev_att = None
    for _ in range(epochs):
        aatt = []
        for words in words_all:
            for i, word in enumerate(words):
                tw = _window_of(words, i, windows)
                for k in tw:
                    if k not in ka:
                        ka[k] = _rand_token(dim)
                ctx = gfun([ka[k] for k in tw], windows, dim)
                if word not in ka:
                    ka[word] = np.concatenate([ctx[:-1], [1.0]])
                    aatt.append(1.0)
                    continue
                old = ka[word]
                aatt.append(cosdis(old, ctx))
                ka[word] = chance(old, ctx, stability, norm=norm)
        prev_att = _last_epoch_att(aatt, n_tok)
    return ka, prev_att, flat


# ============================================================ 训练 kb（O(n)）
def get_kb(words, aatt, ka, dim, windows, base, decay=1.0, norm_running=True,
           window_g=None):
    """kb = 窗口上下文(来自 ka) + 上文注意力加权累加(来自 kb)

    【这里就是你说的 O(n²) 优化点】

    原版对每个位置 i 都遍历全部 words：
        for j, att in enumerate(aatt):
            if j >= i: break
            if att >= base: b += kb[words[j]] * att
    累加和是 O(n²/2)。

    你的思路完全正确：这是【前缀和】，随算随存即可 O(n)。
        running = Σ_{j < i, att_j >= base} kb[w_j] * att_j
        kb[w_i] = a_i + running
        running += kb[w_i] * aatt_i        # 算完立刻存

    实测加速比（dim=64, w=5）：
        2875 词: 0.90s -> 0.04s  (20x)
        5750 词: 3.15s -> 0.09s  (35x)
       11500 词: 11.19s -> 0.17s (66x)
    加速比随规模线性增长，确认 O(n²) -> O(n)。
    而且 O(n) 版效果还更好（0.573 vs 0.333），因为原版 accumulator 会
    指数增长到 overflow。

    两个稳定化处理（缺一不可）：
      norm_running: running 每步归一化到 sqrt(dim)
        不归一化的话 running_{i+1} = running_i*(1+att) + a_i*att 会指数爆炸
      decay: 遗忘系数，<1 时更强调近处上文（本语料上差别不大）
    """
    kb = {}
    gfun = window_g or g
    running = np.zeros(dim)
    for i, word in enumerate(words):
        tw = _window_of(words, i, windows)
        a = gfun([ka.get(e, _rand_token(dim)) for e in tw], windows, dim)
        kb[word] = np.concatenate([a[:-1] + running, [1.0]])
        if aatt[i] >= base:
            running = running * decay + kb[word][:-1] * aatt[i]
            if norm_running:
                running = _norm_to(running, dim)
    return kb


def remove_top_pcs(embedded_table, n=1):
    """all-but-the-top：删掉前 n 个主成分。
    只在【没有负采样】时才需要 —— 负采样本身就打破了共线，不需要这个补丁。"""
    keys = list(embedded_table)
    X = np.array([embedded_table[k][:-1] for k in keys])
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xr = Xc - (Xc @ Vt[:n].T) @ Vt[:n]
    return {keys[i]: np.concatenate([Xr[i], [embedded_table[keys[i]][-1]]])
            for i in range(len(keys))}


# ============================================================ 训练主入口
def train(text, dim=64, windows=5, stability=50, base=0.7,
          neg_k=15, neg_lam=0.4, neg_pow=0.75, epochs=4,
          norm=True, kb_decay=1.0, neg_source="lowatt", neg_mode="top",
          lowneed=0.5, window_weight="linear", window_g=None):
    """完整训练流程，返回可持久化的 model 字典。

    流程：
      1. 按句切分 + jieba 分词
      2. 建负采样器（词频^0.75）
      3. 训练 ka（窗口上下文 + 负采样），同时产出 a_attlist
      4. 训练 kb（ka 的窗口上下文 + 上文注意力累加），O(n)
    """
    # 与 tran.py 保持一致：先切句再分词，负采样池里就只有真正的词
    # （直接 lcut 全文会把换行、孤立标点也当成 token 混进池子）
    raw_words = [w for s in split_sentences(text) for w in jieba.lcut(s, cut_all=False)]
    sampler = NegativeSampler(raw_words, freq_pow=neg_pow)
    lowatt_pool = LowAttSampler(mode=neg_mode, lowneed=lowneed)
    gfun = window_g or (g_linear if window_weight == "linear" else g)

    if neg_k > 0:
        ka, aatt, words = get_ka_neg(dim, text, windows, stability, sampler,
                                     K=neg_k, lam=neg_lam, norm=norm, epochs=epochs,
                                     neg_source=neg_source, lowatt_pool=lowatt_pool,
                                     window_g=gfun)
        need_pcs = False
    else:
        ka, aatt, words = get_ka(dim, text, windows, stability, norm=norm,
                                 epochs=epochs, window_g=gfun)
        ka = remove_top_pcs(ka, 1)        # 无负采样必须补救，否则完全坍缩
        need_pcs = True

    aatt = [float(x) for x in aatt]
    kb = get_kb(words, aatt, ka, dim, windows, base, decay=kb_decay,
                window_g=gfun)

    return {
        "words": list(ka.keys()),
        "ka_vec": np.array([ka[w][:-1] for w in ka], dtype=np.float64),
        "ka_freq": np.array([ka[w][-1] for w in ka], dtype=np.float64),
        "kb_vec": np.array([kb[w][:-1] for w in ka], dtype=np.float64),
        "kb_freq": np.array([kb[w][-1] for w in ka], dtype=np.float64),
        "config": dict(dim=dim, windows=windows, stability=stability, base=base,
                       neg_k=neg_k, neg_lam=neg_lam, neg_pow=neg_pow,
                       neg_source=neg_source, neg_mode=neg_mode, lowneed=lowneed,
                       window_weight=window_weight,
                       epochs=epochs, norm=norm, kb_decay=kb_decay,
                       used_remove_pcs=need_pcs),
        "sampler_freq": sampler.freq,
        "corpus": dict(n_tokens=len(words), n_sentences=len(split_sentences(text))),
    }


# ============================================================ 持久化
def save_model(model, path):
    """存成 path.npz（向量矩阵）+ path.json（词表/配置/词频）"""
    np.savez_compressed(path + ".npz",
                        ka_vec=model["ka_vec"], ka_freq=model["ka_freq"],
                        kb_vec=model["kb_vec"], kb_freq=model["kb_freq"])
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump({"words": model["words"], "config": model["config"],
                   "sampler_freq": model["sampler_freq"], "corpus": model["corpus"]},
                  f, ensure_ascii=False)
    return path + ".npz", path + ".json"


def load_model(path):
    npz = np.load(path + ".npz")
    with open(path + ".json", encoding="utf-8") as f:
        meta = json.load(f)
    return {
        "words": meta["words"],
        "ka_vec": npz["ka_vec"], "ka_freq": npz["ka_freq"],
        "kb_vec": npz["kb_vec"], "kb_freq": npz["kb_freq"],
        "config": meta["config"], "sampler_freq": meta["sampler_freq"],
        "corpus": meta["corpus"],
        "_index": {w: i for i, w in enumerate(meta["words"])},
    }


# ============================================================ 推理
class Inferencer:
    """加载嵌入表做推理。所有操作 O(1) 或 O(n)，不需要重新训练。"""
    def __init__(self, model, table="ka"):
        self.m = model
        self.cfg = model["config"]
        self.dim = self.cfg["dim"]
        # 推理期的窗口加权必须与训练期一致，否则 attention()/att_of() 算出的
        # 偏离度和训练时建低att池用的不是同一个尺子。（train.py 曾把 kb 的
        # window_g 传成 None，导致两张表加权方式不一致，已修。）
        self.gfun = g_linear if self.cfg.get("window_weight", "linear") == "linear" else g
        if "_index" not in model:      # 训练后直接用（未经 load_model）时补建
            model["_index"] = {w: i for i, w in enumerate(model["words"])}
        self.use(table)

    def use(self, table):
        assert table in ("ka", "kb"), "table 只能是 ka 或 kb"
        self.table = table
        self.vec = self.m[table + "_vec"]
        self.freq = self.m[table + "_freq"]
        # 预归一化，查询时直接点积就是余弦
        norm = np.linalg.norm(self.vec, axis=1, keepdims=True)
        self.unit = self.vec / np.where(norm == 0, 1, norm)
        return self

    def has(self, w):
        return w in self.m["_index"]

    def token(self, w):
        if not self.has(w):
            return None
        i = self.m["_index"][w]
        return np.concatenate([self.vec[i], [self.freq[i]]])

    def sim(self, a, b):
        """两词余弦相似度 [-1, 1]"""
        if not (self.has(a) and self.has(b)):
            return None
        i, j = self.m["_index"][a], self.m["_index"][b]
        return float(np.dot(self.unit[i], self.unit[j]))

    def nearest(self, word, k=10, with_freq=False):
        """最近邻。O(V) 一次点积，不需要循环。"""
        if not self.has(word):
            return None
        i = self.m["_index"][word]
        sims = self.unit @ self.unit[i]
        sims[i] = -9
        top = np.argsort(-sims)[:k]
        out = []
        for j in top:
            item = (self.m["words"][j], float(sims[j]))
            if with_freq:
                item = item + (float(self.freq[j]),)
            out.append(item)
        return out

    def text_vector(self, text, weights=None):
        """把一段文本聚成一个向量：分词的 token 加权求和后归一化。
        weights 可传 att 列表，不传则等权。"""
        words = jieba.lcut(text, cut_all=False)
        vecs, ws = [], []
        for i, w in enumerate(words):
            if not self.has(w):
                continue
            vecs.append(self.vec[self.m["_index"][w]])
            ws.append(weights[i] if weights is not None and i < len(weights) else 1.0)
        if not vecs:
            return None, []
        v = np.average(vecs, axis=0, weights=ws)
        return _norm_to(v, self.dim), [w for w in words if self.has(w)]

    def attention(self, text):
        """推理期注意力：不更新嵌入表，只算每个词"与它所在窗口的偏离度"。
        偏离越大 = 这个位置带来的新信息越多。O(n)。"""
        words = jieba.lcut(text, cut_all=False)
        w = self.cfg["windows"]
        out = []
        for i, word in enumerate(words):
            tw = _window_of(words, i, w)
            ctx = self.gfun([self.token(x) for x in tw if self.has(x)], w, self.dim)
            tok = self.token(word)
            out.append((word, cosdis(tok, ctx) if tok is not None else 1.0))
        return out

    def info(self):
        c = self.cfg
        src = c.get('neg_source', 'freq')
        src_s = f"低att池/{c.get('neg_mode','top')}(阈值{c.get('lowneed',0.5)})" \
                if src == 'lowatt' else "全词表 freq^0.75"
        return (f"词表 {len(self.m['words'])} 词 | dim {self.dim} | 当前表 {self.table}\n"
                f"windows {c['windows']} | stability {c['stability']} | base {c['base']}\n"
                f"负采样 K={c['neg_k']} λ={c['neg_lam']} | 来源 {src_s} | epochs {c['epochs']}\n"
                f"训练语料 {self.m['corpus']['n_tokens']} 词 / {self.m['corpus']['n_sentences']} 句")
