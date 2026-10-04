# -*- coding: utf-8 -*-
"""kb 表 + 上文高 att 的预测器（用户方案）

context = 等差加权窗口和(用 kb 向量)  +  上文高att词的 kb 向量加权和

对比 predict.py 的 Predictor：
  旧：context = 末尾 windows 个词的 ka 向量等权求和
  新：context = 末尾 windows 个词的 kb 向量等差加权求和
              + 更早的上文里 att >= base 的词的 kb 向量加权和

本文件有两个预测器：
  KbAttPredictor       纯向量：给下一个词的概率分布（可连成新句子）
  RetrievalPredictor   向量 + 语料检索：给语料里真实出现过的续接片段（infer.py 菜单 5）
"""
import numpy as np
import jieba
import Lpron_core as C


class KbAttPredictor:
    def __init__(self, model, table="kb", base=None):
        self.m = model
        self.cfg = model["config"]
        self.dim = self.cfg["dim"]
        self.windows = self.cfg["windows"]
        self.base = base if base is not None else self.cfg["base"]
        self.inf = C.Inferencer(model).use(table)
        self.vec = self.inf.vec          # kb 向量矩阵
        self.unit = self.inf.unit

    def att_of(self, words):
        """推理期注意力：每个词与它所在窗口的偏离度（越大信息量越高）"""
        w = self.windows
        out = []
        for i, word in enumerate(words):
            if not self.inf.has(word):
                out.append(0.0)
                continue
            tw = C._window_of(words, i, w)
            vs = [self.inf.token(x) for x in tw if self.inf.has(x)]
            if not vs:
                out.append(1.0)
                continue
            # 跟随训练配置选加权函数，别写死 g_linear
            ctx = self.inf.gfun(vs, w, self.dim)
            out.append(C.cosdis(self.inf.token(word), ctx))
        return out

    def context(self, words, use_att=True, att=None, alpha=1.0):
        """上下文向量 = 等差加权窗口 + 上文高att累加"""
        w = self.windows
        idx = self.m["_index"]

        # 1) 窗口部分：末尾 w 个词，等差加权（最近的权重 1）
        tail = words[-w:]
        vs = []
        for x in tail:
            if self.inf.has(x):
                vs.append(self.vec[idx[x]])
        if not vs:
            return None
        ctx_w = C._norm_to(
            sum(v * wi for v, wi in zip(vs, np.arange(1, len(vs)+1)/len(vs))),
            self.dim)

        if not use_att:
            return C._norm_to(ctx_w, self.dim)

        # 2) 上文高 att 部分：窗口之外的更早词，只取 att >= base
        if att is None:
            att = self.att_of(words)
        head = words[:-len(tail)] if len(words) > len(tail) else []
        a_head = att[:len(head)] if len(head) <= len(att) else att
        acc = np.zeros(self.dim)
        cnt = 0
        for x, a in zip(head, a_head):
            if a >= self.base and self.inf.has(x):
                acc = acc + self.vec[idx[x]] * a
                cnt += 1
        if cnt:
            ctx_w = ctx_w + alpha * C._norm_to(acc / cnt, self.dim)
        return C._norm_to(ctx_w, self.dim)

    def next_probs(self, words, temperature=15.0, topk=30, exclude=None,
                   use_att=True, alpha=1.0):
        ctx = self.context(words, use_att=use_att, alpha=alpha)
        if ctx is None:
            return []
        cos = self.unit @ (ctx / np.linalg.norm(ctx))
        idx = np.argsort(-cos)[:topk]
        logits = cos[idx] * temperature
        p = np.exp(logits - logits.max())
        if exclude:
            mask = np.ones(len(idx), dtype=bool)
            for j, wi in enumerate(idx):
                if self.m["words"][wi] in exclude:
                    mask[j] = False
            if mask.any():
                p = p * mask
        if p.sum() <= 0:
            p = np.exp(logits - logits.max())
        p = p / p.sum()
        order = np.argsort(-p)
        return [(self.m["words"][idx[j]], float(p[j])) for j in order]

    def greedy(self, sentence, max_len=12, use_att=True):
        words = [w for w in jieba.lcut(sentence, cut_all=False) if w.strip()]
        for _ in range(max_len):
            c = self.next_probs(words, exclude=set(words), use_att=use_att)
            if not c:
                break
            words.append(c[0][0])
        return "".join(words)


# ============================================================ 语料检索式续写
class RetrievalPredictor:
    """语料检索式续写器 —— infer.py 菜单 5 用的就是它。

    【补这个类的原因】infer.py 里 import 的是 predict.RetrievalPredictor，
    但本文件原先只有 KbAttPredictor，导入失败被 try/except 静默吞掉，
    于是菜单 5 无论有没有 tran.txt 都打印"续写需要 tran.txt"。

    【为什么用"检索 + 向量"，而不是纯向量生成】
    词向量只告诉你"语义上接得上哪些词"，续写要的却是语料里真实出现过的搭配。
    所以主路子是字面检索：在原文里找前缀出现过的位置，取它后面的片段当候选，
    再按"与前缀的语义相关度 + 出现次数"排序；字面没命中才退回语义检索，
    仍不命中才用 KbAttPredictor 逐词兜底。这样候选永远是语料里的原句片段。
    """

    PUNCT_END = "。！？；!?;\n"

    def __init__(self, model, text, table="kb"):
        self.m = model
        self.cfg = model["config"]
        self.dim = self.cfg["dim"]
        self.inf = C.Inferencer(model).use(table)
        self.unit = self.inf.unit
        self.raw = text or ""

        self.sents = [s for s in (x.strip() for x in C.split_sentences(self.raw)) if s]
        self.sent_toks = [[w for w in jieba.lcut(s, cut_all=False) if w.strip()]
                          for s in self.sents]
        # 句子向量预计算：语义兜底时要跟全部句子比一次，别每次现算
        self.sent_vec = []
        for toks in self.sent_toks:
            vs = [self.inf.vec[self.m["_index"][w]] for w in toks if self.inf.has(w)]
            self.sent_vec.append(C._norm_to(np.mean(vs, axis=0), self.dim) if vs else None)
        self._kb = None

    # ---------------- 工具 ----------------
    def _vec(self, text):
        """文本 -> 归一化向量；整段都不在词表时返回 (None, [])"""
        return self.inf.text_vector(text)

    def _cut_tail(self, tail, span):
        """把前缀后面的原文截成一段像样的话：遇到句末标点就停，最多 span 字"""
        out = []
        for ch in tail[:span]:
            out.append(ch)
            if ch in self.PUNCT_END:
                break
        return "".join(out).strip()

    def _score(self, pv, tail, cnt):
        """候选打分：与前缀的余弦为主，出现次数做轻微加权（0.02/次）"""
        v, _ = self._vec(tail)
        if v is None or pv is None:
            return float(cnt) * 0.02
        return float(np.dot(v, pv)) + 0.02 * cnt

    # ---------------- 主入口 ----------------
    def complete(self, prefix, n=3, span=24):
        """返回至多 n 条续接候选（每条都是 prefix + 语料里出现过的后文）"""
        prefix = (prefix or "").strip()
        if not prefix or not self.raw:
            return []
        pv, _ = self._vec(prefix)

        # 1) 字面检索：原文里这个前缀出现过的地方
        hits = {}
        start = 0
        while True:
            i = self.raw.find(prefix, start)
            if i < 0:
                break
            tail = self._cut_tail(self.raw[i + len(prefix):], span)
            if tail:
                hits[tail] = hits.get(tail, 0) + 1
            start = i + 1
            if len(hits) >= 200:          # 极高频前缀别扫全篇
                break

        out = []
        if hits:
            ranked = sorted(((self._score(pv, t, c), t) for t, c in hits.items()),
                            key=lambda x: -x[0])
            out = [prefix + t for _, t in ranked[:n]]

        # 2) 语义兜底：字面没出现过 -> 给语义最接近的语料原句
        if len(out) < n and pv is not None:
            sims = [(float(np.dot(sv, pv)), j) for j, sv in enumerate(self.sent_vec)
                    if sv is not None]
            sims.sort(key=lambda x: -x[0])
            seen = set(out)
            for _s, j in sims:
                cand = self.sents[j]
                if cand in seen or len(cand) <= len(prefix):
                    continue
                seen.add(cand)
                out.append(cand)
                if len(out) >= n:
                    break

        # 3) 字符兜底：前缀里全是词表外的词（pv 为 None），向量帮不上忙
        if not out and pv is None:
            out = self._char_fallback(prefix, n)

        # 4) 词级兜底：语料里完全没线索 -> 用 kb 表逐词接下去
        if not out:
            out = self._vector_complete(prefix)
        return out[:n]

    def _char_fallback(self, prefix, n):
        """前缀整体没出现、词向量也不认识它 -> 找含其任意字的原句

        （OOV 是这套模型的硬伤：没有子词/字符级向量，新词只能靠字面线索）
        """
        chars = [c for c in prefix if c.strip()]
        scored = []
        for j, s in enumerate(self.sents):
            c = sum(1 for ch in chars if ch in s)
            if c:
                scored.append((c, len(s), j))       # 命中字多、句子短的优先
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self.sents[j] for _c, _l, j in scored[:n]]

    def _vector_complete(self, prefix, max_words=6):
        """纯向量续写：kb 表上一步步取概率最高的词拼起来"""
        try:
            if self._kb is None:
                self._kb = KbAttPredictor(self.m, table=self.inf.table)
            ws = [w for w in jieba.lcut(prefix, cut_all=False) if w.strip()]
            tail = []
            for _ in range(max_words):
                c = self._kb.next_probs(ws, exclude=set(ws))
                if not c:
                    break
                ws.append(c[0][0])
                tail.append(c[0][0])
            return [prefix + "".join(tail)] if tail else []
        except Exception as e:
            print(f"  [提示] 向量续写失败：{e}")
            return []


# ============================================================ 真·生成（kb 递推的推理期复刻）
class KbGenerator:
    """把 get_kb 的 running 递推搬到推理期，做【有状态】的逐词生成。

    训练期的原式（Lpron_core.get_kb）：
        a_i        = gfun(窗口内 ka 向量)              # 窗口部分用 ka
        kb[w_i]    = a_i + running_i                  # 不归一化，幅度比即话语权
        running_{i+1} = decay * running_i + kb[w_i] * att_i   (att_i >= base 才累加)
        att_i      = cosdis(ka[w_i], ka 窗口上下文)    # att 在 ka 空间算

    KbAttPredictor 把这三件事都做走了样：窗口用了 kb、running 没有跨步递推、
    att 算在 kb 空间。这里按原式重写，区别：

      * 状态：running 随生成过程逐词演化，高信息量(att>=base)的词被累积进来，
        后续每个候选词都带着这段"读过的上文"参与打分 —— 这才是模型对每个词的理解，
        而不是从语料里挑一句现成的。
      * 因果：预测第 i 个词只用前 windows 个词。这不是近似，训练期 _window_of
        对 i>=windows 本来就返回 words[i-windows:i]。
      * 候选：ctx 落在 kb 空间（kb[w] 本身就是"窗口+上文"的复合），
        所以与 kb 单位向量比余弦。
    """

    def __init__(self, model, base=None, decay=None, window_table="ka"):
        self.m = model
        self.cfg = model["config"]
        self.dim = self.cfg["dim"]
        self.windows = self.cfg["windows"]
        self.base = base if base is not None else self.cfg["base"]
        self.decay = decay if decay is not None else self.cfg.get("kb_decay", 1.0)
        self.ka = C.Inferencer(model).use("ka")
        self.kb = C.Inferencer(model).use("kb")
        self.unit = self.kb.unit
        self.gfun = self.ka.gfun
        # "ka" = 忠实复刻；"kb" = 旧 KbAttPredictor 的写法（窗口也用 kb），留作对照
        self.win = self.ka if window_table == "ka" else self.kb

    # ---------------- 原子操作 ----------------
    def _win_words(self, words, i):
        """因果窗口：预测位置 i 时看它前面 windows 个词（与训练一致）"""
        return words[max(0, i - self.windows):i]

    def _tok(self, inf, w):
        v = inf.vec[self.m["_index"][w]]
        return np.concatenate([v, [1.0]])

    def _a(self, words, i):
        """窗口部分 a_i = gfun(窗口内向量)，零窗口返回零向量"""
        tw = self._win_words(words, i)
        ts = [self._tok(self.win, x) for x in tw if self.win.has(x)]
        if not ts:
            return np.zeros(self.dim)
        return self.gfun(ts, self.windows, self.dim)[:-1]

    def att_at(self, words, i):
        """位置 i 的推理期 att：该词与它因果窗口上下文的偏离度（ka 空间）。

        语义与训练一致 —— 偏离越大=这个词带来的新信息越多=越该被累积进 running。
        窗口为空时返回 1.0（等同训练里"首次出现=全新信息"）。
        """
        w = words[i]
        if not self.ka.has(w):
            return 0.0
        tw = self._win_words(words, i)
        ts = [self._tok(self.ka, x) for x in tw if self.ka.has(x)]
        if not ts:
            return 1.0
        ctx = self.gfun(ts, self.windows, self.dim)
        return C.cosdis(self._tok(self.ka, w), ctx)

    def _ctx(self, words, running):
        """预测下一个词的上下文 = a + running（照抄训练式，不做额外归一化）"""
        return self._a(words, len(words)) + running

    def _probs(self, ctx, temperature=15.0, topk=30, ban=None):
        n = np.linalg.norm(ctx)
        if n == 0:
            return []
        cos = self.unit @ (ctx / n)
        idx = np.argsort(-cos)[:topk]
        logits = cos[idx] * temperature
        p = np.exp(logits - logits.max())
        if ban:
            ban_idx = {self.m["_index"][w] for w in ban if w in self.m["_index"]}
            for j, wi in enumerate(idx):
                if wi in ban_idx:
                    p[j] = 0.0
        s = p.sum()
        if s <= 0:                       # 候选被 ban 光了，就退回未屏蔽的分布
            p = np.exp(logits - logits.max()); s = p.sum()
        p = p / s
        return [(self.m["words"][idx[j]], float(p[j])) for j in np.argsort(-p)]

    def prime(self, words):
        """用前缀把 running 跑起来：前缀里 att>=base 的词按原式累加。

        这是"模型读了一遍前缀"的状态，也是生成区别于检索的关键。
        """
        running = np.zeros(self.dim)
        for i, w in enumerate(words):
            if not self.kb.has(w):
                continue
            a = self.att_at(words, i)
            if a >= self.base:
                running = running * self.decay + self.kb.vec[self.m["_index"][w]] * a
                running = C._norm_to(running, self.dim)
        return running

    def _advance(self, running, words, new_word):
        """生成了一个新词后推进 running（与 prime 同一套递推）"""
        if not self.kb.has(new_word):
            return running
        seq = words + [new_word]
        a = self.att_at(seq, len(seq) - 1)
        if a >= self.base:
            running = running * self.decay + self.kb.vec[self.m["_index"][new_word]] * a
            running = C._norm_to(running, self.dim)
        return running

    # ---------------- 生成 ----------------
    def generate(self, prefix, max_new=8, beam=1, temperature=15.0, topk=30,
                 stop=("。", "！", "？"), no_repeat=2, seed=None):
        """返回 [(文本, 对数似然), ...]，按分数降序，最多 beam 条。

        beam=1 即贪心。no_repeat=n 表示禁止重复最近 n 个位置出现过的词
        （比旧 greedy 的 exclude=set(words) 温和：那个把高频虚词也一次性用掉了）。

        seed：beam 搜索本身是确定性的，seed 只用于【并列候选】的打破
        （抖动 1e-9，小于任何真实概率差，正常排序不受影响）。
        同一 seed → 结果可复现。
        """
        rng = np.random.default_rng(seed)
        words = [w for w in jieba.lcut(prefix, cut_all=False) if w.strip()]
        if not any(self.kb.has(w) for w in words):
            return []
        running = self.prime(words)
        beams = [(0.0, words, running, list(words))]

        for _ in range(max_new):
            nxt = []
            for score, seq, run, allw in beams:
                if seq and seq[-1] in stop:
                    nxt.append((score, seq, run, allw))
                    continue
                ban = set(allw[-no_repeat:]) if no_repeat else None
                ps = self._probs(self._ctx(seq, run), temperature, topk, ban)
                if not ps:
                    nxt.append((score, seq, run, allw))
                    continue
                # 并列打破：抖动 1e-9，只在概率完全相等时改变先后，正常排序不受影响
                ps = [(w, p + float(rng.random()) * 1e-9) for w, p in ps]
                for w, p in ps[:max(beam, 1)]:
                    nxt.append((score + float(np.log(p + 1e-12)),
                                seq + [w], self._advance(run, seq, w), allw + [w]))
            if not nxt:
                break
            nxt.sort(key=lambda x: -x[0])
            # 去重：保留同词序列里分最高的
            seen, beams = set(), []
            for s, seq, run, allw in nxt:
                key = tuple(seq)
                if key in seen:
                    continue
                seen.add(key)
                beams.append((s, seq, run, allw))
                if len(beams) >= beam:
                    break
        out = []
        for score, seq, _r, allw in beams:
            new = "".join(seq[len(words):])
            if new:
                out.append((prefix + new, score))
        out.sort(key=lambda x: -x[1])
        return out[:max(beam, 1)]

    def sample(self, prefix, max_new=8, temperature=15.0, topk=30, seed=None,
               stop=("。", "！", "？"), no_repeat=2):
        """按概率采样一条（temperature 越低越随机，越高越贪心）"""
        rng = np.random.default_rng(seed)
        words = [w for w in jieba.lcut(prefix, cut_all=False) if w.strip()]
        if not any(self.kb.has(w) for w in words):
            return prefix
        running = self.prime(words)
        for _ in range(max_new):
            ban = set(words[-no_repeat:]) if no_repeat else None
            ps = self._probs(self._ctx(words, running), temperature, topk, ban)
            if not ps:
                break
            p = np.array([x[1] for x in ps]); p = p / p.sum()
            w = ps[int(rng.choice(len(ps), p=p))][0]
            words.append(w)
            running = self._advance(running, words[:-1], w)
            if w in stop:
                break
        return "".join(words)
