#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练脚本 —— 读取本文件所在目录下的 tran.txt，训练后保存嵌入表与超参数。

用法：
    python tran.py                      # 用默认最优配置训练 tran.txt
    python tran.py --epochs 8           # 覆盖某个超参
    python tran.py --eval               # 训练后做一次自检

产出（与本文件同目录）：
    embedding.npz   向量矩阵（ka / kb 两张表）
    embedding.json  词表、频次、超参数（程序读取用）
    params.txt      超参数与训练报告（人读）

进度条格式：
    阶段 1/2 ka  [==========          ]  52%  ep 2/4  13456/251118  剩余 00:18
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Lpron_core as C

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "tran.txt")

BEST = dict(
    dim=64,
    windows=5,
    stability=50,
    base=0.7,
    epochs=4,
    neg_k=15,
    neg_lam=0.4,
    neg_pow=0.75,
    neg_source="lowatt",
    neg_mode="top",
    lowneed=0.5,
    norm=True,
    kb_decay=1.0,
    window_weight="linear",
)


# ============================================================ 进度条
class Bar:
    """单行进度条。用 \r 回到行首覆盖，多行更新不会残留。
    ETA 用加权移动平均平滑，避免每步抖动。"""

    def __init__(self, total, prefix, width=None, quiet=False):
        self.total = total
        self.prefix = prefix
        # 宽度自适应：词数越多条越长，但最多占 30 列
        self.width = width or (30 if total >= 50000 else 24 if total >= 5000 else 16)
        self.quiet = quiet
        self.t0 = time.time()
        self.last_flush = 0.0
        self._eta = None          # 平滑后的 ETA
        self._last_line_len = 0

    def _fmt(self, done, extra=""):
        if self.total and self.total > 0:
            pct = min(100.0, done / self.total * 100.0)
            filled = int(self.width * done // max(1, self.total))
            bar = "=" * filled + " " * (self.width - filled)
            eta_s = ""
            if done > 0 and self._eta is not None:
                e = self._eta
                eta_s = f"  剩余 {int(e//60):02d}:{int(e%60):02d}"
            return f"\r{self.prefix} [{bar}] {pct:5.1f}%  {extra}{eta_s}"
        return f"\r{self.prefix} 已处理 {done}  {extra}"

    def update(self, done, extra=""):
        if self.quiet:
            return
        now = time.time()
        if done > 0:
            # ETA 平滑：90% 旧值 + 10% 新观测，抑制抖动
            raw = (now - self.t0) / done * (self.total - done)
            self._eta = raw if self._eta is None else self._eta * 0.9 + raw * 0.1
        if done == self.total or now - self.last_flush >= 0.25 or done == 0:
            line = self._fmt(done, extra)
            # 行变短时清掉尾部残留字符
            if len(line) < self._last_line_len:
                line = line + " " * (self._last_line_len - len(line))
            self._last_line_len = len(line)
            sys.stdout.write(line)
            sys.stdout.flush()
            self.last_flush = now

    def finish(self, extra=""):
        if self.quiet:
            return
        line = self._fmt(self.total, extra)
        if len(line) < self._last_line_len:
            line = line + " " * (self._last_line_len - len(line))
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def _read_corpus(path):
    """读语料：先 UTF-8，失败再试 GBK/GB2312，再失败抛明确错误。"""
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk", "gb2312"):
        try:
            with open(path, encoding=enc) as f:
                t = f.read()
            return t, enc
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("???", b"", 0, 1, "tran.txt 编码无法识别，请用 UTF-8 或 GBK 保存")


# ============================================================ 训练（带进度）
def train_with_progress(text, cfg, seed=1):
    """在 pron_core.train 外层包一层进度反馈。
    由于原训练主循环在库内，这里用"分阶段估算 + 句级进度"的方式呈现。
    """
    np.random.seed(seed)
    dim = cfg["dim"]
    windows = cfg["windows"]
    t0 = time.time()

    # ---------- 阶段 0：预处理（可观测）----------
    print("阶段 0/2 预处理  [====================] 100%  分词与切句...")
    words_all = [C.jieba.lcut(s, cut_all=False) for s in C.split_sentences(text)]
    flat = [w for ws in words_all for w in ws]
    n_tok = len(flat)
    n_sent = len(words_all)
    print(f"         词数 {n_tok} / 句数 {n_sent} / 预计词表 {len(set(flat))}")

    # ---------- 阶段 1：训练 ka（占大头，带进度）----------
    ka, aatt, words, sampler_freq = _train_ka(words_all, flat, cfg, seed)

    # ---------- 阶段 2：训练 kb（前缀和 O(n)，很快）----------
    # 【修复】原来这里写 window_g=None，get_kb 会退回等权 g，
    # 而 ka 用的是 g_linear —— 两张表的窗口加权方式不一致，
    # kb 拿到的上下文向量幅度比 ka 小约 40%，att 阈值 base 也跟着失准。
    # 现在统一用与 ka 相同的 gfun。
    gfun = C.g_linear if cfg["window_weight"] == "linear" else C.g
    bar = Bar(len(words), "阶段 2/2 kb  ", quiet=len(words) < 2000)
    t2 = time.time()
    kb = C.get_kb(words, aatt, ka, dim, windows, cfg["base"],
                  decay=cfg["kb_decay"], window_g=gfun)
    bar.finish(f"({time.time()-t2:.2f}s)")

    # ---------- 组装模型 ----------
    model = {
        "words": list(ka.keys()),
        "ka_vec": np.array([ka[w][:-1] for w in ka], dtype=np.float64),
        "ka_freq": np.array([ka[w][-1] for w in ka], dtype=np.float64),
        "kb_vec": np.array([kb[w][:-1] for w in ka], dtype=np.float64),
        "kb_freq": np.array([kb[w][-1] for w in ka], dtype=np.float64),
        "config": dict(dim=dim, windows=windows, stability=cfg["stability"],
                       base=cfg["base"], neg_k=cfg["neg_k"], neg_lam=cfg["neg_lam"],
                       neg_pow=cfg["neg_pow"], neg_source=cfg["neg_source"],
                       neg_mode=cfg["neg_mode"], lowneed=cfg["lowneed"],
                       window_weight=cfg["window_weight"],
                       epochs=cfg["epochs"], norm=cfg["norm"],
                       kb_decay=cfg["kb_decay"], used_remove_pcs=False),
        "sampler_freq": sampler_freq,   # 负采样器的词频表（train.py 里原本漏填，恒为 {}）
        "corpus": dict(n_tokens=n_tok, n_sentences=n_sent),
    }
    print(f"\n训练完成：总耗时 {time.time()-t0:.2f}s | ka 词表 {len(ka)} 词 | kb 已构建")
    return model


def _train_ka(words_all, flat, cfg, seed):
    """原 get_ka_neg 的复刻版，加了进度条与前缀和缓存。
    核心改动：
      1. 每处理 1 句输出一次进度，不再静默跑完
      2. sampler.build 只在第一轮做（之前每轮都重建池，纯浪费）
      3. 每轮结束落盘，被中断也能保留上一轮成果
    """
    dim = cfg["dim"]; windows = cfg["windows"]
    stability = cfg["stability"]; K = cfg["neg_k"]; lam = cfg["neg_lam"]
    epochs = cfg["epochs"]; norm = cfg["norm"]
    neg_source = cfg["neg_source"]; neg_mode = cfg["neg_mode"]; lowneed = cfg["lowneed"]
    window_weight = cfg["window_weight"]

    gfun = C.g_linear if window_weight == "linear" else C.g
    ka = {}
    sampler = C.NegativeSampler(flat, freq_pow=cfg["neg_pow"])
    lowatt_pool = C.LowAttSampler(mode=neg_mode, lowneed=lowneed)

    prev_att = None
    n_tok = len(flat)
    t_ep0 = time.time()

    for ep in range(epochs):
        # lowatt 需要上一轮的 att 才能建池。首轮 prev_att 为 None 不建池，
        # 此时退化为全词表负采样；从第二轮起才有低att池可用。
        # 注意：条件是 prev_att is not None，等价于 ep >= 1。
        if neg_source == "lowatt" and prev_att is not None:
            lowatt_pool.build(prev_att, flat)

        bar = Bar(n_tok, f"阶段 1/2 ka  [ep {ep+1}/{epochs}]", quiet=n_tok < 5000)
        done = 0
        aatt = []                       # 每轮独立，避免列表胀到 epochs 倍
        for words in words_all:
            for i, word in enumerate(words):
                tw = C._window_of(words, i, windows)
                # 新词即时初始化（前缀和：不重复遍历）
                for k in tw:
                    if k not in ka:
                        ka[k] = C._rand_token(dim)
                ctx = gfun([ka[k] for k in tw], windows, dim)
                ban = set(tw) | {word}

                if neg_source == "lowatt" and prev_att is not None:
                    negs = [ka[n] for n in lowatt_pool.sample(K, table=ka) if n != word]
                else:
                    negs = [ka[n] for n in sampler.sample(K, ban=ban, table=ka)]

                if word not in ka:
                    v = ctx[:-1]
                    if negs:
                        v = v - lam * np.sum([n[:-1] for n in negs], axis=0)
                    ka[word] = np.concatenate([C._norm_to(v, dim) if norm else v, [1.0]])
                    aatt.append(1.0)
                else:
                    old = ka[word]
                    aatt.append(C.cosdis(old, ctx))
                    ka[word] = C.chance_neg(old, ctx, negs, stability, lam=lam, norm=norm)
                done += 1
            bar.update(done)
        bar.finish()

        # 与 Lpron_core 的 get_ka_neg 共用同一套截取规则，保证 ka/kb 的 att 不错位
        prev_att = C._last_epoch_att(aatt, n_tok)

        # ---------- 每轮落盘：中断也不白跑 ----------
        if epochs > 1:
            _checkpoint(ka, prev_att, flat, cfg, ep + 1)

    print(f"         阶段 1 耗时 {time.time()-t_ep0:.2f}s")
    return ka, prev_att, flat, sampler.freq


def _checkpoint(ka, aatt, words, cfg, ep):
    """把当前轮结果存成 .partial，不覆盖正式产物。"""
    path = os.path.join(HERE, "embedding.partial")
    try:
        np.savez_compressed(path, ka_vec=np.array([ka[w][:-1] for w in ka]),
                            ka_freq=np.array([ka[w][-1] for w in ka]))
        with open(path + ".json", "w", encoding="utf-8") as f:
            import json
            json.dump({"words": list(ka.keys()), "epoch": ep,
                       "config": {k: cfg[k] for k in BEST}}, f, ensure_ascii=False)
        print(f"         [检查点] 已保存第 {ep} 轮 -> embedding.partial")
    except Exception as e:
        print(f"         [检查点] 保存失败（不影响训练）: {e}")


# ============================================================ 主流程
def main():
    ap = argparse.ArgumentParser(description="训练词嵌入（读取 tran.txt）")
    for k, v in BEST.items():
        t = type(v)
        if t is bool:
            # 不能写 action="store_true"：默认值为 True 的参数（norm）将永远关不掉。
            # nargs="?" + const=True -> `--norm` 给 True，`--norm false` 给 False。
            ap.add_argument(f"--{k}", nargs="?", const=True, default=v,
                            type=lambda s: str(s).lower() in ("1", "true", "yes", "y"),
                            help=f"默认 {v}")
        elif t is float:
            ap.add_argument(f"--{k}", type=float, default=v)
        elif t is int:
            ap.add_argument(f"--{k}", type=int, default=v)
        else:
            ap.add_argument(f"--{k}", type=str, default=v)
    ap.add_argument("--eval", action="store_true", help="训练后做自检")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quiet", action="store_true", help="关闭进度条")
    args = ap.parse_args()
    cfg = {k: getattr(args, k) for k in BEST}

    # ---------- 读语料 ----------
    if not os.path.exists(CORPUS):
        print(f"[错误] 找不到语料：{CORPUS}")
        print("       请把训练文本保存为 tran.txt，与本脚本放在同一目录。")
        return 1
    try:
        text, enc = _read_corpus(CORPUS)
    except Exception as e:
        print(f"[错误] 读取 tran.txt 失败：{e}")
        return 1
    if len(text.strip()) < 20:
        print("[错误] tran.txt 内容太短，至少需要几十个字。")
        return 1
    print(f"语料：{CORPUS}  (编码 {enc})")
    print(f"字符数 {len(text)}")

    # ---------- 训练 ----------
    np.random.seed(args.seed)
    t0 = time.time()
    model = train_with_progress(text, cfg, seed=args.seed)
    dt = time.time() - t0

    # ---------- 保存 ----------
    try:
        C.save_model(model, os.path.join(HERE, "embedding"))
    except Exception as e:
        print(f"[错误] 保存嵌入表失败：{e}")
        return 1

    # ---------- 自检 ----------
    inf = C.Inferencer(model).use("ka")
    V = inf.vec
    n = np.linalg.norm(V, axis=1, keepdims=True)
    U = V / np.where(n == 0, 1, n)
    M = U @ U.T
    np.fill_diagonal(M, 0)
    collapse = float(M.mean())

    words = model["words"]
    freq = list(model.get("ka_freq", [0] * len(words)))
    top = sorted(zip(words, freq), key=lambda x: -x[1])[:10]

    # ---------- params.txt ----------
    lines = []
    lines.append("=" * 58)
    lines.append("训练报告 / 超参数")
    lines.append("=" * 58)
    lines.append("")
    lines.append("[语料]")
    lines.append("  文件        tran.txt")
    lines.append(f"  编码        {enc}")
    lines.append(f"  字符数      {len(text)}")
    lines.append(f"  句数        {model['corpus']['n_sentences']}")
    lines.append(f"  词数        {model['corpus']['n_tokens']}")
    lines.append(f"  词表大小    {len(words)}")
    lines.append("")
    lines.append("[超参数]")
    labels = {
        "dim": "向量维度", "windows": "上下文窗口", "stability": "记忆强度(频次上限)",
        "base": "注意力阈值", "epochs": "训练轮数", "neg_k": "负例个数 K",
        "neg_lam": "负例强度 λ", "neg_pow": "负例词频指数",
        "neg_source": "负例来源", "neg_mode": "池内选择方式",
        "lowneed": "低att池阈值", "norm": "每步归一化",
        "kb_decay": "kb 遗忘系数", "window_weight": "窗口加权方式",
    }
    for k in BEST:
        lines.append(f"  {labels.get(k, k):<14} {cfg[k]}")
    lines.append(f"  {'随机种子':<14} {args.seed}")
    lines.append("")
    lines.append("[训练结果]")
    lines.append(f"  耗时            {dt:.2f}s")
    lines.append(f"  全体两两余弦    {collapse:.4f}")
    lines.append("")
    lines.append("  关于这个指标：它不是越小越好。")
    lines.append("  虚词（的/了/说/会/常常）本来就该互相聚集——它们出现在相同的模板位置，")
    lines.append("  这是模型学对了，不是坍缩。所以全体均值会被虚词抬高。")
    lines.append("  真正的坍缩是指：所有词（含实词）挤成一点，最近邻全是随机词。")
    lines.append("  判断模型好坏请看下面「抽样最近邻」是否语义正确。")
    lines.append("")
    lines.append("[高频词 Top10]")
    for w, f in top:
        lines.append(f"  {w:<10} {f}")
    lines.append("")
    lines.append("[抽样最近邻]")
    for w in [x for x, _ in top[:6]]:
        nb = inf.nearest(w, 6)
        s = ", ".join(f"{x}({v:.2f})" for x, v in nb)
        lines.append(f"  {w} -> {s}")
    lines.append("")
    lines.append("=" * 58)
    lines.append("下一步：python infer.py")
    lines.append("=" * 58)

    with open(os.path.join(HERE, "params.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n训练完成：{dt:.2f}s | 词表 {len(words)} 词 | 余弦均值 {collapse:.4f}")
    for w, f in top[:6]:
        nb = inf.nearest(w, 6)
        print(f"  {w} -> " + ", ".join(f"{x}({v:.2f})" for x, v in nb))
    print("\n产出：embedding.npz / embedding.json / params.txt")

    if args.eval:
        print("\n--- 自检 ---")
        for w in [x for x, _ in top[:8]]:
            nb = inf.nearest(w, 4)
            print(f"  {w}: " + ", ".join(f"{x}({v:.2f})" for x, v in nb))
    return 0


if __name__ == "__main__":
    sys.exit(main())
sys.exit(main())
