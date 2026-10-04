#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""续写方式横评：检索式 vs 生成式（在【训练语料没见过】的前缀上）

用法：
    python eval_gen.py                # 自动造合成语料，8/2 分训练测试
    python eval_gen.py --real          # 用真实 tran.txt 做留出法（后 20% 句当测试）

为什么必须看"没见过的前缀"：
    检索式在字面命中时是【直接抄原文】，只要前缀在语料里出现过它就必胜，
    这个比较没有意义。真正能区分挑选与生成的，是语料里没出现过的组合。

指标（gold = 测试句里前缀之后、到句末标点为止的词）：
    P/R/F1   生成词与 gold 的词级重合（词袋）
    Acc@1    第一个生成词是否命中 gold 的第一个词（最严格的下一个词预测）
    HitRate  至少命中一个 gold 词的比例
"""
import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Lpron_core as C
import predict


# ---------------------------------------------------------------- 合成语料
def synth(n=800, seed=3):
    """模板语法造语料。刻意把组合空间开得很大（约 2 万种），
    这样测试句里会有大量【训练时没出现过】的前缀 —— 否则检索式直接抄原文必胜，
    比较就没有意义。"""
    rnd = random.Random(seed)
    subj = ["今天", "明天", "昨天", "早晨", "傍晚", "山里", "海边", "城里", "周末",
            "妈妈", "爸爸", "孩子", "老师", "我们", "他们", "我", "她", "小猫",
            "邻居", "朋友"]
    place = ["", "", "在厨房", "在学校", "在路上", "在公园", "在家里"]
    verb = ["天气很好", "可能下雨", "有点凉", "变化很快", "很舒服",
            "做饭", "学会了做饭", "要小心火", "准备了食材", "洗菜",
            "学习新语言", "每天学习", "学习知识", "感到很充实", "复习功课",
            "计划去旅行", "喜欢旅行", "拍照", "准备出发", "收拾行李",
            "看电影", "听音乐", "写信", "跑步", "种花", "修自行车",
            "开会", "休息", "买东西", "打扫房间", "练琴", "画画"]
    obj = ["", "", "一个小时", "很多次", "给朋友", "慢慢来", "认真地",
           "和我一起", "在晚上", "为了考试"]
    tail = ["，因为时间还早。", "，大家都很开心。", "，这是最重要的事情。",
            "，我们要认真对待。", "，需要耐心和坚持。", "，可惜我没赶上。",
            "，后来就习惯了。", "，明天再说吧。", "。", "！"]
    out = []
    seen = set()
    guard = 0
    while len(out) < n and guard < n * 50:
        guard += 1
        s = (rnd.choice(subj) + rnd.choice(place) + rnd.choice(verb)
             + rnd.choice(obj) + rnd.choice(tail))
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def gold_split(sent, k):
    """取前 k 个词当前缀，返回 (前缀字符串, gold 词集合, gold 首词)"""
    import jieba
    toks = [t for t in jieba.lcut(sent) if t.strip()]
    if len(toks) <= k + 1:
        return None
    pref = "".join(toks[:k])
    rest = [t for t in toks[k:] if t not in "，。！？、"]
    if not rest:
        return None
    return pref, set(rest), rest[0]


def prf(pred_set, gold_set):
    if not pred_set:
        return 0.0, 0.0, 0.0
    hit = len(pred_set & gold_set)
    p = hit / len(pred_set)
    r = hit / len(gold_set)
    f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return p, r, f


def run_case(fn, pref, mlen=5):
    """统一调用各种续写器，返回 (生成文本, 生成词集合, 生成首词)"""
    import jieba
    try:
        txt = fn(pref)
    except Exception as e:
        print(f"    [异常] {type(e).__name__}: {e}")
        return "", set(), ""
    if not txt:
        return "", set(), ""
    cont = txt[len(pref):]
    toks = [t for t in jieba.lcut(cont) if t.strip() and t not in "，。！？、"]
    return txt, set(toks[:mlen]), (toks[0] if toks else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="用真实 tran.txt 做留出")
    ap.add_argument("--n", type=int, default=800, help="合成语料句数")
    ap.add_argument("--k", type=int, default=2, help="前缀词数")
    ap.add_argument("--mlen", type=int, default=5, help="最多比较前几个生成词")
    ap.add_argument("--cseed", type=int, default=3, help="合成语料种子")
    args = ap.parse_args()

    HERE = os.path.dirname(os.path.abspath(__file__))
    if args.real:
        p = os.path.join(HERE, "tran.txt")
        if not os.path.exists(p):
            print("没有 tran.txt，改用合成语料")
            sents = synth(args.n, args.cseed)
        else:
            sents = [s.strip() for s in C.split_sentences(open(p, encoding="utf-8").read()) if s.strip()]
    else:
        sents = synth(args.n, args.cseed)

    n_test = max(20, len(sents) // 5)
    train_s, test_s = sents[:-n_test], sents[-n_test:]
    print(f"语料 {len(sents)} 句 | 训练 {len(train_s)} 句 | 测试 {len(test_s)} 句")

    text = "\n".join(train_s)
    np.random.seed(1)
    model = C.train(text, dim=64, windows=5, epochs=4)
    print(f"词表 {len(model['words'])} 词")

    rp = predict.RetrievalPredictor(model, text, table="kb")
    old = predict.KbAttPredictor(model, table="kb")
    g_ka = predict.KbGenerator(model, window_table="ka")    # 忠实复刻
    g_kb = predict.KbGenerator(model, window_table="kb")    # 窗口也用 kb（旧写法）
    # 消融：base=2.0 使 att 永远达不到阈值，running 恒为 0
    # → 退化为"只有窗口、没有上文记忆"，用来量 att 递推到底贡献了多少
    g_norun = predict.KbGenerator(model, window_table="ka", base=2.0)
    vocab = model["words"]

    # 只看测试句里【训练语料没出现过】的前缀
    cases = []
    for s in test_s:
        r = gold_split(s, args.k)
        if r and r[0] not in text:
            cases.append(r)
    print(f"未见过的前缀 {len(cases)} 个（测试集共 {n_test} 句）\n")
    if not cases:
        print("没有可用的未见过前缀，增大 --n 或换 --k")
        return 1

    def f_retr(p):
        r = rp.complete(p, n=1)
        return r[0] if r else ""

    def f_old(p):
        return old.greedy(p, max_len=args.mlen)

    def f_gen1(p):
        r = g_ka.generate(p, max_new=args.mlen, beam=1)
        return r[0][0] if r else ""

    def f_gen4(p):
        r = g_ka.generate(p, max_new=args.mlen, beam=4)
        return r[0][0] if r else ""

    def f_genkb(p):
        r = g_kb.generate(p, max_new=args.mlen, beam=4)
        return r[0][0] if r else ""

    def f_norun(p):
        r = g_norun.generate(p, max_new=args.mlen, beam=4)
        return r[0][0] if r else ""

    def f_rand(p):
        r = np.random.default_rng(abs(hash(p)) % (2**32))
        return p + "".join(r.choice(vocab, size=args.mlen))

    name_fn = [
        ("随机基线", f_rand),
        ("检索式(Retrieval)", f_retr),
        ("旧 greedy(KbAtt)", f_old),
        ("生成 beam1 窗口=ka", f_gen1),
        ("生成 beam4 窗口=ka", f_gen4),
        ("生成 beam4 窗口=kb", f_genkb),
        ("生成 beam4 无running", f_norun),
    ]

    print(f"{'方式':<22}{'P':>7}{'R':>7}{'F1':>7}{'Acc@1':>8}{'HitRate':>9}  样例")
    print("-" * 100)
    for name, fn in name_fn:
        P = R = F = A = H = 0.0
        sample = ""
        for pref, gold, first in cases:
            txt, pset, pfirst = run_case(fn, pref, args.mlen)
            p, r, f = prf(pset, gold)
            P += p; R += r; F += f
            A += 1.0 if pfirst == first else 0.0
            H += 1.0 if (pset & gold) else 0.0
            if not sample and txt:
                sample = txt[:22]
        n = len(cases)
        print(f"{name:<22}{P/n:>7.3f}{R/n:>7.3f}{F/n:>7.3f}{A/n:>8.3f}{H/n:>9.3f}  {sample}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
