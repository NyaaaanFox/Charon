#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理脚本 —— 加载 embedding.npz / embedding.json，提供交互式操作菜单。

用法：
    python infer.py                      # 进入交互菜单
    python infer.py --nearest 猫 -k 10   # 命令行直接查
    python infer.py --info               # 只看模型信息

菜单选项：
    1 最近邻        查一个词周围最相似的 n 个词
    2 查看向量      打印某个词的原始向量（或前若干维）
    3 词相似度      两个词的余弦相似度
    4 文本向量      一句话的整体向量（词向量均值）
    5 句子续写      输入前缀，返回语料中出现过的续接
    6 词表浏览      按频次列出词表
    7 模型信息      超参数与训练统计
    0 退出
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Lpron_core as C

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "embedding")


# ---------------------------------------------------------------- 工具
def prompt(msg="> "):
    try:
        return input(msg).strip()
    except (EOFError, KeyboardInterrupt):
        return "0"


def show_nearest(inf, w, k):
    if not inf.has(w):
        print(f"  词表中没有「{w}」")
        close = [x for x in inf.m["words"] if w and w[0] == x[0]][:5]
        if close:
            print(f"  你要找的是不是：{' / '.join(close)}")
        return
    print(f"  「{w}」的最近邻：")
    for x, s in inf.nearest(w, k):
        print(f"    {s:+.4f}  {x}")


def show_vec(inf, w, head):
    if not inf.has(w):
        print(f"  词表中没有「{w}」")
        return
    v = inf.token(w)             # dim+1，最后一维是训练频次
    print(f"  「{w}」 语义维 {len(v)-1} + 频次维 1 | 频次={v[-1]:.0f}")
    if head and head < len(v):
        print("   前 %d 维：" % head)
        for i in range(0, head, 8):
            row = "  ".join(f"{x:+.3f}" for x in v[i:i + 8])
            print(f"     [{i:>3}] {row}")
    else:
        for i in range(0, len(v), 8):
            row = "  ".join(f"{x:+.3f}" for x in v[i:i + 8])
            print(f"     [{i:>3}] {row}")


def show_text_vec(inf, text, k=8, decenter=True):
    """文本向量 = 词向量均值。

    decenter: 去掉"全体词向量的共同方向"再比较。
      不开的话，均值向量天然靠近全局质心，算出来的近邻全是"不/了/也/变得"
      这类到处出现的虚词（实测如此）——它们和任何文本都"相似"。
      去掉共同成分后才反映这段文本自身的语义。
    """
    v, hit = inf.text_vector(text)
    if v is None:
        print("  这些词都不在词表里")
        return
    print(f"  分词：{' / '.join(hit)}")
    print(f"  文本向量 dim={len(v)}  范数={np.linalg.norm(v):.4f}")
    if 0 < len(v) <= 16:
        print("    " + "  ".join(f"{x:+.3f}" for x in v))
    q = v / (np.linalg.norm(v) + 1e-9)
    if decenter:
        mu = inf.unit.mean(axis=0)                 # 全局共同方向
        mu = mu / (np.linalg.norm(mu) + 1e-9)
        q = q - mu * float(np.dot(q, mu))
        q = q / (np.linalg.norm(q) + 1e-9)
        print("  （已去掉全体词向量的共同方向，否则近邻会全是虚词）")
    sims = inf.unit @ q
    order = np.argsort(-sims)[:k]
    print(f"  与该文本最接近的 {k} 个词：")
    for j in order:
        print(f"    {sims[j]:+.4f}  {inf.m['words'][j]}")


def show_sim(inf, a, b):
    s = inf.sim(a, b)
    if s is None:
        miss = [x for x in (a, b) if not inf.has(x)]
        print(f"  不在词表里：{' / '.join(miss)}")
        return
    print(f"  cos({a}, {b}) = {s:+.4f}")
    print(f"  参考：{a} 的最近邻是 " +
          ", ".join(x for x, _ in inf.nearest(a, 3)))


def show_complete(model, inf, rp, text, n):
    if rp is None:
        print("  续写需要 tran.txt（训练语料）")
        return
    for i, r in enumerate(rp.complete(text, n=n), 1):
        print(f"  候选{i}：{r}")


def show_generate(model, text, n=None, beam=4, mlen=8, seed=None):
    """模型递推生成：不用语料，纯靠 kb 的 running 递推往前接词。

    与菜单 5 的区别：5 是从语料里【挑】现成的句子，这里是模型【生成】，
    候选词可能组合出语料里没出现过的搭配。
    """
    try:
        import predict
    except Exception as e:
        print(f"  生成模块未加载：{e}")
        return
    g = predict.KbGenerator(model, window_table="ka")
    res = g.generate(text, max_new=mlen, beam=beam, seed=seed)
    if not res:
        print("  前缀里的词都不在词表，生成不了（模型没有 OOV 处理）")
        return
    if n and n > 0:
        res = res[:n]                     # 命令行 -n / 菜单候选数在此生效
    print(f"  递推生成（beam={beam}，最多 {mlen} 个新词）：")
    for i, (s, sc) in enumerate(res, 1):
        print(f"  候选{i}：{s}    [logp {sc:+.2f}]")


def _read_text(path):
    """读语料：UTF-8 优先，失败再试 GBK 系（与 tran.py 的读取顺序保持一致）"""
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk", "gb2312"):
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    # 都失败就硬读，乱码总比直接崩好
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def show_vocab(model, n):
    words = model["words"]
    freq = list(model.get("ka_freq", [0] * len(words)))
    pairs = sorted(zip(words, freq), key=lambda x: -x[1])
    print(f"  词表共 {len(words)} 词，按频次前 {n}：")
    for i, (w, f) in enumerate(pairs[:n]):
        print(f"    {w:<10} {f}")
        if (i + 1) % 10 == 0 and i + 1 < n:
            if prompt("  回车继续，q 停止 > ").lower() == "q":
                break


def show_info(model, inf):
    print("  " + inf.info().replace("\n", "\n  "))
    p = os.path.join(HERE, "params.txt")
    if os.path.exists(p):
        print("\n  params.txt 存在，可用文本编辑器查看完整训练报告")


MENU = """
┌──────────────────────────────────────────┐
│  1 最近邻      查一个词周围最相似的词     │
│  2 查看向量    打印某个词的原始向量       │
│  3 词相似度    两个词的余弦相似度         │
│  4 文本向量    一句话的整体向量           │
│  5 句子续写    输入前缀，返回候选续接     │
│  6 词表浏览    按频次列出词表             │
│  8 递推生成    模型自己往前接词(非检索)   │
│  7 模型信息    超参数与训练统计           │
│  0 退出                                  │
└──────────────────────────────────────────┘"""


def interactive(model, inf, rp):
    print(MENU)
    while True:
        c = prompt("\n选择 > ")
        if c in ("0", "q", "quit", "exit"):
            break
        if c == "1":
            w = prompt("  词 > ")
            k = prompt("  个数(默认10) > ") or "10"
            show_nearest(inf, w, int(k) if k.isdigit() else 10)
        elif c == "2":
            w = prompt("  词 > ")
            h = prompt("  只看前几维(回车=全部) > ")
            show_vec(inf, w, int(h) if h.isdigit() else None)
        elif c == "3":
            a = prompt("  词1 > ")
            b = prompt("  词2 > ")
            show_sim(inf, a, b)
        elif c == "4":
            t = prompt("  一句话 > ")
            show_text_vec(inf, t)
        elif c == "5":
            t = prompt("  前缀 > ")
            n = prompt("  候选数(默认3) > ") or "3"
            show_complete(model, inf, rp, t, int(n) if n.isdigit() else 3)
        elif c == "6":
            n = prompt("  显示前几个(默认30) > ") or "30"
            show_vocab(model, int(n) if n.isdigit() else 30)
        elif c == "7":
            show_info(model, inf)
        elif c == "8":
            t = prompt("  前缀 > ")
            b = prompt("  beam(默认4) > ") or "4"
            m = prompt("  最多新词数(默认8) > ") or "8"
            beam_n = int(b) if b.isdigit() else 4
            # 注意顺序：第三个位置参数是候选数 n，不是 beam
            # （原写法把 beam 输入当成 n 传进去，beam 永远取默认值）
            show_generate(model, t, n=beam_n, beam=beam_n,
                          mlen=int(m) if m.isdigit() else 8)
        else:
            print("  无效选项")
    print("再见")


def main():
    ap = argparse.ArgumentParser(description="加载嵌入表做查询")
    ap.add_argument("--nearest", help="查最近邻的词")
    ap.add_argument("-k", type=int, default=10, help="最近邻个数")
    ap.add_argument("--sim", nargs=2, help="两个词的相似度")
    ap.add_argument("--text", help="一句话的文本向量")
    ap.add_argument("--complete", help="句子续写的前缀（检索式）")
    ap.add_argument("--generate", help="递推生成的前缀（模型生成）")
    ap.add_argument("--beam", type=int, default=4, help="生成 beam 宽度")
    ap.add_argument("-n", type=int, default=3, help="续写候选数")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--table", default="ka", choices=["ka", "kb"])
    args = ap.parse_args()

    if not os.path.exists(MODEL + ".json"):
        print(f"找不到嵌入表：{MODEL}.json")
        print("请先运行：python tran.py")
        return 1

    model = C.load_model(MODEL)
    inf = C.Inferencer(model).use(args.table)

    rp = None
    cp = os.path.join(HERE, "tran.txt")
    if os.path.exists(cp):
        try:
            import predict
            rp = predict.RetrievalPredictor(model, _read_text(cp), table=args.table)
        except Exception as e:
            # 别再把错误吞掉：predict.py 里缺类、语料编码不对，都会在这里暴露
            print(f"[提示] 续写模块未加载（菜单 5 不可用）：{type(e).__name__}: {e}")
            rp = None

    if args.info:
        show_info(model, inf)
    elif args.nearest:
        show_nearest(inf, args.nearest, args.k)
    elif args.sim:
        show_sim(inf, args.sim[0], args.sim[1])
    elif args.text:
        show_text_vec(inf, args.text)
    elif args.complete:
        show_complete(model, inf, rp, args.complete, args.n)
    elif args.generate:
        show_generate(model, args.generate, args.n, beam=args.beam)
    else:
        interactive(model, inf, rp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
