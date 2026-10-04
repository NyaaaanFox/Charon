#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把训练产物 embedding.npz / embedding.json 导出成通用的 word2vec 文本格式（.vec）。

导出的文件可以被 gensim、fastText、Annoy 等工具直接加载：

    from gensim.models import KeyedVectors
    kv = KeyedVectors.load_word2vec_format("ka.vec", binary=False)
    print(kv.most_similar("小猫", topn=10))

用法：
    python export_vec.py                      # 导出 ka 和 kb 两张表
    python export_vec.py --table ka           # 只导出 ka
    python export_vec.py --topn 50000         # 只导出词频最高的 5 万个词
    python export_vec.py --min-freq 5         # 过滤出现次数少于 5 的词
    python export_vec.py --out my.vec         # 指定输出文件名

注意：
  1. word2vec 文本格式以空格分隔，词本身不能含空格或换行，
     这类词会被跳过（并在结束时报告数量）。
  2. 本项目的 token 是 dim+1 维，最后一维是词频，不参与语义计算，
     导出时会自动砍掉。
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    """加载训练产物。path 为不带扩展名的前缀。"""
    npz_p, json_p = path + ".npz", path + ".json"
    if not (os.path.exists(npz_p) and os.path.exists(json_p)):
        sys.exit(f"找不到 {npz_p} / {json_p}，请先运行：python tran.py")
    npz = np.load(npz_p)
    with open(json_p, encoding="utf-8") as f:
        meta = json.load(f)
    return {
        "words": meta["words"],
        "ka_vec": npz["ka_vec"], "ka_freq": npz["ka_freq"],
        "kb_vec": npz["kb_vec"], "kb_freq": npz["kb_freq"],
        "config": meta.get("config", {}),
    }


def export(model, table, out, topn=None, min_freq=1):
    """导出一张表到 out（word2vec 文本格式）。"""
    words = model["words"]
    vec = model[table + "_vec"]
    freq = model[table + "_freq"]

    # 按词频降序取前 topn 个
    order = np.argsort(-freq)
    if topn:
        order = order[:topn]

    kept, skipped = 0, 0
    with open(out, "w", encoding="utf-8") as f:
        # 先收集，写完才知道真实词数（第一行要写词数）
        lines = []
        for i in order:
            w = words[i]
            if freq[i] < min_freq:
                continue
            # 空格/换行/制表符会破坏 word2vec 的空格分隔格式，跳过
            if (not w) or any(c in w for c in " \t\r\n"):
                skipped += 1
                continue
            lines.append(w + " " + " ".join(f"{x:.6f}" for x in vec[i]))
            kept += 1

        dim = vec.shape[1]
        f.write(f"{kept} {dim}\n")
        f.write("\n".join(lines))
        if lines:
            f.write("\n")

    size_mb = os.path.getsize(out) / 1024 / 1024
    print(f"  {out}: {kept} 词 / {dim} 维 / {size_mb:.2f} MB")
    if skipped:
        print(f"    跳过 {skipped} 个含空格或换行的词（word2vec 格式不支持）")
    return kept


def main():
    ap = argparse.ArgumentParser(description="导出 word2vec 格式的嵌入表")
    ap.add_argument("--model", default=os.path.join(HERE, "embedding"),
                    help="模型前缀，默认 embedding（即 embedding.npz/.json）")
    ap.add_argument("--table", default="both", choices=["ka", "kb", "both"],
                    help="导出哪张表，默认两张都导")
    ap.add_argument("--topn", type=int, default=None,
                    help="只导出词频最高的 N 个词")
    ap.add_argument("--min-freq", type=int, default=1,
                    help="过滤出现次数少于 N 的词，默认 1（不过滤）")
    ap.add_argument("--out", default=None,
                    help="输出文件名；导出两张表时此项失效，自动用 ka.vec / kb.vec")
    args = ap.parse_args()

    model = load(args.model)
    tables = ["ka", "kb"] if args.table == "both" else [args.table]

    print(f"加载 {args.model}.npz：共 {len(model['words'])} 词")
    for t in tables:
        out = args.out if (args.out and len(tables) == 1) else os.path.join(
            HERE, t + ".vec")
        export(model, t, out, args.topn, args.min_freq)

    print("\n加载示例：")
    print("  from gensim.models import KeyedVectors")
    print(f"  kv = KeyedVectors.load_word2vec_format('{tables[0]}.vec', binary=False)")


if __name__ == "__main__":
    main()
