# Charon —— 无梯度下降的中文词嵌入

一个纯 NumPy,jieba 实现的可解释的无梯度词向量实验项目：**没有神经网络、没有反向传播、没有损失函数**。
词向量靠「窗口上下文的加权平均 + 负采样推开」逐次更新得到，全程 O(n)。

- `ka`：窗口上下文嵌入表
- `kb`：在 ka 之上叠加「上文注意力累加」的嵌入表（用前缀和把 O(n²) 降到 O(n)）
- `att`：每个词与其上下文的偏离度，衡量「这个词带来了多少新信息」

token 统一是 `dim+1` 维，最后一维是训练中出现频次，不参与语义计算。

## 快速开始

先装依赖（只需一次）：

```bash
pip install -r requirements.txt
```

### 如果上面的看不懂，照这个来

**① 训练**

在 `.py` 文件所在的同一个文件夹里，新建一个 `tran.txt`，把中文语料粘进去。
纯文本就行，UTF-8 或 GBK 都可以。

> ⚠️ **不要提前分词！** 直接粘原始连续文本（`我爱北京天安门`），
> 不要写成空格分隔的（`我 爱 北京 天安门`）。
> 代码内部会用 jieba 重新分词；如果语料自带空格，jieba 会把每个空格当成
> 一个独立 token 塞进词表，白白稀释真实词之间的关系。
> 标点不用特意删，但逗号、顿号等也会成为词表中的一员（无害，只是占位）。

然后运行 `tran.py`，看进度条，泡个茶等着就行。

**② 推理**

运行 `infer.py`，跟着菜单提示走，傻瓜式操作：查最近邻、算相似度、续写、生成都在里面。

**③ 想要通用格式的嵌入表**

运行 `export_vec.py`，导出 word2vec 文本格式，gensim / fastText / Annoy 都能直接读。
详见下方「导出通用嵌入表」。

### 命令行一览

```bash
# 训练
python tran.py                  # 默认最优配置
python tran.py --epochs 8 --dim 128

# 推理
python infer.py                 # 交互菜单
python infer.py --nearest 猫 -k 10
python infer.py --generate "小猫在公园" --beam 4

# 横评：检索式 vs 生成式（在训练语料没见过的前缀上比较）
python eval_gen.py
python eval_gen.py --real       # 用真实 tran.txt 做留出法

# 导出 word2vec 格式
python export_vec.py
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `Lpron_core.py` | 核心算法库：分词切句、窗口加权、负采样、ka/kb 训练、存取、推理器 |
| `tran.py` | 训练脚本，读 `tran.txt`，产出 `embedding.npz` / `embedding.json` / `params.txt` |
| `infer.py` | 推理脚本，命令行 + 交互菜单 |
| `predict.py` | 三个预测器：`KbAttPredictor`（无状态向量）、`RetrievalPredictor`（语料检索续写）、`KbGenerator`（有状态递推生成） |
| `eval_gen.py` | 续写方式横评：随机基线 / 检索 / 旧 greedy / 生成（含消融组） |
| `export_vec.py` | 把 `embedding.*` 导出成通用 word2vec 文本格式（`.vec`） |

## 导出通用嵌入表

`embedding.npz` 是本项目自己的格式，别家工具读不了。要拿到能被 gensim、fastText、
Annoy 等直接加载的通用嵌入表，运行：

```bash
python export_vec.py                  # 导出 ka.vec 和 kb.vec
python export_vec.py --table ka       # 只要 ka
python export_vec.py --topn 50000     # 只导词频最高的 5 万词
python export_vec.py --min-freq 5     # 过滤出现少于 5 次的词
```

导出的是标准 word2vec 文本格式（首行 `词数 维度`，之后每行 `词 向量...`），可直接：

```python
from gensim.models import KeyedVectors
kv = KeyedVectors.load_word2vec_format("ka.vec", binary=False)
print(kv.most_similar("小猫", topn=10))
```

导出时会自动砍掉 token 末尾那维词频（它不参与语义计算），并跳过含空格或换行的词
（word2vec 格式靠空格分隔，这类词会破坏格式，脚本会报告跳过数量）。

## 原始版本（`origin/`）

`origin/Charon.py` 是整套算法的**最初原型**，比 `Lpron_core.py` 更早，保留下来是为了记录演化过程。
它证明了上面「关键设计」里第一条坑的真实存在：原型用的是 `np.random.rand`（均值 0.5），
现版本改成 `np.random.randn`，这个改动直接把 KNN@3 从 0.04 拉到 0.60。

**该文件是未完成的原型，不可直接运行**，已知未修问题：

- `windows_context_suport_token` 定义 10 个参数，但两处调用只传 9 个（缺 `ka`），会 `TypeError`
- `get_ka` / `get_kb` 仅在词表超过 `iteration_wordbase` 且迭代满指定轮数时返回，小语料落空返回 `None`
- 文件末尾有模块级 `open("nn.txt")`，`import` 即触发；语料文件名与主版本的 `tran.txt` 不同，且未收录

不要 `import` 它，也不要拿它跑实验。要看能用的版本请用 `Lpron_core.py`。
`origin/Charon.txt` 是作者对原型诞生过程的说明。

## 几个关键设计（都是踩坑后定的）

- **初始化必须用零均值分布**（`randn` 而非 `rand`）。所有 token 都是若干 token 的正系数加和，
  `rand` 的均值 0.5 会让向量全体指向「全 1」方向，实测两两余弦 0.8994（几乎共线）。
- **必须切句**。不切句时窗口跨句，实测 KNN@3 从 0.60 掉到 0.04。
- **必须有负采样**。否则所有 token 都朝「所有上下文的合力」走，最终坍缩成一点（两两余弦 1.000）。
- **负例项要和正样本项同步除以 (w+1)**，否则高频词被负例推飞（KNN@3 0.04 vs 0.60）。
- **att 必须在更新前算**。更新后再算，token 已被拉向该上下文，残差恒为 0。
- **ka 与 kb 必须用同一个窗口加权函数**，否则 kb 拿到的上下文幅度与 ka 不一致，`base` 阈值失准。

## 已知问题

1. **没有 OOV 处理**：没有子词/字符级向量，词表外的新词只能靠字面线索兜底。
2. **生成会重复词组**：`no_repeat` 只禁止最近 n 个位置的词，`A B A B` 这类间隔重复挡不住，
   需要 n-gram 级别的重复抑制。
3. **代码注释里引用的 `README_DROP_LOWEST.md` 未包含在本目录**（那份实验记录未随代码一起整理）。

## 引用

本项目采用 MIT 许可，使用、修改、商用均无需事先取得许可，**也不强制要求署名**。

如果你在论文、报告或项目里用到了这份代码，欢迎引用 —— 不是义务，但对作者是实打实的鼓励。
点击仓库页面右侧的 **"Cite this repository"** 可自动获取 BibTeX / APA 格式，
内容由 [`CITATION.cff`](CITATION.cff) 生成。

```bibtex
@misc{pron2026,
  author = {NyaaaanFox},
  title  = {Charon: 无梯度下降的中文词嵌入},
  year   = {2026},
  url    = {https://github.com/NyaaaanFox/Charon}
}
```

## 许可

MIT，见 [LICENSE](LICENSE)。

注意：MIT 只覆盖**代码**。训练产物（`embedding.*`）的许可取决于你所用的 `tran.txt` 语料，
本仓库默认不收录语料，请自行确认你有权公开它。
