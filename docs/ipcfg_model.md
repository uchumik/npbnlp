# データ駆動 iPCFG

## 記号

| 記号 | 意味 |
| --- | --- |
| `w=(w_1,...,w_n)` | 長さ `n` の観測単語列。`w_i` は位置 `i` の単語。 |
| `T` | `w` を覆う潜在二分構文木。`I` は内部ノード集合、`P` は pre-terminal 集合。 |
| `v` | 木のノード。`A_v` はその親ラベル、`B_v,C_v` は左右の子ラベル。root の `A` は固定値 `0`。 |
| `(i,j)` | ノードが覆う閉区間。幅は `j-i+1`、split `b` は `i<=b<j`。 |
| `P(B)` | 内部ノードの左子ラベル `B` の HPYP 予測分布（root 文脈）。 |
| `P(C|B)` | 左子を見た右子ラベル `C` の HPYP 予測分布。文脈 `B` に対応。 |
| `P(A|C,B)` | 子対をまとめ上げる親ラベル `A` の HPYP 予測分布。文脈 `C,B` に対応。 |
| `P(A)` | pre-terminal ラベルの HPYP 予測分布（空文脈）。 |
| `W_A(w)` | クラス `A` の単語 HPYP による語 `w` の予測分布。基底は全クラス共有 VPYP。 |
| `P_span(i,j)` | 非 root 内部 span の幾何減衰。`p` を stop 確率として `p(1-p)^(j-i-1)`。root と terminal には掛けない。 |
| `q` | 分割点事前の幾何パラメータ。`q->1` は `L=1`（左子が1語）に集中 = **右分岐**、`q->0` は分割点一様。`Beta(a_q,b_q)` 事前から griddy Gibbs で毎 epoch サンプルする（打ち切り項が `w` に依存するので共役ではない）。 |
| `P_split(b|i,j)` | 分割点 `b` の打ち切り幾何事前。左子幅 `L=b-i+1`、幅 `w=j-i+1` として `q(1-q)^(L-1)/(1-(1-q)^(w-1))`。台 `[1,w-1]` 上で厳密に正規化される。**root を含む全内部ノード**に掛ける（`P_span` と適用範囲が異なる）。`w=2` では恒等的に 1。 |
| `alpha[i,j](A)` | leave-one-out counts の下での CYK inside 値（対数領域ではその対数）。 |
| `Z` | root inside `alpha[0,n-1](0)`。slice 格子に条件付けた proposal の正規化定数。 |
| `s(T)` | traceback で選んだ各局所規則と terminal の対数確率和（proposal の raw score）。 |
| `U` | 現在木に条件付けて引いた全セルの slice 変数と、その許可格子。 |
| `M_-s` | 文 `s` の木を remove した後のモデル counts。 |
| `p_seq(T|M_-s)` | root から順に生成・add する順序で評価した collapsed HPYP の木の確率。 |
| `q(T|U,M_-s)` | slice 格子 `U` 上の CYK+traceback proposal 確率。 |

### `q` の事前と初期値の区別

`q` に対する **事前知識の注入口は `Beta(a_q,b_q)` だけ**である（CLI では
`--split_alpha` / `--split_beta`）。向きの規約:

| 設定 | 意味 |
| --- | --- |
| `a_q > b_q` | `q` を大きい側へ引く = **右分岐**寄り。英語のような右分岐言語の知識 |
| `b_q > a_q` | `q` を小さい側へ引く = **左分岐**寄り。日本語のような左分岐言語の知識 |
| `a_q = b_q = 1`（既定） | 無情報。**分割の偏りを純粋にデータから獲得する** |

`--split_q_init` は **griddy Gibbs サンプラーの初期値であって事前知識ではない**。
正しく mixing していれば結果は初期値に依存しないはずで、初期値感度テスト
（`0.1 / 0.5 / 0.9` から同一 `q` へ収束するか）はこれを利用した mixing 診断である。
初期値付近に留まるなら学習できておらず、事前を埋め込んだだけと判定する。

## 生成過程

観測文を `w_1,...,w_n`、二分木を `T`、各ノードの潜在非終端を `z` とする。root は固定ラベル `0`、その他はデータから生成されるラベルである。規則表は与えない。

内部ノードの親を `A`、子を `B,C` として、文法 HPYP の三つの条件付き文脈から

$$
B\sim P(\cdot),\qquad C\sim P(\cdot\mid B),\qquad A\sim P(\cdot\mid C,B)
$$

を bottom-up に生成する。すなわち規則因子は `P(B)P(C|B)P(A|C,B)` であり、
観測単語列から出発して pre-terminal で単語を抽象化し、それらをまとめ上げる
向きに対応する（原案 `unsupervised_learnings_for_nlp.md` の infinite PCFG 節）。
分割点 `b` は `P_split(b|i,j)` から生成し、非root内部ノードには Beta--geometric
span prior を掛ける。pre-terminal は

$$
P(A\to w)=P(A)W_A(w)
$$

であり、各 `W_A` の基底は共有文字 VPYP である。

`_nonterm` の root 文脈には pre-terminal ラベル `A` と内部ノードの左子 `B` が
直接着席する。これはラベルに対する単一のカテゴリ事前を全ノードで共有するという
原案の設計であり、両者を別レストランへ分離しないこと。

top-down 因子 `G_L(B|A)G_R(C|A,B)` は `docs/ipcfg_rule_factor_reformulation.md`
で提案されたが WO-011 で不採用となり、実装から撤去された。

## 同時確率

$$
\log p(T,w)=\sum_{v\in I}[\log P_{split}(b_v|i_v,j_v)+\log P(B_v)+\log P(C_v|B_v)+\log P(A_v|C_v,B_v)+\log P_{span}(v)] + \sum_{v\in P}[\log P(A_v)+\log W_{A_v}(w_v)].
$$

`--split` を指定しないときは `P_split` の代わりに一様分布 `-\log(|v|-1)` を用いる
（WO-012 以前の挙動）。**この一様因子は逐次 target `p_seq` にしか入っておらず、
CYK inside と traceback は `b` について重み 1 で総和していた。すなわち分割は同時確率上
正規化されていなかった。** WO-012 の `P_split` は逐次 target・CYK・traceback・slice・
init のすべてに同一に入るため、この非正規性が解消される。`P_split` と一様因子は
どちらか一方のみを掛ける（二重に掛けない）。

collapsed HPYP では各項を、それ以前の生成項を add した後の予測確率で評価する。

## CYK と traceback

leave-one-out counts を固定した proposal の inside は

$$
\alpha_{i,j}(A)=\sum_b\sum_{B,C}P_{split}(b|i,j)P(B)P(C|B)P(A|C,B)P_{span}(i,j)\alpha_{i,b}(B)\alpha_{b+1,j}(C),
$$

terminal は `alpha[i,i](A)=P(A)W_A(w_i)` である。root inside `Z=alpha[0,n-1](0)` を正規化定数とし、traceback は各候補を inside 寄与に比例して選ぶ。木の再帰 raw score を `s(T)` とすれば `log q(T)=s(T)-log Z` である。

## slice と MH

slice の各セルの score は `P_split` を含む（したがって分割点に依存する）。
`_slice_nonterm_cond` / `_slice_root_cond` は現在木の分割点 `b_cur` を使って
`score_cur` を作るので、現在の割当は必ず許可集合に残る。

各セルでは現在木の局所 score `r_cur` に条件付け、`mu=log u+r_cur` (`u` は Beta 分布) として、score が `mu` 以上の状態だけを残す。current tree は必ず許容される。同じ slice 格子で旧木・提案木の `q` を計算する。

逐次 HPYP add で得る target `p_seq` に対する MH 受理率は

$$
\min\{1,\frac{p_{seq}(T'|M_{-s})q(T|U,M_{-s})}{p_{seq}(T|M_{-s})q(T'|U,M_{-s})}\}.
$$

棄却時は提案木を remove して旧木を add し、共有 VPYP への customer を add/remove 対称に保つ。
