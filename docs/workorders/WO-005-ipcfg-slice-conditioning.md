# WO-005: ipcfg スライス変数の現在状態への条件付け + root μ の括弧バグ精査

> **改訂 (2026-07-12, 司令塔)**: 理論仕様は著者資料
> `/mnt/d/Work/Notes/unsupervised_learnings_for_nlp.md` の「Grammar Induction / infinite PCFG」
> 節が正典(実装前に必読)。要点:
> - μ は**スパンごと** $\mu_{i,j}$(`c.mu[i][j]` と整合)。現行の「同一長スパンで ν を
>   pivot 共有」は理論から逸脱しており、**cur あり(学習)経路はスパンごと独立 μ に是正**する。
> - 条件付き分布(Blunsom & Cohn 2010 型):
>   ルール $A\to BC \in T$(現在の木)のスパンでは
>   $P(\mu_{i,j}|T)=\mathbb{I}(\mu\le P(BC|A))\,\beta(\mu;a,b)/P(BC|A)$
>   → 実装は `mu = log(be(_a,_b)) + log P(r_cur)`。$\notin T$ では $\beta(\mu;a,b)$。
> - $P(BC|A)\propto P(A|B,C)P(B,C)$ はコードの `lp(m,s)+lp_l+lp_r` に対応(bottom-up Bayes)。
> - **off-path の逸脱を許可**: 資料の無スケール $\beta(\mu;a,b)$ は現行 $a,b$ のままでは
>   閾値が高すぎ探索空間を破壊するため、off-path は現行の「スパンごと fresh draw
>   `log(be)+table[id]`」を維持してよい(μ は remove 後の固定量にのみ依存すれば正当性に
>   影響しない設計自由度)。この逸脱をコードコメントに明記すること。
> - **cur==nullptr(parse)は現行挙動(pivot 共有 ν 含む)を乱数消費まで完全維持**
>   (bit 再現)。cur あり経路のみスパンごと独立 μ に切り替える。
> 下記仕様の「共有 ν の min 合成」案はこの改訂で**破棄**(理論準拠のスパンごと独立 μ を採る)。

- 担当: impl-core (opus)。npbnlp-math スキル §2 必読。WO-003/004 と同系だが**スライス構造が異なる**(共有 ν)ため設計精読必須
- 対象: pa/include/ipcfg.h, pa/src/ipcfg.cc, pa/src/pa.cc
- 背景: 司令塔調査(2026-07-12)で P5 同型を確認。
  - `_slice_preterm`(pa/src/ipcfg.cc:~535 系の active 版): セルごと `ln_draw`+`log(be)+table[id]` の fresh draw
  - `_draw(l,p,p+m)`(:443-480): pivot スパン p で fresh draw した μ を `c.mu[i][j]` に置き、同一長 m の**他の全スパンに `_slice_nonterm(l,i,i+m,nu)` として共有**(pivot はランダムウォーク)
  - `_slice_root`(:~xxx): fresh draw。**加えて `mu = log(be(_a,_b)+table[id])` と括弧が他と異なる**(他は `log(be)+table[id]`)。table が log 領域なら負値の log = NaN の疑い → 精査項目
- pa.cc の学習ループ(:304-337)は `g.remove(corpus[rd])` → `tree tr = g.sample(f, rd)` → `corpus[rd]=tr` → `g.add(...)` — **sample 実行中 corpus[rd] は現在の木を保持** → 条件付けに利用可能。

## 仕様

### シグネチャ連鎖(WO-003/004 と同型、後方互換)
```cpp
virtual tree sample(io& f, int i);                 // 既存: 委譲
virtual tree sample(io& f, int i, tree *cur);      // 追加
void _slice(cyk& l, tree *cur);                    // 引数追加(parse 経路は nullptr)
```
`_minfer` 相当の効率化経路が ipcfg にもある場合(要確認)は同様に cur を伝搬。
pa.cc: `tree tr = g.sample(f, rd[j+t], &corpus[rd[j+t]]);`(全分岐)。

### 条件付けアルゴリズム(共有 ν 構造を保ったまま生存保証)
cur != nullptr のとき、木を走査して **span マップ** span(i,j) → {現在のルール構成(左子ラベル l_cur, 右子ラベル r_cur, 分割点 k_cur, 親ラベル m_cur)} を作る(preterm は (i,i)→ 前終端ラベル、root は (0,n-1)→ ルートラベル)。

1. **preterm**: on-path セル (i,i) は `mu_i = log(be) + table[idx(現在の前終端)]`(ln_draw なし)。off-path は現行どおり。
2. **nonterm(共有 ν)**: 長さ m の ν を
   `nu_m = min( pivot での現行 draw 値, min_{on-path span (i,i+m)} [ log(be_i) + score_cur(i,i+m) ] )`
   とする(be_i はスパンごと独立に引く)。score_cur は現行 `_draw`/`_slice_nonterm` が
   table に積むのと同じ量(`lp(m|s)+lp_l+lp_r`)を現在構成 (l_cur,r_cur,k_cur,m_cur) で評価。
   log(be)<=0 より全 on-path スパンの現在構成が必ず許可集合に残る(正当性条件)。
   pivot 自体が on-path の場合は pivot 側を条件付け draw(`log(be)+score_cur`)にしてよい。
3. **root**: on-path(cur あり)なら `mu = log(be) + table[idx(現在のルート構成)]`。
4. cur==nullptr は全経路現行どおり(乱数消費まで同一 = parse の bit 再現)。
5. `#ifndef NDEBUG` assert: span マップ上の全構成が許可集合(c.k[i][j])に生存。

### root μ 括弧の精査(修正はここまでのスコープ)
`_slice_root` の table が log 領域か確認し、log 領域なら `log(be(_a,_b))+table[id]` に修正
(= 他箇所と同型)。**NaN になっていた場合、修正で root の枝刈りが「効き始める」= 挙動変化**
なので、変更前後の parse 出力比較と学習完走確認を必ず取り、報告に明記。log 領域でなければ
現状維持で理由を報告。

## 受け入れ基準
1. ビルド(pa ターゲット)
2. cur=nullptr の bit 再現: 既存 ipcfg モデルがあれば parse 出力 diff 一致(無ければ
   「新規学習モデルでの self-parse 再現」で代替し、その旨報告)。root 括弧修正が入った
   場合はこの限りでない(挙動変化を分離報告)
3. assert 有効で 100 文 × 50 epoch 完走、assertion failure ゼロ
4. 適用前後 corpus logP トレンド比較(monitor 採取・司令塔判定)
5. diff が本仕様の範囲に収まる
