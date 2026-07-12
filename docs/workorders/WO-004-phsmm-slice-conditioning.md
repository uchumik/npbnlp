# WO-004: phsmm スライス変数の現在状態への条件付け(WO-003 の phsmm 版)

- 担当: impl-core (opus)。npbnlp-math スキル §2 を必読
- 対象: ma/include/phsmm.h, ma/src/phsmm.cc, ma/src/ma.cc
- 背景: WO-003(nphsmm)と同一の P5 系問題を司令塔が phsmm でも確認(2026-07-12)。
  `phsmm::_slice`(ma/src/phsmm.cc:712)の μ 決定が
  `int id = rd::ln_draw(table); double mu = log(be(_a,_b))+table[id];` と
  現在の割当に条件付けられておらず、現在状態がスライスから脱落しうる
  (詳細釣り合い・既約性の破れ)。呼び出しは parse(:448)/sample(:556)/
  `_minfer`(:856、`--original` でない限りの既定経路)の 3 箇所。
  なお長さ側スライス(nu)はコメントアウト済みの死にコードであり対象外。

## 仕様(WO-003 と同型)

### シグネチャ連鎖(後方互換を保つ)
phsmm.h / phsmm.cc:
```cpp
virtual sentence sample(io& f, int i);                  // 既存: sample(f,i,nullptr) へ委譲
virtual sentence sample(io& f, int i, sentence *cur);   // 追加
sentence _minfer(io& f, int i, bool best);              // 既存: cur=nullptr で委譲
sentence _minfer(io& f, int i, bool best, sentence *cur); // 追加
void _slice(lattice& l, sentence *cur);                 // 引数追加
```
(io/lattice/sentence の実型名は現物に合わせる。parse() は cur=nullptr。)

### ma.cc の呼び出し変更
mcmc 系ループ内の全 `lm.sample(f, rd[j+t])` を
`lm.sample(f, rd[j+t], &corpus[rd[j+t]])` に(OpenMP 有無・semi/un の全分岐。
grep `lm.sample(` で全箇所を列挙して漏れなく)。remove() は corpus の中身を
消さないので remove 後も corpus[·] が現在の割当(単語分割 len + pos)を保持する。

### _slice(lattice& l, sentence *cur) のアルゴリズム
1. cur != nullptr のとき経路マップ構築: cur の単語を先頭から累積し、
   **現行 _slice のセル添字系(位置 t × 単語長 len)と同じ座標**で
   pathmap[t] = {len, pos_cur} を作る(添字の起点・終端の扱いは現行ループを
   読んで正確に合わせる)。pos_cur が [1,_k] 外なら登録しない(off-path)。
2. 各セルの table 計算は現行どおり。
3. μ: on-path セル → `mu = log(be(_a,_b)) + table[pos_cur-1]`(ln_draw を呼ばない)。
   off-path → 現行どおり ln_draw。**cur==nullptr は全セル off-path = 現行と
   乱数消費まで完全同一**(parse の bit 再現性)。
4. 許可集合 push は現行どおり(log(be)<=0 より on-path で pos_cur は必ず生存)。
5. NPBNLP_NOSLICE / NPBNLP_SLICE_CHECK / NPBNLP_NAIVE_SLICE の経路は不変。
   naive 経路(メモ化前 A/B 用)にも同じ μ 規則が及ぶ場合は同様に適用し、
   及ばない構造なら理由を報告に明記。
6. `#ifndef NDEBUG` で pathmap 全セルの pos_cur ∈ 許可集合 assert。
7. 「セルごと補助変数の近似ビーム、時刻ごと μ_t 完全準拠は将来課題」コメントを残す。

## 受け入れ基準
1. ビルド(OpenMP 有無両方)
2. cur=nullptr の bit 再現: 既存 ma モデル(bccwj_core_ot_semi.20260703)で
   NPBNLP_NOSLICE+DEBUG_LK の parse lk が orig/eff 一致、かつ本 WO 適用前後で不変
   (DEBUG_LK 相当が phsmm に無い場合は parse 出力ファイルの diff 一致で代替)
3. assert 有効で 100 文 × 50 epoch、assertion failure ゼロ
4. 適用前後の corpus log P トレース比較で適用後が高水準で安定(monitor 採取・司令塔判定)
5. diff が本仕様の範囲(phsmm.h/phsmm.cc/ma.cc)に収まる
