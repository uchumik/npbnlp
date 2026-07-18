# WO-006: Switching NPYLM — 教師なし NER の新モデル実装(DRAFT・司令塔起草、ユーザー承認待ち)

Status: closed (merged into dev)

- 担当: impl-core (opus)。理論は `.claude/skills/npbnlp-math/SKILL.md` §4 が正典(完全定式化・変数定義・生成過程・推論)。`/mnt/d/Work/Notes/unsupervised_learnings_for_nlp.md` も参照
- 位置づけ: **nphsmm とは独立した新手法**(既存 ne 系のコード・モデルには一切触れない)。P6(O クラス不在)と E-7(un レジームの under-merge)への構造的回答
- 前提: WO-003/004/005 マージ後(slice 条件付けの確立した流儀を最初から採用する)

## 設計サマリ(math スキル §4 の実装写像)

生成の骨子: 文は背景 LM(G^bg)がテンプレートトークン列 v_1..v_{M+1}(通常語 or NE_k 記号)として生成し、v_j=NE_k のときだけクラス別表層モデル H_k が文字列 x_j を放出する。**タグ遷移モデルは持たない** — NE_k 記号が G^bg の n-gram 語彙に入ることで タグ⇄語 の依存を背景 LM が直接学習する。

| 部品 | 実装 |
|---|---|
| クラス `snpylm` | 新規 tg/include/snpylm.h + tg/src/snpylm.cc(命名は既存流儀の小文字短語) |
| G^bg | word HPYP(n=2)+ 基底 G0 = (1-π)·綴りモデル(letter VPYP+Poisson) + π·Σ ρ_k δ_{NE_k}。ma/npylm の word/letter 構成を流用 |
| NE_k 記号 | wid 辞書に合成表層(例 "\x01NE\x01k")で登録し実 id を確保(id 0/1 予約と衝突しない)。k↔id の対応表を snpylm が保持・serialize |
| H_k | チャンク単位 PYP(hpyp)→ 基底 = クラス別 letter VPYP + Poisson(λ_k) + ψ(文字種変化数事前)。nphsmm の chunk→letter 構成から word 層を抜いた形 |
| π | Beta(1,γ) 共役。G0 での NE/通常語テーブル数からサンプル |
| ρ (GEM) | クラス CRP カウントから棒折り。無限クラスは nphsmm の _resize/_shrink パターン |
| λ_k | Gamma-Poisson 共役 |
| 格子 | clattice2 を流用し **z=0 は長さ 1 固定**の制約を追加(z≥1 は 1..L) |
| 推論 | blocked Gibbs + **現在状態条件付き slice(WO-003 流儀を最初から)** + semi-Markov FFBS(α[t][ℓ][k]、E_0=1, E_k=P(x|H_k)、遷移は G^bg の n-gram) |
| CLI | 新バイナリ tg/src/sne.cc(--train/--parse/--model/--dic/--tokenizer/--wdic、n/k/epoch/threads/prefix。教師なし専用で開始) |

## 実装フェーズ(WO 分割)

### Phase 1(本 WO-006): 生成モデルと台帳 — skeleton + add/remove + roundtrip
- snpylm クラス: メンバ(G^bg word hpyp / letter vpyp、H_k 群、NE_k id 表、π/ρ/λ カウント)、コンストラクタ、set 系
- nsentence(チャンク列: z=0 は 1 語チャンク、z≥1 は NE チャンク)に対する add / remove / init:
  - add: テンプレート列を構成(通常語→word id、NE→NE_k id)し G^bg に着席; z≥1 チャンクは H_{z} にも表層着席; π/ρ/λ カウント更新
  - remove は完全対称(CRP 不変量)
- estimate: 各 LM の d/θ、λ_k(Gamma)、π(Beta)、GEM 更新
- serialize(save/load、末尾追記規約)
- **受け入れ: nphsmm と同型の add/remove ラウンドトリップテスト(tests/test_snpylm_roundtrip.cc)green**

### Phase 2(WO-007): 推論 — slice + FFBS + 学習ループ + CLI
- _slice(現在状態条件付き、WO-003 と同じ pathmap 方式。math §2 準拠)
- forward(α[t][ℓ][k])+ 後ろ向きサンプル(math §4.5 の式)。効率化(minfer 相当)は初版では作らない(--original 相当の単純経路のみ。プロファイル後に判断)
- sne.cc: mcmc ループ(remove→sample(cur)→add)、snapshot、進捗
- **受け入れ: tiny 学習完走(NaN/例外なし)、slice 生存 assert ゼロ、logP トレンド安定、全チャンクが O に退化していない(NE クラス使用率を MODEL_STATS 相当で出力)**

### Phase 3(WO-008): 評価と退化対策
- 定型評価(un レジーム、precision 併記): un_bpdur / un_bpwclass と比較。特に pred_multi(under-merge が解けるか)と purity
- 退化時の対策ノブ(math §4.7): γ、ψ 強化、焼きなまし τ、(必要なら)少数シード
- HANDOFF / memory 更新

## リスク・設計判断メモ
- **全 O 退化が最大リスク**(§4.7)。推進力は「未知語は背景では綴りコストが高いが、NE クラス特化文字モデルなら安い」という差分。初版から NE クラス使用率の診断出力を仕込み、退化を早期検知する
- E_k の χャンク表層は文字列 PYP なので、**E-7 の under-merge 問題とは力学が違う**(背景 n-gram が NE_k 記号の出現位置を文脈で学習する = マージ圧力が語彙・文脈両方から来る)
- n≥2 必須(テンプレート文脈がクラスを意味で割る唯一の力 — §4.7)
- 既存資産の流用は「コードのコピー」ではなく「同じクラスの再利用」を優先(hpyp/vpyp/clattice2 はそのまま使えるはず。使えない箇所が出たら司令塔に報告)

## 検証(全フェーズ共通)
ctest(roundtrip)、NPBNLP_* 診断 env の移植(NOSLICE/DEBUG_LK/SLICE_CHECK 相当)、
100 文 × 50ep assert 学習、教師なし 10k 学習 → ne_evaluate.py。
