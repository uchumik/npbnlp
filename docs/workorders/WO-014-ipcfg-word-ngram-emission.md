# WO-014: iPCFG の pre-terminal 出力を単語 n-gram にする

Status: open

担当: impl-core(opus) / ブランチ: fix/WO-014
対象: `pa/src/ipcfg.cc`, `pa/include/ipcfg.h`, `pa/src/pa.cc`, `docs/ipcfg_model.md`, `tests/`
前提: dev の最新（`b62ff4b` 以降）から分岐

## 背景

iPCFG は PTB 1K・20 epoch・5 seed で Evalb F1 平均 40.5 と、右分岐ヒューリスティック
55.66 に 15 点負けている。クラスは縮退していない（`internal_active` 12〜19）ので
崩壊ではなく表現力の問題。直前の分割点事前（WO-013）は `q` が初期値に張り付いて
学習せず失敗した。

より土台に近い欠陥が実測で見つかった。**pre-terminal クラスが decode 時に退化している**
（seed1729 の 50 文・346 トークン vs gold 品詞）:

| 指標 | 値 |
|---|---|
| many-to-one 品詞純度 | **0.153** |
| decode 時の実効クラス数 | 7（学習時は `preterm_active=15`） |
| 最大クラス占有 | **86%**（内訳 `DT:36 NNP:31 NN:28 RB:27`） |

出力が `P(A)P(w_i|A)` の unigram なので、未知語・低頻度語ではどのクラスも基底測度
（文字 VPYP）に落ちて同値になり、argmax が `P(A)` 最大のクラスを選ぶだけになる。

## モデル（司令塔確定 — 変更禁止。矛盾発見時は停止して報告）

pre-terminal の出力を

$$
P(A)\,P(w_i \mid w_{i-1},\dots,w_{i-n+1},\,A)
$$

にする。各 `w_i` は一度だけ生成されるので二重計上は無い。条件部は**観測済みの前方単語**
なので、この因子はクラス列を固定すると木の形に依らない。

`ma/src/phsmm.cc` が同一構造を実装済み。**クラスは文脈木に混ぜず配列 `(*_word)[k]` のまま**、
各要素を `hpyp(n)` にする。文脈木のキーは先行単語 id のみ（`ma/include/phsmm.h:48-50`,
`ma/src/phsmm.cc:352-364` が add、`:392-402` が remove）。

置換パターン:

```
scoring: (*_word)[k]->lp(w, (*_word)[k]->h())  ->  ...->lp(w, (*_word)[k]->find(s, i))
add:     ...->add(w, ...->h())                 ->  ...->add(w, (*_word)[k]->make(s, i))
remove:  ...->remove(w, ...->h())              ->  ...->remove(w, (*_word)[k]->find(s, i))
```

`hpyp::find/make(sentence&, int)`（`lm/src/hpyp.cc:195-205, 239-245`）が既製。

### 安全性の根拠（確認済み。実装者は前提として使ってよい）

- 文脈キーは観測文 `s` からのみ引くので、木の走査順にも add 済みかにも依存しない。
  `_add`/`_remove` が同一キー列に到達することが構造的に保証される。
- `sentence::operator[](int i)` は `i<0` で 0(BOS) を返す（`io/src/sentence.cc:57-61`）。
  文頭は自動 BOS 埋め、`make` は位置によらず常にフル深度。
- `sentence` ctor が `dic->index(wd)` で実 id を採番（`io/src/sentence.cc:18`）。
  未知語も id ≥ 2 になるので「id==1 を文脈木のキーにしない」は自動的に充足。
- `_n == 1` では `find`/`make` のループが回らず `_h.get()`（=`h()`）を返す。
  **`--wngram 1` の現行完全一致は構造的に保証される。**
- `lp_root_base`（root 専用の手書き補間）は **iPCFG から呼ばれていない**ことを
  `grep -rn lp_root_base pa/` で確認済み（ヒット 0）。

## 危険箇所（3 つ。ここが本 WO の主眼）

### 危険 1（最重要）: scoring で `make` を呼ぶと OpenMP データ競合

`context::make`（`lm/src/context.cc:104-116`）は `_mutex` 下で `_child` に**書き込む**が、
`context::find`（`:95-101`）は `_child` を**ロック無しで読む**。scoring 5 箇所は
OpenMP 並列ブロック内（`pa/src/pa.cc:475-491`）で走るため、scoring から `make` を呼ぶと
`unordered_map` への同時読み書き = **未定義動作**。

**scoring は必ず `find`。`make` を書いてはいけない。**

### 危険 2: `find` の浅い返却によるカウント破壊

`find` は miss すると浅いノードで打ち切って返す（`hpyp.cc:199-202`）。`_remove` が
`_add` より浅いノードを引くと、`hpyp::remove`（`:711-719`）が root の `_stop` を負にし、
`_crp_remove` が**他文の客を外す**。`_bc_remove` はランダムな witness を選ぶ
（`:663-664`）ので復旧不能。

理屈上は起きないが、**snpylm の流儀（`tg/src/snpylm.cc:964-978`）に倣い、`_remove` では
`find` の歩数が `_n-1` に達したことを検査し、達しなければ throw する。**
`phsmm` は無防備な `find` を使っているが、これは真似しないこと。

### 危険 3: load 時の `_n` 上書きによる異種混在モデル

`hpyp::load` は保存された `_n` で**無条件に上書き**する（`hpyp.cc:127`）。v6 モデルを
`--wngram 2` で読むと、既存クラスは n=1、`_resize()` が作る新クラスは n=2 になり、
`_slice_preterm` が**異なる次数の LM のスコアを同一表で比較**する不整合が起きる。

**load 後に全クラスの `_n` が一致することを検査**し、CLI 指定と食い違うなら
ファイル側の値を採用して**警告を出す**（黙って無視しない）。採用した実効 `_n` を
`_wn` メンバに持ち、以後の `_resize()` はその値を使う。

## 実装

**`pa/src/ipcfg.cc`**

1. scoring 5 箇所を**同時に同一の式**に（1 つでも漏れると slice の table と
   `_calc_preterm` が食い違い assert 失敗）:
   `_traceback`(:872), `_traceback_logprob`(:922), `_calc_preterm`(:966),
   `_slice_preterm`(:1256), `_slice_preterm_cond`(:1278)
2. MH 逐次 target `_init_logprob_and_add`(:353) の lp と直後の add(:356) を対で
3. カウント: `_init_node`(:286), `_add`(:575-576), `_remove`(:622-623)
4. ctor 2 箇所(:26, :35) と `_load`(:144) の `new hpyp(1)` → `hpyp(_wn)`
5. **`_resize()`(:1339) の `resize(_k+1, shared_ptr(new hpyp(1)))` を `push_back` ループに
   直す。** CLAUDE.md が禁じる同一オブジェクト共有パターンで、この行はどのみち書き換わる
6. `_wn` を serialize（下記）

**`pa/include/ipcfg.h`** — `_wn` メンバと `word_ngram(int)` setter

**`pa/src/pa.cc`** — `--wngram=int`（既定 2、`1` で現行完全一致）、usage 追記

**`docs/ipcfg_model.md`** — 記号表・生成過程・同時確率の pre-terminal を n-gram に。
**原案（`/mnt/d/Work/Notes/unsupervised_learnings_for_nlp.md` の infinite PCFG 節、
142 行目付近）は `P(w|A)P(A)` の unigram なので、これは意図的な逸脱である。
逸脱の理由（クラスの decode 時退化、純度 0.153）と影響を明記すること。**
併せて「この因子はクラス列を固定すると木の形に依らない」ことも書く。

**`tests/`** — `--wngram 2` での add/remove ラウンドトリップ。
既存 `tests/CMakeLists.txt` の未コミット差分は触らず、必要行だけ足すこと。

### serialize

tail block を **version 7** にし `_wn` を記録。v6 以前は `_wn=1` として**正常に load する**
（n=1 は n-gram の特殊ケースで拒否理由が無い）。実行中の 10K ジョブが v6 を書くので
この後方互換は必須。危険 3 の一致検査を必ず併せて入れる。

## 受け入れ基準

1. `make pa` 警告 0、`ctest` 全通過、`git diff --check` クリーン。
2. **grep ゲート**: scoring 5 箇所に `make(` が 1 つも無いこと（危険 1）。
3. **後方互換**: `--wngram 1` で `[ipcfg-diag]`/`[span]`/`[split]` が変更前バイナリと
   **バイト単位一致**。
4. **ラウンドトリップ**: `--wngram 2` で全文 add → 全文 remove 後に `ipcfg::valid()` が真、
   かつ**全 `_word[k]->empty()` が真**（深さ≥1 の phantom 着席を捕まえるのは後者）。
5. **スライス経路生存**: `--wngram 2 --mh` を assert 有効ビルド（`-UNDEBUG`）で
   1K・3 epoch 完走、assert 発火 0。
6. v6 モデルが `_wn=1` として load でき `--parse` が変更前と同一出力。
   v6 モデルに `--wngram 2` を指定すると警告が出ること（危険 3 の検査）。
7. OpenMP 有効/無効の両方でビルド。

## 追加で出してほしい診断

本番実験で失敗モードを検出するために、以下を **epoch ごとに cerr へ出す**こと
（既存の `[ipcfg-diag]` 行に足すか、新しい `[preterm]` 行を作る）:

- `_slice_terminal_labels / _slice_terminal_cells`（`ipcfg.h:74-75` に既存）。
  **1.0 に近づくと「単語を見ればクラスが一意」= 文法に決める余地が無い**という
  失敗モードのサイン。
- 各クラスの `_bc` サイズ（init 後に 1 回でよい）。n-gram では root に到達する客が減り
  Poisson 補正の推定母数が痩せるため。

MH 受理率は既に `pa.cc:459-465` で出ているのでそのままでよい。

## 注意

- **実行中のジョブを kill しないこと**（10K bottom-up PID 96179 と UD 多言語 10 本）。
  `experiments/` 配下に一切書き込まないこと。動作確認は `/tmp` に出す。
- 評価 parse は `NPBNLP_NOSLICE=1`（スライス parse は同一モデルでも ±4 点振れる）。
- 本番実験（`--wngram` 1/2/3 の比較）は**実行しなくてよい**。コマンドだけ用意すること。
