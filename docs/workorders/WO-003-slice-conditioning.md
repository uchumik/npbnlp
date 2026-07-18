# WO-003: スライス変数の現在状態への条件付け

Status: closed (merged into dev)

- 担当: impl-core (opus)。数学的正当性は npbnlp-math スキル §2 を必読
- 対象: tg/include/nphsmm.h, tg/src/nphsmm.cc, tg/src/ne.cc
- 前提: WO-001, WO-002 マージ後に着手
- 背景: diagnosis P5。現行 _slice() は remove 後にユニグラム事後から id を引き直して
  μ を決めており、現在の割当がスライスから脱落しうる（詳細釣り合い・既約性の破れ）。

## 仕様

### シグネチャ変更（後方互換を保つ）
nphsmm.h:
```cpp
virtual nsentence sample(nio& f, int i);                    // 既存: 委譲に変更
virtual nsentence sample(nio& f, int i, nsentence *cur);    // 追加
...
void _slice(clattice& l, nsentence *cur);                   // 引数追加
```
既存 `sample(f, i)` は `sample(f, i, nullptr)` へ委譲。parse() は `_slice(l, nullptr)`。

### ne.cc の呼び出し変更
mcmc() 内（OpenMP/非 OpenMP の両分岐）:
```cpp
nsentence s = lm.sample(f, rd[j+t], &corpus[rd[j+t]]);
```
remove() は corpus の中身を消さないので、remove 後も corpus[·] が「現在の割当」を保持
している点を利用する。

### _slice(clattice& l, nsentence *cur) のアルゴリズム
1. cur != nullptr のとき、現在の割当の経路マップを作る:
   セグメントを先頭から累積し、終了位置 e（0 起点の単語 index）→ (len, k_cur)。
   k_cur が 1.._k の範囲外（remove 後の _shrink でクラス消滅）の場合はそのセルを
   経路マップに入れない（off-path 扱い）。
2. 各セル (t, len 候補 c) について現行どおり正規化テーブル
   table[k-1] = log P(k | c) （文脈なし周辺、k=1.._k）を計算。
3. μ の決定（セルごと）:
   - セル (t, len) が経路マップに一致する場合（on-path）:
     `mu = log(be(_a,_b)) + table[k_cur-1]`  ← 現在の割当に条件付け
   - それ以外（off-path）: 現行どおり `id = rd::ln_draw(table); mu = log(be(_a,_b)) + table[id]`
4. 許可集合: table[i] >= mu の i+1 を l.k[t][len-1] に push（現行どおり）。
   log(be) <= 0 より、on-path セルでは k_cur が必ず生存し（正当性条件）、
   off-path セルでも id が必ず生存する（空セルなし＝格子連結性は現行と同等）。

### デバッグ用 assert（NDEBUG 時は無効）
_slice の末尾で、経路マップ上の全 (t, len, k_cur) が許可集合に含まれることを assert。

## 設計上の注意（実装者への申し送り）
- これは「セルごとの補助変数」による近似的ビームであり、on-path セルの条件付けにより
  現在状態の生存（既約性の要）を回復するのが目的。設計メモの μ_t（時刻ごと1変数）へ
  完全準拠させる案は、off-path セルが空になり格子連結性の扱いが複雑化するため本 WO では
  採らない。将来課題としてコメントを残すこと。
- _slice 内で参照する (*_chunk)[k]->lp / _class->lp は remove 済みモデルに対する評価で
  よい（μ の条件付き分布は現在の「割当」に依存すればよく、カウントには依存しない）。

## 受け入れ基準
1. ビルド（OpenMP 有無両方）
2. assert 有効ビルドで 100 文 x 50 epoch、assertion failure ゼロ
3. 同条件で WO-003 適用前後のコーパス log P トレースを比較し、適用後の方が
   高い水準で安定する（monitor に採取させ司令塔が判定）
4. parse 経路（cur=nullptr）の出力が従来と同形式で得られる
5. diff が上記仕様の範囲に収まっている
