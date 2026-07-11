# WO-002: hpyp::gibbs() のチャンク基底コーパス (_cbc) 対応

> **改訂 (2026-07-12, 司令塔)**: 本 WO は「base corpus の静的統合」設計に置き換える。
> 理由: (1) 下記 after 案はループを複製し、かつ _cbc 側で wrap::add_a/remove_a を直呼び
> するため `_cbase_add/_cbase_remove` フック(posbase の基底委譲)を素通りする。
> (2) _bc/_cbc の add/remove/gibbs 機構は完全相似で、`template<class T> using bcorpus
> = unordered_map<int, vector<T>>` + 着席 shim(`_seat_base(word&)`/`_seat_base(chunk&)`
> のオーバーロードに cbase 分岐を封じ込め)+ テンプレート共通実装 `_bc_add/_bc_remove/
> _gibbs_impl` に統合できる。gibbs は存在するコーパス(両方あれば両方)に _gibbs_impl を
> 回す形にし、P4 はその系として解消する。shuffle 結果 rd[j] の未使用バグも同時に修正。
> poisson_correction/estimate_l は word 固有のため _bc ガードのまま。
> 担当は **impl-core (opus)** に変更(CRP 着席/退席/再配置の不変量に触れるため)。
> 対象: lm/include/hpyp.h, lm/src/hpyp.cc(+ tests/test_nphsmm_roundtrip.cc の拡張)。
> 受け入れ基準は本文の 1-3 に加え: ctest 全件 green(roundtrip は add 後に estimate(3) を
> 挟む拡張版)、orig/eff parity 維持、--posbase tiny 学習完走、変更ファイルは上記のみ。
> 以下の原文 before/after は背景資料として残す(そのままは適用しない)。

- 担当: ~~impl-routine (sonnet)~~ → **impl-core (opus)**(改訂)
- 対象: lm/src/hpyp.cc
- 背景: diagnosis スキル P4。gibbs() が _bc（word 基底コーパス）しか見ておらず、
  チャンク HPYP（基底コーパスは _cbc）では即 return する。このため
  nphsmm::estimate() 内の (*_chunk)[i]->gibbs(iter) が無動作となり、
  チャンク→単語基底へのテーブル再配置（座席の再サンプル）が行われない。

## 変更: lm/src/hpyp.cc gibbs()

before:
```cpp
void hpyp::gibbs(int iter) {
	if (!_base || _bc == nullptr)
		return;
	for (int i = 0; i < iter; ++i) {
		for (auto it = _bc->begin(); it != _bc->end(); ++it) {
			int size = it->second.size();
			int rd[size] = {0};
			rd::shuffle(rd, size);
			for (int j = 0; j < size; ++j) {
				lock_guard<mutex> m(_mutex);
				wrap::remove_a(it->second[j], _base);
				wrap::add_a(it->second[j], _base);
			}
		}
	}
}
```
after:
```cpp
void hpyp::gibbs(int iter) {
	if (!_base || (_bc == nullptr && _cbc == nullptr))
		return;
	for (int i = 0; i < iter; ++i) {
		if (_bc != nullptr) {
			for (auto it = _bc->begin(); it != _bc->end(); ++it) {
				int size = it->second.size();
				int rd[size] = {0};
				rd::shuffle(rd, size);
				for (int j = 0; j < size; ++j) {
					lock_guard<mutex> m(_mutex);
					wrap::remove_a(it->second[rd[j]], _base);
					wrap::add_a(it->second[rd[j]], _base);
				}
			}
		}
		if (_cbc != nullptr) {
			for (auto it = _cbc->begin(); it != _cbc->end(); ++it) {
				int size = it->second.size();
				int rd[size] = {0};
				rd::shuffle(rd, size);
				for (int j = 0; j < size; ++j) {
					lock_guard<mutex> m(_mutex);
					wrap::remove_a(it->second[rd[j]], _base);
					wrap::add_a(it->second[rd[j]], _base);
				}
			}
		}
	}
}
```
注意: 既存コードは shuffle した rd[] を作りながら j を直接使っていた（shuffle が無意味）。
本 WO で rd[j] 参照に統一する。remove_a/add_a は wrap のディスパッチにより
word/chunk の両オーバーロードが存在することを convinience.h で確認済み。

## 受け入れ基準
1. 全バイナリ（ma/ tg/ pa/ sb/）のビルドが通る（gibbs は共有コードのため全影響確認）
2. add/remove ラウンドトリップ（gibbs(5) を挟んでも客数・テーブル数の総和が保存され、
   全 remove 後に 0）
3. ma/ws（単語分割）の小規模学習が従来どおり完走する（word 側の rd[j] 化のリグレッション確認）
