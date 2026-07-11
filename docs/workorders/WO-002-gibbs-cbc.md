# WO-002: hpyp::gibbs() のチャンク基底コーパス (_cbc) 対応

- 担当: impl-routine (sonnet)
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
