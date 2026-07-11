# WO-001: クラス遷移の有効化とカウントリーク防止

- 担当: impl-routine (sonnet)
- 対象: tg/src/ne.cc, tg/src/nphsmm.cc
- 背景: diagnosis スキル P1/P2/P3。デフォルト n=1 で遷移が存在せず、n≥2 では remove() が
  無音でカウントをリークする。格子の未知チャンク(id=1)が文脈木で衝突する。

## 変更1: ne.cc — デフォルト次数を 2 に

before:
```cpp
static int n = 1;
```
after:
```cpp
static int n = 2;
```
usage() 内の `"-n, --chunk_order=int(default 1)\n"` を `default 2` に、
`"-k, --class=int(default 500)\n"` を `default 50` に修正（表記と実値の一致）。

## 変更2: nphsmm.cc remove() — find の NULL を fail-fast に

before（remove() 内の第2ループ）:
```cpp
	for (int i = 0; i < s.size()+1; ++i) {
		chunk& ch = s.ch(i);
		context *h = (*_chunk)[ch.k]->find(s, i);
		(*_chunk)[ch.k]->remove(ch, h);
		context *c = _class->h();
		for (int j = 1; j < _n; ++j) {
			chunk& x = s.ch(i-j);
			c = c->find(x.k);
		}
		_class->remove(ch.k, c);
	}
```
after:
```cpp
	for (int i = 0; i < s.size()+1; ++i) {
		chunk& ch = s.ch(i);
		context *h = (*_chunk)[ch.k]->find(s, i);
		if (!h)
			throw "emission context not found in nphsmm::remove";
		(*_chunk)[ch.k]->remove(ch, h);
		context *c = _class->h();
		for (int j = 1; j < _n && c; ++j) {
			chunk& x = s.ch(i-j);
			c = c->find(x.k);
		}
		if (!c)
			throw "class context not found in nphsmm::remove";
		_class->remove(ch.k, c);
	}
```
方針: 握りつぶし(no-op)ではなく throw。リークはモデル不整合のバグ表面化なので隠さない。

## 変更3: nphsmm.cc — 未知チャンク id=1 を文脈キーにしない

sample() / parse() のトップレベル 2 箇所 + _forward / _backward の再帰内 2 箇所、計 4 箇所。

before（例: sample() 内）:
```cpp
					if (_n > 1)
						h = c->find(prev.id);
```
after:
```cpp
					if (_n > 1 && prev.id != 1)
						h = c->find(prev.id);
```
_forward/_backward 内は対象変数が `y` である点に注意:
before: `if (!unk && n > 1)` → after: `if (!unk && n > 1 && y.id != 1)`
（id 0 = BOS/EOS は正当な文脈キーなので除外しないこと）

## 受け入れ基準
1. OpenMP 有効/無効の両方でビルドが通る
2. add/remove ラウンドトリップ（diagnosis §3.1）: n=2 で 100 文 init→add→remove 後、
   _class と全 (*_chunk)[k] の客数・テーブル数が 0
3. n=2, K=20, 100文 x 20 epoch で例外なく完走し、コーパス log P が悪化トレンドでない
4. 上記以外のコード変更が diff に含まれない
