#include"lpb_cache.h"
#include<unordered_map>

using namespace std;
namespace npbnlp {
	atomic<unsigned long> lpb_generation(0);
	struct lpb_hash {
		size_t operator()(const lpb_key& k) const {
			size_t h = reinterpret_cast<size_t>(k.lm);
			for (int x : k.v)
				h = h * 1000003u ^ static_cast<unsigned int>(x);
			return h;
		}
	};
	static thread_local unordered_map<lpb_key, double, lpb_hash> _lpb;
	static thread_local unsigned long _generation = 0;

	static void push(lpb_key& k, word& w, bool m) {
		k.v.push_back(w.len);
		for (int i = 0; i < w.len; ++i)
			k.v.push_back((int)w[i]);
		if (m)
			k.v.insert(k.v.end(), w.m.begin(), w.m.end());
	}

	lpb_key lpb_word_key(const void *lm, int kind, word& w, bool m) {
		lpb_key k{lm, {kind}};
		push(k, w, m);
		return k;
	}

	lpb_key lpb_chunk_key(const void *lm, int kind, chunk& b, bool n, bool m) {
		lpb_key k{lm, {kind, b.len}};
		for (int i = 0; i < b.len+1; ++i) {
			k.v.push_back(b[i]);
			if (n)
				k.v.push_back(b.n[i]);
			push(k, b.wd(i), m);
		}
		return k;
	}

	bool lpb_find(const lpb_key& k, double& lp) {
		unsigned long g = lpb_generation.load();
		if (_generation != g) {
			_lpb.clear();
			_generation = g;
		}
		auto it = _lpb.find(k);
		if (it == _lpb.end())
			return false;
		lp = it->second;
		return true;
	}

	void lpb_store(const lpb_key& k, double lp) {
		_lpb[k] = lp;
	}
}
