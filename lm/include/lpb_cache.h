#ifndef NPBNLP_LPB_CACHE_H
#define NPBNLP_LPB_CACHE_H

#include"word.h"
#include"chunk.h"
#include<vector>
namespace npbnlp {
	struct lpb_key {
		const void *lm;
		std::vector<int> v;
		bool operator==(const lpb_key& k) const {
			return lm == k.lm && v == k.v;
		}
	};
	lpb_key lpb_word_key(const void *lm, int kind, word& w, bool m);
	lpb_key lpb_chunk_key(const void *lm, int kind, chunk& b, bool n, bool m);
	bool lpb_find(const lpb_key& k, double& lp);
	void lpb_store(const lpb_key& k, double lp);
	void lpb_clear();
}
#endif
