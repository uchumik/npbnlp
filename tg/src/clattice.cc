#include"clattice.h"
#include"chartype.h"
#include"wordtype.h"
#include"chunktype.h"
#include<cstdlib>
#include<cstdio>
#ifdef _OPENMP
#include<omp.h>
#endif

using namespace std;
using namespace npbnlp;

static chunk eos;
static vector<int> bos(1, 0);

clattice2::clattice2(nio& f, int i, std::vector<int>& chsize) {
	int head = f.head[i];
	int tail = f.head[i+1];
	vector<type> wt;
	for (auto j = head; j < tail; ++j) {
		wt.push_back(wordtype::get((*f.raw)[j]));
	}
	c.resize(wt.size());
	k.resize(wt.size());
	shared_ptr<cid> dic = cid::create();
	for (auto j = 0; j < (int)wt.size(); ++j) {
		chtype t = chunktype2::start(wt[j]);
		for (auto k = j; k >= 0 && j-k < chsize[t]; --k) {
			chunk ch(*f.raw, head+k, 1+j-k);
			ch.id = (*dic)[ch];
			ch.type = t;
			c[j].push_back(ch);
			if (k > 0) {
				t = chunktype2::transition(t, wt[k-1], wt[k]);
				if (t < 0)
					break;
			}
		}
		k[j].resize(c[j].size());
	}
	// diagnostics (env gated): dump tokenizer word boundaries and every chunk
	// candidate span in absolute char offsets, for measuring gold-NE lattice
	// coverage (does the transition table / _clength drop NE spans?).
	if (getenv("NPBNLP_LATTICE_COVER")) {
		int nw = (int)wt.size();
		std::vector<int> cum(nw+1, 0);
		for (int p = 0; p < nw; ++p)
			cum[p+1] = cum[p] + (*f.raw)[head+p].len; // word char length
		fprintf(stderr, "tok %d", i);
		for (int p = 0; p <= nw; ++p)
			fprintf(stderr, " %d", cum[p]);
		fprintf(stderr, "\n");
		for (int j = 0; j < nw; ++j)
			for (auto& ch : c[j]) {
				int s = j - (ch.len - 1); // start word index
				fprintf(stderr, "cov %d %d-%d %d\n", i, cum[s], cum[j+1], (int)ch.type);
			}
	}
}

clattice2::~clattice2() {
}

chunk& clattice2::ch(int i, int len) {
	if (i < 0 || i >= (int)c.size())
		return eos;
	if (len-1 >= (int)c[i].size())
		throw "invalid chunk size";
	return c[i][len-1];
}

chunk* clattice2::cp(int i, int len) {
	if (i < 0 || i >= (int)c.size())
		return &eos;
	if (len-1 >= (int)c[i].size())
		throw "invalid chunk size";
	return &c[i][len-1];
}

int clattice2::size(int i) {
	if (i < 0 || i >= (int)c.size())
		return 1;
	return c[i].size();
}

vector<int>::iterator clattice2::begin(int i, int j) {
	if (i < 0 || i >= (int)k.size())
		return bos.begin();
	return k[i][j].begin();
}

vector<int>::iterator clattice2::end(int i, int j) {
	if (i < 0 || i >= (int)k.size())
		return bos.end();
	return k[i][j].end();
}
