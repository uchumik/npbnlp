// WO-006 phase-2 acceptance (self-contained, no external tokenizer): run the
// full blocked-Gibbs inference (remove -> sample -> add) plus MAP parse over a
// synthetic tokenized corpus, then remove everything and assert the ledger
// closes (roots back to 0,0). Exercises _slice / _forward / _backward / the
// emission+transition and the O(=class 0, length 1) constraint end to end.
// Built in Debug (NDEBUG unset) the slice-survival assert in _slice is active.
#include"snpylm.h"
#include<cassert>
#include<cstdio>
#include<memory>
#include<random>
#include<vector>

using namespace std;
using namespace npbnlp;

struct snpylm_probe : public snpylm {
	using snpylm::_bg;
	using snpylm::_spell;
	using snpylm::_hk;
	using snpylm::_hkletter;
	using snpylm::_k;
	snpylm_probe(int n, int hn, int hl, int k): snpylm(n, hn, hl, k) {}
};

static bool root_zero(hpyp *h, const char *name, int idx) {
	context *r = h->h();
	if (r->c() != 0 || r->t() != 0) {
		fprintf(stderr, "leak: %s[%d] c=%d t=%d\n", name, idx, r->c(), r->t());
		return false;
	}
	return true;
}

int main() {
	const int NSENT = 40;
	const int NVOCAB = 20;
	const int N = 2, HN = 1, HL = 6, K = 6;
	const int EPOCH = 6;

	mt19937 rng(20260712);
	uniform_int_distribution<int> wlen_d(1, 3);
	uniform_int_distribution<int> letter_d(0, 15);
	uniform_int_distribution<int> nword_d(3, 8);
	uniform_int_distribution<int> vocab_d(0, NVOCAB-1);

	// persistent letter buffer + reused vocabulary with real wid ids.
	vector<unsigned int> letters;
	vector<word> vocab;
	shared_ptr<wid> wdic = wid::create();
	for (int v = 0; v < NVOCAB; ++v) {
		int len = wlen_d(rng);
		int head = (int)letters.size();
		for (int i = 0; i < len; ++i)
			letters.push_back((unsigned int)('a' + letter_d(rng)));
		word w(letters, head, len);
		w.id = wdic->index(w);
		vocab.push_back(w);
	}

	// synthetic tokenized sentences (each word references the shared buffer).
	vector<sentence> ws(NSENT);
	for (int i = 0; i < NSENT; ++i) {
		int nw = nword_d(rng);
		for (int j = 0; j < nw; ++j)
			ws[i].w.push_back(vocab[vocab_d(rng)]);
		ws[i].n.resize(nw+1, 0);
	}
	nio f(ws);

	snpylm_probe lm(N, HN, HL, K);
	lm.slice(1.0, 5.0);

	// cold start: all-O (each word a length-1 class-0 chunk).
	vector<nsentence> corpus(NSENT);
	for (int i = 0; i < NSENT; ++i) {
		int head = f.head[i], tail = f.head[i+1];
		for (int w = head; w < tail; ++w) {
			chunk c(*f.raw, w, 1);
			c.k = 0;
			corpus[i].c.emplace_back(c);
		}
		corpus[i].n.resize(corpus[i].c.size()+1, 0);
	}

	try {
		for (int e = 0; e < EPOCH; ++e) {
			for (int i = 0; i < NSENT; ++i) {
				if (e > 0)
					lm.remove(corpus[i]);
				nsentence s = lm.sample(f, i, &corpus[i]);
				// every sampled segmentation must cover the whole sentence.
				int cover = 0;
				for (int j = 0; j < s.size(); ++j) {
					cover += s.ch(j).len;
					// O is length-1 only; NE is class in [1,K].
					if (s.ch(j).k == 0)
						assert(s.ch(j).len == 1);
					assert(s.ch(j).k >= 0 && s.ch(j).k <= lm._k);
				}
				assert(cover == f.head[i+1]-f.head[i]);
				corpus[i] = s;
				lm.add(corpus[i]);
			}
			lm.estimate(3);
		}
	} catch (const char *ex) {
		fprintf(stderr, "exception during training: %s\n", ex);
		return 1;
	}

	// MAP parse must also produce a full cover for every sentence.
	try {
		for (int i = 0; i < NSENT; ++i) {
			nsentence s = lm.parse(f, i);
			int cover = 0;
			for (int j = 0; j < s.size(); ++j)
				cover += s.ch(j).len;
			if (cover != f.head[i+1]-f.head[i]) {
				fprintf(stderr, "parse cover mismatch sent %d: %d != %d\n",
						i, cover, f.head[i+1]-f.head[i]);
				return 1;
			}
		}
	} catch (const char *ex) {
		fprintf(stderr, "exception during parse: %s\n", ex);
		return 1;
	}

	// ledger closure: remove the whole trained corpus, every root back to 0,0.
	try {
		for (int i = NSENT-1; i >= 0; --i)
			lm.remove(corpus[i]);
	} catch (const char *ex) {
		fprintf(stderr, "remove after training threw: %s\n", ex);
		return 1;
	}
	bool ok = true;
	ok &= root_zero(lm._bg.get(), "bg", 0);
	ok &= root_zero(lm._spell.get(), "spell", 0);
	for (int i = 0; i <= lm._k; ++i) {
		ok &= root_zero((*lm._hk)[i].get(), "hk", i);
		ok &= root_zero((*lm._hkletter)[i].get(), "hkletter", i);
	}
	if (!ok) {
		fprintf(stderr, "test_snpylm_infer FAILED: ledger did not close after training\n");
		return 1;
	}
	printf("test_snpylm_infer OK\n");
	return 0;
}
