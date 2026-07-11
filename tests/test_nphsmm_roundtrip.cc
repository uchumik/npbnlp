// WO-001 acceptance criterion 2: add/remove roundtrip must return every
// HPYP/vpyp root (class, per-class chunk/word/letter LM) to c()==0, t()==0
// after a full init() then reverse-order remove() of the same corpus.
// Regression guard for count leaks in nphsmm::add/init/remove.
#include"nphsmm.h"
#include"chunktype.h"
#include<cassert>
#include<cstdio>
#include<memory>
#include<random>
#include<vector>

using namespace std;
using namespace npbnlp;

// expose the protected LM members nphsmm keeps its restaurants in.
struct nphsmm_probe : public nphsmm {
	using nphsmm::_class;
	using nphsmm::_chunk;
	using nphsmm::_word;
	using nphsmm::_letter;
	using nphsmm::_k;
	nphsmm_probe(int n, int m, int l, int k): nphsmm(n, m, l, k) {}
};

static bool root_zero(hpyp *h, const char *name, int idx) {
	context *r = h->h();
	int c = r->c();
	int t = r->t();
	if (c != 0 || t != 0) {
		fprintf(stderr, "leak: %s[%d] root customers=%d tables=%d (expected 0,0)\n", name, idx, c, t);
		return false;
	}
	return true;
}

int main() {
	const int NSENT = 60;
	const int NVOCAB = 24;
	const int N = 2, M = 3, L = 20, K = 10;

	mt19937 rng(20260711); // fixed seed for reproducibility
	uniform_int_distribution<int> wlen_d(2, 4);
	uniform_int_distribution<int> letter_d(0, 25);
	uniform_int_distribution<int> nchunk_d(3, 6);
	uniform_int_distribution<int> wperchunk_d(1, 3);
	uniform_int_distribution<int> vocab_d(0, NVOCAB-1);

	// synthetic document: a shared letter buffer (built once, never resized
	// again afterwards so word/chunk `_doc` pointers stay valid) and a small
	// reused vocabulary of words, mirroring how real corpora share word ids.
	vector<unsigned int> letters;
	vector<word> vocab;
	shared_ptr<wid> wdic = wid::create();
	for (int v = 0; v < NVOCAB; ++v) {
		int len = wlen_d(rng);
		int head = (int)letters.size();
		for (int i = 0; i < len; ++i)
			letters.push_back((unsigned int)('a' + letter_d(rng)));
		word w(letters, head, len);
		w.id = wdic->index(w); // real word dictionary, same as ne.cc::load_label
		vocab.push_back(w);
	}

	// per-sentence word document (backs the chunks of that sentence) and the
	// nsentence objects themselves; sized upfront so no reallocation moves
	// the vector<word> objects that chunk::_doc points at.
	vector<vector<word> > docs(NSENT);
	vector<nsentence> corpus(NSENT);
	for (int i = 0; i < NSENT; ++i) {
		int nc = nchunk_d(rng);
		vector<word>& doc = docs[i];
		nsentence& s = corpus[i];
		int head = 0;
		for (int cx = 0; cx < nc; ++cx) {
			int wc = wperchunk_d(rng);
			for (int w = 0; w < wc; ++w)
				doc.push_back(vocab[vocab_d(rng)]);
			chunk ch(doc, head, wc);
			ch.type = chunktype::get(ch); // same as init_corpus()/chunking() in ne.cc
			s.c.emplace_back(ch);
			head += wc;
		}
		s.n.resize(s.c.size()+1, 0);
	}

	nphsmm_probe lm(N, M, L, K);

	try {
		for (int i = 0; i < NSENT; ++i)
			lm.init(corpus[i]);
	} catch (const char *ex) {
		fprintf(stderr, "exception during init(): %s\n", ex);
		return 1;
	}

	// WO-002: interleave estimate() (which runs gibbs() on every chunk/word LM)
	// so the roundtrip also guards the _cbc table re-seating path. Count-
	// preservation under gibbs means every root must still fall back to 0,0.
	try {
		lm.estimate(3);
	} catch (const char *ex) {
		fprintf(stderr, "exception during estimate(): %s\n", ex);
		return 1;
	}

	try {
		for (int i = NSENT-1; i >= 0; --i)
			lm.remove(corpus[i]);
	} catch (const char *ex) {
		fprintf(stderr, "remove() threw -> add/remove roundtrip is not balanced (leak path found): %s\n", ex);
		return 1;
	}

	bool ok = true;
	ok &= root_zero(lm._class.get(), "class", 0);
	for (int i = 0; i <= lm._k; ++i) {
		ok &= root_zero((*lm._chunk)[i].get(), "chunk", i);
		ok &= root_zero((*lm._word)[i].get(), "word", i);
		ok &= root_zero((*lm._letter)[i].get(), "letter", i);
	}

	if (!ok) {
		fprintf(stderr, "test_nphsmm_roundtrip FAILED: non-zero root counters remain after full remove()\n");
		return 1;
	}
	printf("test_nphsmm_roundtrip OK\n");
	return 0;
}
