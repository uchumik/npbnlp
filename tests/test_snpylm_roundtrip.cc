// WO-006 phase-1 + WO-007 acceptance: a full init/add over a synthetic
// switching-NPYLM corpus followed by estimate() and a reverse-order remove()
// must return every restaurant (G^bg, the generic-slot _bg_gen, per-class H_k /
// H_k-letter, the shared spelling VPYP) to c()==0, t()==0. Regression guard for
// count leaks in the snpylm add/remove/estimate ledger, including the WO-007
// targeted generic-slot ledger (NE position + exit-frame word + EOS seatings).
//
// Whole-tree emptiness of _bg / _bg_gen is verified two ways: (1) remove() must
// not throw -- any (a)/(b)/(c) seating whose remove find-path diverges from its
// add make-path leaves a "context not found" and aborts; (2) the ROOT reaching
// c()==0, t()==0 implies the whole tree is empty for a plain hpyp: a node's
// customer total equals the sum of its children's table totals, and in the CRP a
// node has customers iff it has tables, so zero at the root propagates down by
// induction. Run at n=2 and n=3 so the deeper gvid contexts are exercised.
#include"snpylm.h"
#include<cassert>
#include<cstdio>
#include<memory>
#include<random>
#include<vector>

using namespace std;
using namespace npbnlp;

// expose the protected restaurants snpylm keeps its counts in.
struct snpylm_probe : public snpylm {
	using snpylm::_bg;
	using snpylm::_bg_gen;
	using snpylm::_spell;
	using snpylm::_hk;
	using snpylm::_hkletter;
	using snpylm::_k;
	using snpylm::_pine;
	using snpylm::_piw;
	using snpylm::_pieos;
	using snpylm::_psi;
	using snpylm::_theta_sh;
	using snpylm::_theta_k;
	using snpylm::_gate_ne;
	using snpylm::_gate_o;
	snpylm_probe(int n, int hn, int hl, int k): snpylm(n, hn, hl, k) {}
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

static int run(int N) {
	const int NSENT = 60;
	const int NVOCAB = 24;
	const int HN = 1, HL = 8, K = 10;

	mt19937 rng(20260711 + N); // fixed seed for reproducibility
	uniform_int_distribution<int> wlen_d(2, 4);
	uniform_int_distribution<int> letter_d(0, 25);
	uniform_int_distribution<int> nchunk_d(3, 6);
	uniform_int_distribution<int> welen_d(1, 3);   // NE span length in words
	uniform_int_distribution<int> vocab_d(0, NVOCAB-1);
	uniform_int_distribution<int> class_d(1, 5);   // NE class z in [1,5]
	uniform_real_distribution<double> coin(0.0, 1.0);

	// shared, never-resized-again letter buffer and a small reused vocabulary,
	// mirroring how real corpora share word ids (word/chunk _doc must stay valid).
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

	// per-sentence word document (backs that sentence's chunks) and the
	// nsentence objects; sized upfront so no reallocation moves the vector<word>
	// objects the chunks' _doc points at.
	vector<vector<word> > docs(NSENT);
	vector<nsentence> corpus(NSENT);
	for (int i = 0; i < NSENT; ++i) {
		int nc = nchunk_d(rng);
		vector<word>& doc = docs[i];
		// reserve so the doc never reallocates while chunks point into it.
		doc.reserve(nc*3);
		nsentence& s = corpus[i];
		int head = 0;
		for (int cx = 0; cx < nc; ++cx) {
			bool ne = (coin(rng) < 0.30); // ~30% NE chunks
			int wc = ne ? welen_d(rng) : 1; // O chunks are length 1
			for (int w = 0; w < wc; ++w)
				doc.push_back(vocab[vocab_d(rng)]);
			chunk ch(doc, head, wc);
			ch.k = ne ? class_d(rng) : 0; // z: 0 = O, >=1 = NE class
			s.c.emplace_back(ch);
			head += wc;
		}
		s.n.resize(s.c.size()+1, 0);
	}

	snpylm_probe lm(N, HN, HL, K);
	// WO-012: exercise the soft NE gate in its self-estimating mode, where the
	// per-chunktype NE / O counters are actually read by the score. The counters
	// are seated for EVERY segment (O and NE alike), so a full add/remove cycle
	// must return them all to zero exactly like the theta ledger.
	lm.set_gate_learn(true);

	try {
		for (int i = 0; i < NSENT; ++i)
			lm.init(corpus[i]);
	} catch (const char *ex) {
		fprintf(stderr, "exception during init(): %s\n", ex);
		return 1;
	}

	// guard against a vacuous gate check: the counters must actually be seated by
	// init() (both sides -- the corpus has ~30% NE chunks and ~70% O), otherwise
	// the all-zero assertion after remove() would prove nothing.
	{
		long gne = 0, go = 0;
		for (size_t i = 0; i < lm._gate_ne.size(); ++i)
			gne += lm._gate_ne[i];
		for (size_t i = 0; i < lm._gate_o.size(); ++i)
			go += lm._gate_o[i];
		if (gne <= 0 || go <= 0) {
			fprintf(stderr, "gate ledger not exercised after init(): ne=%ld o=%ld\n", gne, go);
			return 1;
		}
	}

	// interleave estimate() (gibbs re-seats the G^bg base corpus and the H_k
	// bases); count preservation under gibbs means every root still falls to 0,0.
	try {
		lm.estimate(2);
	} catch (const char *ex) {
		fprintf(stderr, "exception during estimate(): %s\n", ex);
		return 1;
	}

	try {
		for (int i = NSENT-1; i >= 0; --i)
			lm.remove(corpus[i]);
	} catch (const char *ex) {
		fprintf(stderr, "remove() threw -> add/remove roundtrip not balanced: %s\n", ex);
		return 1;
	}

	bool ok = true;
	ok &= root_zero(lm._bg.get(), "bg", 0);
	ok &= root_zero(lm._bg_gen.get(), "bg_gen", 0);
	ok &= root_zero(lm._spell.get(), "spell", 0);
	for (int i = 0; i <= lm._k; ++i) {
		ok &= root_zero((*lm._hk)[i].get(), "H", i);
		ok &= root_zero((*lm._hkletter)[i].get(), "Hletter", i);
	}
	// switching counters must also net to zero after a full remove.
	if (lm._pine != 0 || lm._piw != 0 || lm._pieos != 0) {
		fprintf(stderr, "leak: switching counters pine=%d piw=%d pieos=%d (expected 0)\n",
				lm._pine, lm._piw, lm._pieos);
		ok = false;
	}
	// psi char-type counts must net to zero too.
	for (size_t i = 0; i < lm._psi.size(); ++i)
		if (lm._psi[i] != 0) {
			fprintf(stderr, "leak: psi[%zu]=%d (expected 0)\n", i, lm._psi[i]);
			ok = false;
			break;
		}
	// theta chunktype counts (both layers) must net to zero too (WO-009).
	for (size_t i = 0; i < lm._theta_sh.size(); ++i)
		if (lm._theta_sh[i] != 0) {
			fprintf(stderr, "leak: theta_sh[%zu]=%d (expected 0)\n", i, lm._theta_sh[i]);
			ok = false;
			break;
		}
	for (size_t i = 0; i < lm._theta_k.size(); ++i)
		if (lm._theta_k[i] != 0) {
			fprintf(stderr, "leak: theta_k[%zu]=%d (expected 0)\n", i, lm._theta_k[i]);
			ok = false;
			break;
		}
	// soft NE gate counters, both sides, must net to zero too (WO-012).
	for (size_t i = 0; i < lm._gate_ne.size(); ++i)
		if (lm._gate_ne[i] != 0) {
			fprintf(stderr, "leak: gate_ne[%zu]=%d (expected 0)\n", i, lm._gate_ne[i]);
			ok = false;
			break;
		}
	for (size_t i = 0; i < lm._gate_o.size(); ++i)
		if (lm._gate_o[i] != 0) {
			fprintf(stderr, "leak: gate_o[%zu]=%d (expected 0)\n", i, lm._gate_o[i]);
			ok = false;
			break;
		}

	if (!ok) {
		fprintf(stderr, "test_snpylm_roundtrip FAILED (n=%d): non-zero counters after full remove()\n", N);
		return 1;
	}
	printf("test_snpylm_roundtrip OK (n=%d)\n", N);
	return 0;
}

int main() {
	if (run(2) != 0)
		return 1;
	if (run(3) != 0)
		return 1;
	return 0;
}
