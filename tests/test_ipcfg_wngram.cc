#include"ipcfg.h"
#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<memory>
#include<vector>

using namespace npbnlp;

// WO-014: the pre-terminal emission became P(A) P(w_i | w_{i-1}.., A).  The add
// path builds its word context with hpyp::make(sentence&,int) and the remove
// path walks the same key sequence with a strict find, so the whole point of
// this test is that a corpus added at order n unseats back to a completely
// empty model -- including every restaurant at depth >= 1, which ipcfg::empty()
// checks through hpyp::empty()'s recursion over context::_child.
static bool roundtrip(io& f, int wn) {
	ipcfg g;
	g.set(8);
	g.word_ngram(wn);
	if (g.word_ngram() != wn) {
		fprintf(stderr, "word_ngram(%d) was not applied\n", wn);
		return false;
	}
	int n = (int)f.head.size()-1;
	std::vector<tree> corpus_trees;
	for (int i = 0; i < n; ++i)
		corpus_trees.push_back(g.init(f, i));
	if (!g.valid()) {
		fprintf(stderr, "n=%d: CRP counts inconsistent after init\n", wn);
		return false;
	}
	// One blocked-Gibbs sweep: remove, resample on a lattice conditioned on the
	// current tree, add.  This exercises the scoring paths (which must use
	// find(), never make()) between a matched remove/add pair.
	for (int i = 0; i < n; ++i) {
		g.remove(corpus_trees[i]);
		corpus_trees[i] = g.sample(f, i, &corpus_trees[i]);
		g.add(corpus_trees[i]);
		if (!g.valid()) {
			fprintf(stderr, "n=%d: CRP counts inconsistent after resampling %d\n", wn, i);
			return false;
		}
	}
	for (int i = 0; i < n; ++i)
		g.remove(corpus_trees[i]);
	if (!g.empty()) {
		fprintf(stderr, "n=%d: add/remove did not round-trip to empty\n", wn);
		return false;
	}
	return true;
}

// The order is baked into hpyp at construction, so switching it after the model
// holds customers would silently discard them.  It must be refused instead.
static bool rejects_late_change(io& f) {
	ipcfg g;
	g.set(8);
	g.word_ngram(2);
	tree t = g.init(f, 0);
	bool threw = false;
	try {
		g.word_ngram(3);
	} catch (const char *ex) {
		threw = true;
	}
	g.remove(t);
	if (!threw) {
		fprintf(stderr, "word_ngram() accepted an order change on a non-empty model\n");
		return false;
	}
	return true;
}

// Danger 3: hpyp::load overwrites _n unconditionally, so a model trained at one
// order must never be resumed at another.  The stored order wins.
static bool file_order_wins(io& f) {
	const char *path = "/tmp/test_ipcfg_wngram.model";
	{
		ipcfg g;
		g.set(8);
		g.word_ngram(2);
		tree t = g.init(f, 0);
		g.save(path);
		g.remove(t);
	}
	ipcfg loaded;
	loaded.word_ngram(3);
	loaded.load(path);
	remove(path);
	if (loaded.word_ngram() != 2) {
		fprintf(stderr, "loaded order %d, expected the stored 2\n", loaded.word_ngram());
		return false;
	}
	if (!loaded.valid()) {
		fprintf(stderr, "loaded model has inconsistent CRP counts\n");
		return false;
	}
	return true;
}

int main(int argc, char **argv) {
	if (argc != 2) {
		fprintf(stderr, "usage: %s corpus\n", argv[0]);
		return 2;
	}
	try {
		// One io for the whole process: the keys in the global wid singleton
		// are words pointing into io's raw buffer, so a second io would leave
		// dangling keys behind it.
		io f(argv[1]);
		// n=1 is the historical unigram emission and must keep working.
		for (int wn = 1; wn <= 3; ++wn) {
			if (!roundtrip(f, wn))
				return 1;
		}
		if (!rejects_late_change(f))
			return 1;
		if (!file_order_wins(f))
			return 1;
	} catch (const char *ex) {
		fprintf(stderr, "%s\n", ex);
		return 1;
	}
	printf("ok\n");
	return 0;
}
