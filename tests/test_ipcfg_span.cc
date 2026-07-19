#include"ipcfg.h"
#include<cstdio>
#include<cstdlib>
#include<cmath>

using namespace npbnlp;

int main(int argc, char **argv) {
	if (argc != 2) {
		fprintf(stderr, "usage: %s corpus\\n", argv[0]);
		return 2;
	}
	try {
		io f(argv[1]);
		// init() updates the model itself.  Its returned tree must therefore
		// remove cleanly without any intervening add().
		ipcfg initialized;
		initialized.set(8);
		initialized.span(1., 1.);
		tree initialized_tree = initialized.init(f, 0);
		if (!initialized.valid()) {
			fprintf(stderr, "iPCFG CRP counts are inconsistent after init\n");
			return 1;
		}
		initialized.remove(initialized_tree);
		if (!initialized.empty()) {
			fprintf(stderr, "iPCFG init/remove did not round-trip to empty\n");
			return 1;
		}

		// MH proposal QC: the current tree is held out, both proposal
		// probabilities are evaluated on the same current-tree-conditioned
		// slice lattice, and removing a rejected proposal restores a model that
		// can re-add the old tree without CRP leakage.
		ipcfg mh;
		mh.set(8);
		mh.span(1., 1.);
		tree old = mh.init(f, 0);
		mh.remove(old);
		std::unique_ptr<ipcfg> score_state = mh.snapshot();
		double log_q_new = 0., log_q_old = 0.;
		tree proposal = mh.mh_propose(f, 0, &old, log_q_new, log_q_old);
		double log_p_old = score_state->mh_logprob_and_add(old);
		double log_p_new = mh.mh_logprob_and_add(proposal);
		if (!std::isfinite(log_q_new) || !std::isfinite(log_q_old) ||
		    !std::isfinite(log_p_old) || !std::isfinite(log_p_new)) {
			fprintf(stderr, "non-finite iPCFG MH score\n");
			return 1;
		}
		mh.remove(proposal);
		mh.add(old);
		if (!mh.valid()) {
			fprintf(stderr, "iPCFG CRP counts are inconsistent after MH reject path\n");
			return 1;
		}
		mh.remove(old);
		if (!mh.empty()) {
			fprintf(stderr, "iPCFG MH reject path did not round-trip to empty\n");
			return 1;
		}

		ipcfg g;
		g.set(8);
		g.span(1., 1.);
		const int initial_categories = g.category_count();
		tree t = g.sample(f, 0);
		// Exercise the highest valid label explicitly: compact() must not pop
		// its model while it is still present in the fully-added corpus.
		for (auto& n : t.c) {
			if (n.k >= 0)
				n.k = initial_categories;
		}
		t[t.s.size()-1].k = 0; // structural root
		g.add(t);
		if (!g.valid()) {
			fprintf(stderr, "iPCFG CRP counts are inconsistent after add\n");
			return 1;
		}
		// A binary tree over the five words in the fixture has three non-root
		// internal nodes; every one contributes exactly one stop event.
		if (g.span_stops() != t.s.size()-2) {
			fprintf(stderr, "unexpected span stop count: %lld\\n", g.span_stops());
			return 1;
		}
		g.compact();
		if (g.category_count() != initial_categories) {
			fprintf(stderr, "compact discarded an active maximum category\n");
			return 1;
		}
		g.estimate(1);
		if (!g.span_enabled() || !(g.span_probability() > 0.) ||
		    !(g.span_probability() < 1.)) {
			fprintf(stderr, "invalid span posterior draw\\n");
			return 1;
		}
		g.save("/tmp/qc_ipcfg_span.model");
		ipcfg restored;
		restored.load("/tmp/qc_ipcfg_span.model");
		if (!restored.span_enabled() || restored.span_stops() != g.span_stops() ||
		    restored.span_continues() != g.span_continues()) {
			fprintf(stderr, "span prior was not restored\\n");
			return 1;
		}
		g.remove(t);
		if (g.category_count() != initial_categories) {
			fprintf(stderr, "category was compacted while its tree was held out\n");
			return 1;
		}
		if (g.span_stops() != 0 || g.span_continues() != 0) {
			fprintf(stderr, "span counts did not round-trip to zero\\n");
			return 1;
		}
		if (!g.empty()) {
			fprintf(stderr, "iPCFG CRP counts did not round-trip to empty\n");
			return 1;
		}
		g.compact();
		if (g.category_count() != 1) {
			fprintf(stderr, "empty categories were not compacted at epoch boundary\n");
			return 1;
		}
		// Assignment must replace the latent tree, not append to it.  In the
		// latter case iPCFG indexes the stale root at N-1 after every sample.
		tree replacement(t);
		replacement[replacement.s.size()-1].k = 7;
		tree slot(t);
		slot = replacement;
		if (slot.c.size() != replacement.c.size() ||
		    slot[slot.s.size()-1].k != 7) {
			fprintf(stderr, "tree copy assignment did not replace state\n");
			return 1;
		}
		tree malformed(t);
		malformed[malformed.s.size()-1].k = 99;
		bool rejected = false;
		try {
			g.add(malformed);
		} catch (const char *) {
			rejected = true;
		}
		if (!rejected) {
			fprintf(stderr, "inactive label was not rejected\n");
			return 1;
		}
	} catch (const char *ex) {
		fprintf(stderr, "exception: %s\\n", ex);
		return 1;
	}
	printf("test_ipcfg_span OK\\n");
	return 0;
}
