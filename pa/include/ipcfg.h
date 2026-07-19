#ifndef NPBNLP_IPCFG_H
#define NPBNLP_IPCFG_H

#include"io.h"
#include"tree.h"
#include"hpyp.h"
#include"vpyp.h"
#include"vtable.h"
#include"cyk.h"
#include<functional>
#include<memory>
#include<atomic>
#include<unordered_map>
#include<cstdio>

namespace npbnlp {
	class ipcfg {
		public:
			ipcfg();
			ipcfg(int m);
			virtual ~ipcfg();
			virtual tree sample(io& f, int i);
			virtual tree sample(io& f, int i, tree *cur);
			// Slice-conditioned CYK proposal used by the MH correction.  The
			// current tree and the proposal are both scored on the same sampled
			// lattice, using root-inside plus traceback probabilities.
			virtual tree mh_propose(io& f, int i, tree *cur, double& log_q,
								 double& log_q_cur);
			// Evaluate the same recursive generation process as init(): every
			// predictive probability is read before its matching add().  The tree
			// is left added to this model on return.
			virtual double mh_logprob_and_add(tree& t);
			// A serialization-based score snapshot is required because hpyp's C++
			// copy constructor deliberately shares CRP state.  It is used only for
			// predictive scoring (never remove/gibbs), because base-corpus witness
			// lists are intentionally not part of persisted models.
			virtual std::unique_ptr<ipcfg> snapshot() const;
			// Sequential, model-based initialization.  This creates and returns a
			// complete latent tree while updating the HPYPs as every production is
			// generated; it deliberately does not invoke the slice sampler.
			virtual tree init(io& f, int i);
			virtual tree parse(io& f, int i);
			virtual void add(tree& t);
			virtual void remove(tree& t);
			virtual void estimate(int iter);
			virtual void poisson_correction(int n = 100);
			// Set only the maximum number of latent categories.  HPYP/VPYP
			// vocabularies are inferred by the language models themselves.
			virtual void set(int k);
			// Order of the pre-terminal word HPYPs: the emission becomes
			// P(A) P(w_i | w_{i-1},...,w_{i-n+1}, A).  n == 1 is the historical
			// unigram emission and is bit-exact with it.  Must be called before
			// any tree is added (the HPYPs are rebuilt) and before load(), whose
			// stored order wins.
			virtual void word_ngram(int n);
			int word_ngram() const { return _wn; }
			// Diagnostic: per-class size of the word base corpus (_bc).  Deeper
			// n-gram contexts starve the base measure that feeds the Poisson
			// length correction, so this is reported once after init.
			void base_corpus_sizes(std::vector<long long>& out) const;
			virtual void slice(double a, double b);
			// Geometric prior over non-root internal span widths.  A span of
			// width d=j-i emits one stop and d-1 continue events.
			virtual void span(double a = 1., double b = 1.);
			bool span_enabled() const { return _span; }
			double span_probability() const { return _span_p; }
			long long span_stops() const { return _span_stop; }
			long long span_continues() const { return _span_continue; }
			// Truncated geometric prior over the split point of every internal
			// node, including the root.  For a span of width w=j-i+1 split at b,
			// the left-child width L=b-i+1 lives in [1,w-1] and
			// P(b|i,j) = q(1-q)^(L-1) / (1 - (1-q)^(w-1)).
			// Unlike span(), this applies to the root as well: the root's split
			// direction is part of the branching bias the prior is meant to learn.
			virtual void split(double a = 1., double b = 1., double q = .5,
							bool fixed = false);
			bool split_enabled() const { return _split; }
			double split_probability() const { return _split_q; }
			long long split_count() const { return _split_n; }
			long long split_left_excess() const { return _split_sum; }
			// Public only so the unit test can check the truncated normalisation
			// Sum_{L=1}^{w-1} exp(split_logprob(i,j,b)) == 1.
			double split_logprob(int i, int j, int b) const { return _split_lp(i, j, b); }
			int category_count() const { return _k; }
			// Atomically return and clear slice candidate diagnostics accumulated
			// by sample().  Values exclude the structural root cell.
			void slice_diagnostics(long long& terminal_cells, long long& terminal_labels,
							long long& internal_cells, long long& internal_labels);
			// Remove only trailing categories that are absent from a fully-added
			// corpus.  Do not call while a sentence is temporarily held out.
			void compact();
			bool valid() const;
			bool empty() const;
			virtual void save(const char *file);
			virtual void load(const char *file);
		private:
			int _m;
			int _k;
			int _K;
			int _v;
			// Effective order of every (*_word)[k].  All classes must share it;
			// load() enforces that and lets the stored value win over the CLI.
			int _wn;
			double _a;
			double _b;
			bool _span;
			// All class-specific word HPYPs back off to this one shared character
			// VPYP.  The vector retains one slot per class solely for legacy model
			// serialization and bounds checks; its entries alias when enabled.
			bool _shared_letter;
			bool _split;
			bool _split_fixed;
			double _span_a;
			double _span_b;
			double _span_p;
			long long _span_stop;
			long long _span_continue;
			double _split_a;
			double _split_b;
			double _split_q;
			// Sufficient statistics for q, maintained with exactly the same
			// discipline as _span_stop/_span_continue: they describe the current
			// corpus state and return to zero once every tree is removed.
			// Width-2 spans carry no information about q and are excluded.
			long long _split_n;
			long long _split_sum;
			std::vector<long long> _split_hist;
			std::atomic<long long> _slice_terminal_cells;
			std::atomic<long long> _slice_terminal_labels;
			std::atomic<long long> _slice_internal_cells;
			std::atomic<long long> _slice_internal_labels;
			std::unordered_map<int, int> _tfreq;
			std::shared_ptr<hpyp> _nonterm;
			std::shared_ptr<std::vector<std::shared_ptr<hpyp> > > _word;
			std::shared_ptr<std::vector<std::shared_ptr<vpyp> > > _letter;
			std::mutex _mutex;
			double _traceback(cyk& c, int i, int j, int z, vt& a, tree& tr, bool best = false);
			double _traceback_logprob(cyk& c, int i, int j, int z, vt& a, tree& tr);
			tree _sample(io& f, int i, tree *cur, bool full_cyk, double *log_q,
						 double *log_q_cur = nullptr);
			void _init_node(tree& t, int idx, int label);
			double _init_logprob_and_add(tree& t, int idx);
			void _add(tree& t, int i);
			void _remove(tree& t, int i);
			void _check_label(const node& z, const char *where) const;
			// Strict counterpart of hpyp::find(sentence&,int) for _remove.
			// hpyp::find silently stops at the deepest existing node, so a
			// shallow return would un-seat a customer from a restaurant that
			// never seated one (and _bc_remove would then evict a random
			// witness, irrecoverably).  _add always seats at the full-depth
			// node built by make(), so failing to reach depth _wn-1 means the
			// ledger is already broken: throw instead of corrupting it.
			context* _word_context_remove(int k, sentence& s, int i) const;
			// Grammar factorisation used everywhere a binary rule is scored:
			// G_L(B|A) G_R(C|A,B).  The contexts are kept in one HPYP so their
			// backing-off hierarchy is still learned from data.
			double _rule_lp(int parent, int left, int right) const;
			void _rule_add(int parent, int left, int right);
			void _rule_remove(int parent, int left, int right);
			void _calc_preterm(cyk& c, int j, vt& a);
			void _calc_nonterm(cyk& c, int i, int j, vt& a);
			void _slice(cyk& l, tree *cur, bool full_cyk = false);
			void _slice_preterm(cyk& l, int i);
			void _slice_preterm_cond(cyk& l, int i, int label);
			//void _slice_nonterm(cyk& c, int i, int j);
			double _draw(cyk& c, int i, int j);
			double _marginalize(cyk& c, int i, int j);
			void _slice_nonterm(cyk& c, int i, int j, double mu);
			void _slice_nonterm_cond(cyk& c, int i, int j, int lc, int rc, int kc, int mc);
			void _slice_root(cyk& c);
			void _slice_root_cond(cyk& c, int lc, int rc, int bc);
			void _collect_spans(tree& t, int idx, std::vector<std::vector<const node*> >& on);
			bool _parent_allowed(int parent, int left, int right) const;
			void _record_slice(cyk& c);
			double _span_lp(cyk& c, int i, int j);
			double _split_lp(int i, int j, int b) const;
			void _split_add(int i, int j, int b);
			void _split_remove(int i, int j, int b);
			void _estimate_split();
			void _resize();
			void _shrink();
			void _share_letters();
			void _unshare_letters();
			void _save(FILE *fp) const;
			void _load(FILE *fp);
	};
}
#endif
