#ifndef NPBNLP_VHMM_H
#define NPBNLP_VHMM_H

#include "io.h"
#include "vhdp.h"
#include "hpyp.h"
#include "vpyp.h"
#include "vtable.h"
#include "vlattice.h"
#include <memory>
#include <mutex>
#include <vector>

namespace npbnlp {
	class vhmm {
		public:
			vhmm();
			vhmm(int n, int min_n, int m, int k);
			virtual ~vhmm();
			virtual sentence sample(sentence& s);
			virtual sentence parse(sentence& s);
			virtual void inference_init(sentence& s);
			virtual void add(sentence& s);
			virtual void remove(sentence& s);
			// Indexed forms store the emission depth needed by remove(); sentence::n[]
			// is already used for the transition order.  idx < 0 means do not record,
			// which is valid only for emission order 1.
			void add(sentence& s, int idx);
			void remove(sentence& s, int idx);
			void init(sentence& s, int idx);
			virtual void estimate(int iter);
			virtual void poisson_correction(int n = 3000);
			virtual void set(int v, int k);
			// Ceiling only: _v is a count the letter models learn, and ghmm has no
			// letter models to seed at all.
			void set_k(int k);
			virtual void slice(double a, double b);
			virtual void set_alpha(double value);
			// Emission n-gram order over preceding observed words; order 1 uses the
			// root context.
			void set_word_ngram(int n);
			int word_ngram() const;
			// Freeze the state space before inference.
			virtual void set_fixed();
			void emission_argmax(sentence& s);
			virtual void init(sentence& s);
			virtual int n();
			virtual int min_n();
			virtual int m();
			virtual int k();
			virtual void save(const char *file);
			virtual void load(const char *file);
			double log_probability(sentence& s);
			void log_probability_parts(sentence& s, double& em, double& tr);
			void draw_order(sentence& s);
			long long clamp_hits() const;
			long long clamp_total() const;
			double mean_beam() const;
			void refresh_cache();
			// The transition max-cache is valid while customers are fixed; callers
			// refresh it once before a block of parallel samples.
			void cache_max();
			void dump_tree_stats(const char *tag);
			void dump_timing() const;
		protected:
			int _n, _mn, _m, _v, _k, _K, _wn;
			bool _fixed;
			long long _clamp_hits, _clamp_total;
			long long _beam_total = 0, _beam_positions = 0;
			double _a, _b;
			std::shared_ptr<vhdp> _pos;
			std::shared_ptr<std::vector<std::shared_ptr<hpyp> > > _word;
			// [corpus sentence index][position] -> emission depth used for removal.
			// Grown on demand and not serialized.
			std::vector<std::vector<int> > _ed;
			std::shared_ptr<std::vector<std::shared_ptr<vpyp> > > _letter;
			std::mutex _mutex;
			// Separate from _mutex because init() holds _mutex while calling _resize().
			std::mutex _grow;
			int _draw_emission_n(sentence& s, int i, int k);
			context* _emission_ctx(sentence& s, int i, int k) const;
			double _emission_word_lp(sentence& s, int i, int k);
			void _set_ed(int idx, int i, int d);
			int _get_ed(int idx, int i) const;
			void _draw_order(sentence& s);
			void _draw_order(vlattice& l);
			int _draw_order(sentence& s, int i);
			void _slice(vlattice& l);
			int _build_lattice(vlattice& l, bool best = false);
			double _ln_slice_density(double u, double ln_pr_tr) const;
			// Subclasses override the emission terms; transition sampling, slicing,
			// lattice construction, and FFBS remain shared.
			virtual double _emission_lp(vlattice& l, int i, int k);
			void _forward(vlattice& l, double u, int i, int k, int m, int n, std::vector<vt>& a);
			void _forward(context *c, vlattice& l, double u, double ln_pr_em, double ln_pr_tr, int i, int k, int j, int m, int n, vt *b, vt& a, std::vector<int>& prefix);
			int _backward(vlattice& l, double u, int i, int k, int m, int n, std::vector<vt>& a);
			double _backward(context *c, vlattice& l, double u, double ln_pr_tr, int i, int k, int m, int n, vt *b, bool& cutoff);
			double _marginalize(vt& node);
			void _scale(vt& node, double z);
			void _resize();
			// The growth hook: subclasses extend this, not _resize(), because the
			// lattice grows the model with _grow already held.
			virtual void _resize_locked();
			void _shrink();
			sentence _sample(sentence& s, bool best);
	};
}

#endif
