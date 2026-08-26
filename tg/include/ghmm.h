#ifndef NPBNLP_GHMM_H
#define NPBNLP_GHMM_H

#include "vhmm.h"
#include "niw.h"
#include "gseq.h"
#include <memory>
#include <vector>

namespace npbnlp {
	// The continuous-observation variable-order infinite HMM: vhmm's transition
	// model with the word HPYP / character VPYP emission replaced by a
	// Normal-Inverse-Wishart Gaussian per state.
	//
	// The transition side is inherited verbatim -- order sampling, slice, lattice,
	// FFBS, the block-parallel sweep.  Only the emission is overridden, so the
	// four bugs found while porting vhmm (ln_pr carried through the order walk,
	// the root base measure, the lattice's boundary state, and looking contexts up
	// through vhdp_context) are inherited already fixed.
	//
	// vhmm indexes observations through vlattice, which wraps a sentence, so each
	// gsentence gets a shadow sentence of the same length whose word ids are
	// indices into a flat observation table.  _emission_lp reads the vector back
	// out by that id.  This keeps the shadow immutable during sampling, which the
	// parallel block needs, and avoids templating the whole of vhmm on the
	// sequence type.
	class ghmm : public vhmm {
		public:
			ghmm();
			ghmm(int n, int min_n, int k, int d);
			virtual ~ghmm();

			// Register a corpus.  Builds the shadow sentences and the flat
			// observation table; must be called before init/sample/add/remove.
			void set_corpus(gio& f);
			void init_prior(gio& f);
			void set_prior(const std::vector<double>& mu0, double kappa0, double nu0, double lambda);
			void set_sample_mode(bool f);
			void seed_sample();

			// Sequence-indexed API.  The index is the position in the corpus given
			// to set_corpus.
			void init(int seq);
			// Assign states without seating anyone, for inference on a loaded
			// model -- init() adds customers and would pollute it.
			void inference_init(int seq);
			void sample(int seq);
			void parse(int seq);
			void add(int seq);
			void remove(int seq);
			void store(int seq, gsentence& g) const; // write states/orders back

			virtual void estimate(int iter);
			virtual void save(const char *file);
			virtual void load(const char *file);
			int dim() const;
			int seq_count() const;
			void dump(FILE *fp) const;
			// Posterior mean of mu and Sigma per state, off-diagonals included.
			void dump_posterior(FILE *fp) const;

		protected:
			virtual double _emission_lp(vlattice& l, int i, int k);
			virtual void _resize();

			int _d;
			std::shared_ptr<niw> _obs;
			std::vector<fvector> _x;      // flat observation table, indexed by word id
			std::vector<sentence> _shadow;
			void _build_shadow(gio& f);
	};
}

#endif
