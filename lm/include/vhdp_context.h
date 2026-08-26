#ifndef NPBNLP_VHDP_CONTEXT_H
#define NPBNLP_VHDP_CONTEXT_H

#include "context.h"
#include "rd.h"
#include "beta.h"
#include <cfloat>

namespace npbnlp {
	class vhdp_context : public context {
		public:
			vhdp_context();
			vhdp_context(int k, context *h);
			virtual ~vhdp_context();
			bool add(int k, lm *m);
			bool remove(int k);
			context* find(int k) const;
			context* make(int k);
			int stick_stop(int k) const;
			int stick_size() const; // number of distinct states tracked here
			int stick_n(int k) const;
			double cache_pr(int k) const;
			double cache_pass(int k) const;
			double cache_gamma(int k) const;
			double cache_beta_inf(int k) const;
			void set_cache_pr(int k, double p);
			void set_cache_pass(int k, double p);
			void set_cache_gamma(int k, double p);
			void set_cache_beta_inf(int k, double p);
			void clear_cache();
			// Upper bound on lp(k, .) over this node and every context below it,
			// in logs.  The forward recursion only ever evaluates transitions at
			// this node or deeper, so a bound below the slice threshold lets it
			// drop the whole subtree.  (The prototype has the same idea but
			// stores a linear probability and compares it against a log
			// threshold, so its prune never fires.)
			double subtree_max(int k) const;
			void set_subtree_max(int k, double v);
			void estimate_a(std::vector<double>& a, std::vector<double>& b, lm *m);
			const children& child() const;
			void save(FILE *fp);
			void load(FILE *fp);
			// context::cleanup/estimate_d/estimate_t are not part of the vhdp
			// execution path; vhdp uses clear_cache/estimate_a and these own
			// save/load methods instead.  Do not call the non-virtual base
			// implementations here: they recurse through context::_child.
		private:
			struct stick { int stop; int pass; };
			std::shared_ptr<std::vector<stick> > _cdp;
			std::shared_ptr<children> _child;
			std::shared_ptr<std::vector<double> > _pr, _pass, _gamma, _beta_inf, _smax;
			bool _crp_add(int k, lm *m);
			bool _crp_remove(int k);
			void _ensure(std::vector<stick>& v, int k);
		};
}

#endif
