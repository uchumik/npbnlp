#ifndef NPBNLP_VHDP_H
#define NPBNLP_VHDP_H

#include "lm.h"
#include "sentence.h"
#include "vhdp_context.h"
#include <memory>
#include <mutex>
#include <vector>

namespace npbnlp {
	class vhdp : public lm {
		public:
			vhdp();
			vhdp(int n, double a=1, double b=1);
			virtual ~vhdp();
			double pr(int k, const context *h);
			double lp(int k, const context *h);
			double pr(word& w, const context *h);
			double pr(chunk& c, const context *h);
			double lp(word& w, const context *h);
			double lp(chunk& c, const context *h);
			bool add(int k, context *h);
			bool remove(int k, context *h);
			void add(word& w, context *h);
			void remove(word& w, context *h);
			void add(chunk& c, context *h);
			void remove(chunk& c, context *h);
			context* h() const;
			context* find(sentence& s, int i) const;
			context* make(sentence& s, int i);
			context* find(word& w, int i) const;
			context* find(chunk& c, int i) const;
			context* find(nsentence& s, int i) const;
			context* make(word& w, int i);
			context* make(chunk& c, int i);
			context* make(nsentence& s, int i);
			int n() const;
			double alpha(int n) const;
			double discount(int n) const;
			double strength(int n) const;
			int draw_n(sentence& s, int i);
			int draw_n(word& w, int i);
			int draw_n(chunk& c, int i);
			int draw_n(nsentence& s, int i);
			int draw_k(context *h);
			void estimate(int iter);
			void save(const char *file);
			void load(const char *file);
			void save(FILE *fp);
			void load(FILE *fp);
			void set_alpha(int n, double v);
			double lp_order(int k, const context *c, double& ln_pass, double& ln_pr);
			context* find_exist(sentence& s, int i, int order) const;
			context* find(sentence& s, int i, int order) const;
			context* make(sentence& s, int i, int order);
			void cache_max(int k_max);
			double max_lp(int k, int order) const;
			void clear_cache();
		private:
			int _n;
			double _a, _b;
			double _c, _d;
			std::shared_ptr<vhdp_context> _h;
			std::shared_ptr<std::vector<double> > _alpha;
			std::shared_ptr<std::vector<std::vector<double> > > _max;
			int _cache_kmax = 0;
			std::mutex _mutex;
			double _gamma(int k, const context *c);
			double _gamma_base(int k, const context *c);
			double _pr_pass(int k, const context *c);
			double _beta_inf(int k, const context *c);
			double _pr_base() const;
			double _cache_max(int order, context *c, int k);
			void _estimate_gamma();
	};
}

#endif
