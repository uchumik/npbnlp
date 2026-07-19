#ifndef NPBNLP_HPYP_H
#define NPBNLP_HPYP_H

#include"cache.h"
#include"nsentence.h"
#include"sentence.h"
#include"chunk.h"
#include"word.h"
#include"lm.h"
#include"context.h"
#include<memory>
#include<mutex>
#include<functional>
namespace npbnlp {
	class hpyp : public lm {
		public:
			hpyp();
			hpyp(int n, double a=1, double b=1);
			hpyp(const hpyp& lm);
			hpyp(hpyp&& lm);
			hpyp& operator=(const hpyp& lm);
			hpyp& operator=(const hpyp&& lm) noexcept;
			virtual ~hpyp();
			double pr(chunk& c, const context *h);
			double pr(word& w, const context *h);
			double pr(int k, const context *h);
			double lp(chunk& c, const context *h);
			double lp(word& w, const context *h);
			double lp(int k, const context *h);
			// exact helpers for cached slice-time emission computation.
			// lp_root_base: PY interpolation at the root context using a
			//   precomputed base (= _lpb value) instead of recursing.
			// wlp: word log-prob given explicit preceding-word ids (most
			//   recent first, 0 = BOS), replicating _lpb's context walk.
			double lp_root_base(chunk& c, double base);
			double lp_root_base(word& w, double base);
			double wlp(word& w, const int *prev, int np);
			double wlp(int k, const int *prev, int np);
			double discount(int n) const;
			double strength(int n) const;
			double alpha(int n) const;
			context* h() const;
			context* find(nsentence& s, int i)const;
			context* find(sentence& s, int i)const;
			context* find(chunk& c, int i) const;
			context* find(word& s, int i) const;
			context* make(nsentence& s, int i);
			context* make(sentence& s, int i);
			context* make(chunk& c, int i);
			context* make(word& s, int i);
			context* find(nsentence& s, int i, int n) const;
			context* find(sentence& s, int i, int n) const;
			context* find(chunk& c, int i, int n) const;
			context* find(word& w, int i, int n) const;
			context* make(nsentence& s, int i, int n);
			context* make(sentence& s, int i, int n);
			context* make(chunk& c, int i, int n);
			context* make(word& w, int i, int n);
			int draw_n(word& w, int i);
			int draw_n(chunk& c, int i);
			int draw_n(sentence& s, int i);
			int draw_n(nsentence& s, int i);
			int draw_k(context *h);
			int n() const;
			int v() const;
			// Diagnostic only: number of tokens currently held in the word
			// base corpus (_bc), i.e. the witnesses backing the base measure.
			// Used to detect base measures starved by deeper n-gram contexts.
			long long base_customers() const;
			void add(chunk& c, context *h);
			void add(word& w, context *h);
			bool add(int k, context *h);
			void remove(chunk& c, context *h);
			void remove(word& w, context *h);
			bool remove(int k, context *h);
			void set_base(lm *b);
			// optional chunk base-measure delegation: when set, _lpb(chunk&)
			// returns _cbase(c) instead of walking the word-LM base, and the
			// new-table seating into the base uses _cbase_add/_cbase_remove
			// instead of wrap::add_a/remove_a on the word LM.
			void set_cbase(std::function<double(chunk&)> f);
			void set_cbase_add(std::function<void(chunk&)> f);
			void set_cbase_remove(std::function<void(chunk&)> f);
			// true when a chunk base-measure delegate (e.g. the pos-seq base) is
			// installed; false means the chunk base falls back to the word LM.
			bool has_cbase() const { return (bool)_cbase; }
			void set_v(int v);
			void estimate(int iter);
			void poisson_correction(int n = 1000);
			bool valid() const;
			bool empty() const;
			void gibbs(int iter);
			void save(const char *file);
			void load(const char *file);
			void save(FILE *fp);
			void load(FILE *fp);
		protected:
			template<class T>
			using bcorpus = std::unordered_map<int, std::vector<T> >;
			using base_corpus = bcorpus<word>;
			using cbase_corpus = bcorpus<chunk>;
			int _n;
			double _a;
			double _b;
			lm *_base;
			std::function<double(chunk&)> _cbase;
			std::function<void(chunk&)> _cbase_add;
			std::function<void(chunk&)> _cbase_remove;
			int _v;
			std::shared_ptr<context> _h;
			std::shared_ptr<std::vector<double> > _discount;
			std::shared_ptr<std::vector<double> > _strength;
			std::shared_ptr<base_corpus> _bc;
			std::shared_ptr<cbase_corpus> _cbc;
			std::shared_ptr<std::vector<double> > _poisson;
			std::shared_ptr<std::vector<double> > _lambda;
			int _f;
			std::shared_ptr<std::vector<double> > _length;
			double _prb(chunk& c) const;
			double _prb(word& w) const;
			double _lpb(chunk& c) const;
			double _lpb(word& w) const;
			double _correct(word& w) const;
			// base-corpus seating shims: hide the word/chunk overload and the
			// chunk-side _cbase_add/_cbase_remove delegation behind one name so
			// the templated corpus machinery below stays type-agnostic.
			// (defined in hpyp.cc where wrap:: is visible.)
			void _seat_base(word& w);
			void _seat_base(chunk& c);
			void _unseat_base(word& w);
			void _unseat_base(chunk& c);
			// unified add/remove/gibbs over either base corpus (_bc or _cbc).
			template<class T> void _bc_add(std::shared_ptr<bcorpus<T> >& bc, T& x);
			template<class T> void _bc_remove(std::shared_ptr<bcorpus<T> >& bc, T& x);
			template<class T> void _gibbs_impl(bcorpus<T>& bc, int iter);
			void _estimate_poisson();
			void _estimate_length(int n);
			void _sample(std::vector<unsigned int>& w);
			std::mutex _mutex;
			cache _cache;
	};
}
#endif
