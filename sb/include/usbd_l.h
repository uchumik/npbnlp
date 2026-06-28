#ifndef NPBNLP_USBD_LETTER_H
#define NPBNLP_USBD_LETTER_H

#include"usbd.h"
#include"dalm.h"
#include"vtable.h"

namespace npbnlp {
	class usbd_l : public usbd {
		public:
			usbd_l();
			usbd_l(int n, smoothing s);
			usbd_l(const usbd_l& d);
			usbd_l& operator=(const usbd_l& d);
			virtual ~usbd_l();
			virtual void init(cio& f, int iter);
			virtual void pretrain(io& f, int iter);
			virtual void set_prior(double g, double c, double p);
			virtual void set_general_prior(double p);
			virtual void set_cr_prior(double p);
			virtual void set_punc_prior(double p);

			virtual void set_punc(unsigned int c);
			virtual void set_hyper_general(double a, double b);
			virtual void set_hyper_cr(double c, double d);
			virtual void set_hyper_punc(double e, double f);
			virtual void add(io& f, std::vector<int>& head);
			virtual void remove(io& f, std::vector<int>& head);
			virtual void estimate_lm_hyper(int iter);
			virtual void estimate_prior(cio& c, std::vector<std::vector<int> >& boundaries);
			virtual void enable_duration(double a0, double b0, double alpha, double beta);
			virtual void estimate_duration(std::vector<std::vector<int> >& boundaries);
			virtual void eval(cio& c, cio& t, std::vector<double>& score);
			virtual void save(const char *file);
			virtual void load(const char *file);
			virtual void sample(io& f, std::vector<int>& b);
			virtual void parse(io& f, std::vector<int>& b);
		protected:
			int _n;
			std::shared_ptr<dalm> _lm;
			double _A;
			double _B;
			double _C;
			double _D;
			double _E;
			double _F;
			std::vector<unsigned int> _punc;
			double _general_prior;
			double _cr_prior;
			double _pnc_prior;
			// explicit negative-binomial sentence-length (duration) prior.
			// model: (L-1) ~ NB(_dur_r, _dur_p) so P(L)=C(L-2+r,L-1) p^r (1-p)^{L-1}
			// (Gamma-Poisson form, valid for real r; r=1 -> geometric = old _general_prior).
			// replaces the per-position general boundary prior when _use_dur.
			bool _use_dur;
			double _dur_r;      // NB dispersion (real, sampled via Gamma+CRT)
			double _dur_p;      // NB success prob (sampled via Beta)
			double _dur_a0, _dur_b0;       // Gamma(a0, scale=1/b0) prior on _dur_r
			double _dur_alpha, _dur_beta;  // Beta(alpha,beta) prior on _dur_p
			double _dur_logp(int L) const; // log P(segment length L) under the NB duration
			int _crt(int n, double r) const; // Chinese-restaurant-table aux draw for the r-update
			void _sample(io& f, std::vector<int>& b);
			void _parse(io& f, std::vector<int>& b);

			void _estimate_gen_cr_prior(cio& corpus, std::vector<std::vector<int> >& boundaries);
			void _estimate_punc_prior(cio& corpus, std::vector<std::vector<int> >& boundaries);
			void _eos(io& d, std::vector<std::vector<double> >& eos);
			void _bos(io& d, std::vector<std::vector<double> >& bos);
			void _cumurative(io& d, std::vector<double>& cum);

	};
}

#endif
