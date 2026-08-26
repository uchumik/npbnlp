#ifndef NPBNLP_NIW_H
#define NPBNLP_NIW_H
#include "gseq.h"
#include <cstdio>
#include <mutex>
#include <vector>
namespace npbnlp {
	class niw {
	public:
		niw(); explicit niw(int d); virtual ~niw();
		void set_prior(const std::vector<double>& mu0, double kappa0, double nu0,
		               const std::vector<double>& lambda0);
		void set_prior(const std::vector<double>& mu0, double kappa0, double nu0, double lambda);
		void init_prior_from_data(gio& f);
		double lp(int k, const fvector& x);
		void add(int k, const fvector& x); void remove(int k, const fvector& x);
		void estimate(); void set_sample_mode(bool f); void resize(int k);
		int dim() const; int count(int k) const;
		void posterior_sigma_mean(int k, std::vector<double>& out);
		void sufficient_statistics(int k, std::vector<double>& s,
		                          std::vector<double>& ss) const;
		void sampled_sigma(int k, std::vector<double>& out) const;
		void dump(FILE *fp) const; void save(const char *file); void save(FILE *fp);
		void load(const char *file); void load(FILE *fp);
	private:
		int _d; double _kappa0, _nu0; std::vector<double> _mu0, _lambda0;
		std::vector<int> _n; std::vector<std::vector<double> > _s, _ss; std::vector<bool> _dirty;
		std::vector<std::vector<double> > _mu_n, _chol; std::vector<double> _logdet, _kappa_n, _nu_n;
		bool _sample_mode; std::vector<std::vector<double> > _mu_k, _sigma_k; mutable std::mutex _mutex;
		void _refresh(int k); void _check_x(const fvector& x) const; void _check_k(int k) const;
	};
}
#endif
