#include <cstdlib>
#include "niw.h"
#include "dense.h"
#include "generator.h"

#include <cmath>
#include <random>

namespace npbnlp {

	niw::niw()
		: _d(0), _kappa0(0.0), _nu0(0.0), _sample_mode(false) {}

	niw::niw(int d)
		: _d(d), _kappa0(0.0), _nu0(0.0), _sample_mode(false) {
		if (d < 1) {
			throw "niw: invalid dimension";
		}
	}

	niw::~niw() {}

	void niw::_check_x(const fvector& x) const {
		if (x.size() != _d) {
			throw "niw: dimension mismatch";
		}
	}

	void niw::_check_k(int k) const {
		if (k < 0 || k >= static_cast<int>(_n.size())) {
			throw "niw: state out of range";
		}
	}

	void niw::set_prior(const std::vector<double>& mu0, double kappa0,
	                   double nu0, const std::vector<double>& lambda0) {
		if (static_cast<int>(mu0.size()) != _d ||
			static_cast<int>(lambda0.size()) != _d * _d ||
			kappa0 <= 0.0 || nu0 <= _d + 1.0) {
			throw "niw: invalid prior";
		}
		std::vector<double> cholesky;
		if (!dense::chol(lambda0, _d, cholesky)) {
			throw "niw: Lambda_0 is not positive definite";
		}
		_kappa0 = kappa0;
		_nu0 = nu0;
		_mu0 = mu0;
		_lambda0 = lambda0;
		for (size_t i = 0; i < _dirty.size(); ++i) {
			_dirty[i] = true;
		}
	}

	void niw::set_prior(const std::vector<double>& mu0, double kappa0,
	                   double nu0, double lambda) {
		if (lambda <= 0.0) {
			throw "niw: invalid prior";
		}
		std::vector<double> lambda0(_d * _d, 0.0);
		for (int i = 0; i < _d; ++i) {
			lambda0[i * _d + i] = lambda;
		}
		set_prior(mu0, kappa0, nu0, lambda0);
	}

	void niw::init_prior_from_data(gio& f, double kappa0) {
		if (f.dim() != _d || f.dim() == 0) {
			throw "niw: invalid training data dimension";
		}
		std::vector<double> mean(_d, 0.0);
		std::vector<double> variance(_d, 0.0);
		long long n = 0;
		for (auto& sentence : f.seq) {
			for (auto& x : sentence.v) {
				++n;
				for (int i = 0; i < _d; ++i) {
					mean[i] += x[i];
				}
			}
		}
		if (n == 0) {
			throw "niw: empty training data";
		}
		for (int i = 0; i < _d; ++i) {
			mean[i] /= n;
		}
		for (auto& sentence : f.seq) {
			for (auto& x : sentence.v) {
				for (int i = 0; i < _d; ++i) {
					double z = x[i] - mean[i];
					variance[i] += z * z;
				}
			}
		}
		double average_variance = 0.0;
		for (double x : variance) {
			average_variance += x / n;
		}
		average_variance /= _d;
		double nu0 = _d + 2.0;
		set_prior(mean, kappa0, nu0, average_variance * (nu0 - _d - 1.0));
	}

	void niw::resize(int k) {
		if (k < 0) {
			throw "niw: invalid state count";
		}
		while (static_cast<int>(_n.size()) < k) {
			_n.push_back(0);
			_s.push_back(std::vector<double>(_d, 0.0));
			_ss.push_back(std::vector<double>(_d * _d, 0.0));
			_dirty.push_back(true);
			_mu_n.push_back(std::vector<double>());
			_chol.push_back(std::vector<double>());
			_logdet.push_back(0.0);
			_kappa_n.push_back(0.0);
			_nu_n.push_back(0.0);
			_mu_k.push_back(std::vector<double>());
			_sigma_k.push_back(std::vector<double>());
		}
	}

	void niw::add(int k, const fvector& x) {
		// Maintain n_k, s_k = sum x, and S_k = sum x x^T.
		std::lock_guard<std::mutex> guard(_mutex);
		_check_k(k);
		_check_x(x);
		++_n[k];
		for (int i = 0; i < _d; ++i) {
			_s[k][i] += x[i];
		}
		dense::syr(_ss[k], _d, x.v, 1.0);
		_dirty[k] = true;
	}

	void niw::remove(int k, const fvector& x) {
		// Maintain n_k, s_k = sum x, and S_k = sum x x^T.
		std::lock_guard<std::mutex> guard(_mutex);
		_check_k(k);
		_check_x(x);
		if (_n[k] <= 0) {
			throw "niw: remove from empty state";
		}
		--_n[k];
		for (int i = 0; i < _d; ++i) {
			_s[k][i] -= x[i];
		}
		dense::syr(_ss[k], _d, x.v, -1.0);
		_dirty[k] = true;
	}

	void niw::_refresh(int k) {
		// xbar = s_k/n_k and C_k = S_k - n_k xbar xbar^T; n_k=0 leaves the prior unchanged.
		std::vector<double> lambda = _lambda0;
		std::vector<double> mean(_d, 0.0);
		double kappa = _kappa0 + _n[k];
		double nu = _nu0 + _n[k];
		// Posterior: kappa_n=kappa_0+n, nu_n=nu_0+n,
		// mu_n=(kappa_0 mu_0+n xbar)/kappa_n,
		// Lambda_n=Lambda_0+C_k+(kappa_0 n/kappa_n)(xbar-mu_0)(xbar-mu_0)^T.
		if (kappa <= 0.0 || _nu0 <= _d + 1.0) {
			throw "niw: prior is not initialized";
		}
		for (int i = 0; i < _d; ++i) {
			mean[i] = (_kappa0 * _mu0[i] + _s[k][i]) / kappa;
		}
		if (_n[k] > 0) {
			std::vector<double> bar(_d, 0.0);
			std::vector<double> diff(_d, 0.0);
			for (int i = 0; i < _d; ++i) {
				bar[i] = _s[k][i] / _n[k];
				diff[i] = bar[i] - _mu0[i];
			}
			for (int i = 0; i < _d; ++i) {
				for (int j = 0; j < _d; ++j) {
					lambda[i * _d + j] += _ss[k][i * _d + j]
						- _n[k] * bar[i] * bar[j]
						+ _kappa0 * _n[k] / kappa * diff[i] * diff[j];
				}
			}
		}
		if (getenv("NIW_DIAG")) {
			for (int i = 0; i < _d; ++i)
				for (int j = 0; j < _d; ++j)
					if (i != j) lambda[i * _d + j] = 0.0;
		}
		std::vector<double> psi = lambda;
		double scale = (kappa + 1.0) / (kappa * (nu - _d + 1.0));
		for (double& value : psi) {
			value *= scale;
		}
		// Recompute Lambda_n from sufficient statistics after removals; rank-one
		// downdates can accumulate rounding error and lose positive definiteness.
		std::vector<double> target = _sample_mode ? lambda : psi;
		std::vector<double> cholesky;
		if (!dense::chol(target, _d, cholesky)) {
			double trace = 0.0;
			for (int i = 0; i < _d; ++i) {
				trace += lambda[i * _d + i];
			}
			double epsilon = 1e-9 * trace / _d;
			for (int i = 0; i < _d; ++i) {
				target[i * _d + i] += epsilon;
			}
			if (!dense::chol(target, _d, cholesky)) {
				throw "niw: Lambda_n is not positive definite";
			}
		}
		_mu_n[k] = mean;
		_chol[k] = cholesky;
		_logdet[k] = dense::logdet(cholesky, _d);
		_kappa_n[k] = kappa;
		_nu_n[k] = nu;
		_dirty[k] = false;
	}

	double niw::lp(int k, const fvector& x) {
		std::lock_guard<std::mutex> guard(_mutex);
		_check_k(k);
		_check_x(x);
		bool sampled = _sample_mode && !_mu_k[k].empty();
		if (!sampled && _dirty[k]) {
			_refresh(k);
		}
		// The marginal predictive is t_{nu_n-d+1}(mu_n,
		// Lambda_n(kappa_n+1)/(kappa_n(nu_n-d+1))).  Cholesky gives log|Psi|
		// and the quadratic form without explicitly forming an inverse or determinant.
		std::vector<double> difference(_d, 0.0);
		for (int i = 0; i < _d; ++i) {
			difference[i] = x[i] - (sampled ? _mu_k[k][i] : _mu_n[k][i]);
		}
		if (sampled) {
			return -0.5 * (_d * std::log(2.0 * M_PI)
				+ _logdet[k] + dense::quad(_chol[k], _d, difference));
		}
		double degrees = _nu_n[k] - _d + 1.0;
		double quadratic = dense::quad(_chol[k], _d, difference);
		return std::lgamma((degrees + _d) / 2.0)
			- std::lgamma(degrees / 2.0)
			- _d / 2.0 * std::log(degrees * M_PI)
			- 0.5 * _logdet[k]
			- (degrees + _d) / 2.0 * std::log1p(quadratic / degrees);
	}

	void niw::estimate() {
		std::lock_guard<std::mutex> guard(_mutex);
		std::shared_ptr<generator> gen = generator::create();
		for (int k = 0; k < static_cast<int>(_n.size()); ++k) {
			if (_dirty[k]) {
				_refresh(k);
			}
			std::vector<double> lambda = _lambda0;
			double kappa = _kappa_n[k];
			double nu = _nu_n[k];
			if (_n[k] > 0) {
				std::vector<double> bar(_d, 0.0);
				std::vector<double> diff(_d, 0.0);
				for (int i = 0; i < _d; ++i) {
					bar[i] = _s[k][i] / _n[k];
					diff[i] = bar[i] - _mu0[i];
				}
				for (int i = 0; i < _d; ++i) {
					for (int j = 0; j < _d; ++j) {
						lambda[i * _d + j] += _ss[k][i * _d + j]
							- _n[k] * bar[i] * bar[j]
							+ _kappa0 * _n[k] / kappa * diff[i] * diff[j];
					}
				}
			}
			std::vector<double> lower;
			if (!dense::chol(lambda, _d, lower)) {
				throw "niw: Lambda_n is not positive definite";
			}

			std::vector<double> bartlett(_d * _d, 0.0);
			// With Lambda_n=L L^T, Bartlett has A_ii=sqrt(chi^2_{nu_n-i+1}),
			// A_ij~N(0,1) for j<i;
			// solving X A^T=L gives Sigma=X X^T ~ IW(Lambda_n,nu_n).
			for (int i = 0; i < _d; ++i) {
				for (int j = 0; j <= i; ++j) {
					if (i == j) {
						std::gamma_distribution<double> chi_square((nu - i) / 2.0, 2.0);
						bartlett[i * _d + j] = std::sqrt(chi_square((*gen)()));
					} else {
						std::normal_distribution<double> normal(0.0, 1.0);
						bartlett[i * _d + j] = normal((*gen)());
					}
				}
			}

			std::vector<double> factor;
			dense::solve_rtri(lower, bartlett, _d, factor);
			dense::xxt(factor, _d, _sigma_k[k]);
			std::vector<double> sigma_lower;
			if (!dense::chol(_sigma_k[k], _d, sigma_lower)) {
				throw "niw: sampled covariance is not positive definite";
			}
			_mu_k[k].assign(_d, 0.0);
			std::normal_distribution<double> normal(0.0, 1.0);
			for (int i = 0; i < _d; ++i) {
				double z = 0.0;
				for (int j = 0; j <= i; ++j) {
					z += sigma_lower[i * _d + j] * normal((*gen)());
				}
				_mu_k[k][i] = _mu_n[k][i] + z / std::sqrt(kappa);
			}
			_chol[k] = sigma_lower;
			_logdet[k] = dense::logdet(sigma_lower, _d);
		}
	}

	void niw::set_sample_mode(bool f) {
		std::lock_guard<std::mutex> guard(_mutex);
		_sample_mode = f;
		for (size_t i = 0; i < _dirty.size(); ++i) {
			_dirty[i] = true;
		}
	}

	int niw::dim() const {
		return _d;
	}

	int niw::count(int k) const {
		_check_k(k);
		return _n[k];
	}

	void niw::posterior_sigma_mean(int k, std::vector<double>& out) {
		std::lock_guard<std::mutex> guard(_mutex);
		_check_k(k);
		if (_dirty[k]) {
			_refresh(k);
		}
		out = _lambda0;
		double kappa = _kappa_n[k];
		double nu = _nu_n[k];
		if (_n[k] > 0) {
			std::vector<double> bar(_d, 0.0);
			std::vector<double> diff(_d, 0.0);
			for (int i = 0; i < _d; ++i) {
				bar[i] = _s[k][i] / _n[k];
				diff[i] = bar[i] - _mu0[i];
			}
			for (int i = 0; i < _d; ++i) {
				for (int j = 0; j < _d; ++j) {
					out[i * _d + j] += _ss[k][i * _d + j]
						- _n[k] * bar[i] * bar[j]
						+ _kappa0 * _n[k] / kappa * diff[i] * diff[j];
				}
			}
		}
		double denominator = nu - _d - 1.0;
		for (double& value : out) {
			value /= denominator;
		}
	}

	void niw::sufficient_statistics(int k, std::vector<double>& s,
	                                std::vector<double>& ss) const {
		std::lock_guard<std::mutex> guard(_mutex);
		_check_k(k);
		s = _s[k];
		ss = _ss[k];
	}

	void niw::sampled_sigma(int k, std::vector<double>& out) const {
		std::lock_guard<std::mutex> guard(_mutex);
		_check_k(k);
		if (_sigma_k[k].empty()) {
			throw "niw: no sampled covariance";
		}
		out = _sigma_k[k];
	}

	void niw::dump(FILE* fp) const {
		for (int k = 0; k < static_cast<int>(_n.size()); ++k) {
			fprintf(fp, "state %d n %d\n", k, _n[k]);
			for (double x : _mu_k[k]) {
				fprintf(fp, " %.17g", x);
			}
			fprintf(fp, "\n");
			for (double x : _sigma_k[k]) {
				fprintf(fp, " %.17g", x);
			}
			fprintf(fp, "\n");
		}
	}

	static void write_data(FILE* file, const void* data, size_t size, size_t count) {
		if (fwrite(data, size, count, file) != count) {
			throw "niw: write failed";
		}
	}

	static void read_data(FILE* file, void* data, size_t size, size_t count) {
		if (fread(data, size, count, file) != count) {
			throw "niw: read failed";
		}
	}

	void niw::save(const char* file) {
		FILE* fp = fopen(file, "wb");
		if (!fp) {
			throw "niw: failed to open save file";
		}
		save(fp);
		fclose(fp);
	}

	void niw::save(FILE* fp) {
		write_data(fp, &_d, sizeof(int), 1);
		write_data(fp, &_kappa0, sizeof(double), 1);
		write_data(fp, &_nu0, sizeof(double), 1);
		write_data(fp, _mu0.data(), sizeof(double), _d);
		write_data(fp, _lambda0.data(), sizeof(double), _d * _d);
		int states = _n.size();
		write_data(fp, &states, sizeof(int), 1);
		for (int k = 0; k < states; ++k) {
			write_data(fp, &_n[k], sizeof(int), 1);
			write_data(fp, _s[k].data(), sizeof(double), _d);
			write_data(fp, _ss[k].data(), sizeof(double), _d * _d);
		}
		int mode = _sample_mode ? 1 : 0;
		write_data(fp, &mode, sizeof(int), 1);
		if (mode) {
			for (int k = 0; k < states; ++k) {
				write_data(fp, _mu_k[k].data(), sizeof(double), _d);
				write_data(fp, _sigma_k[k].data(), sizeof(double), _d * _d);
			}
		}
	}

	void niw::load(const char* file) {
		FILE* fp = fopen(file, "rb");
		if (!fp) {
			throw "niw: failed to open load file";
		}
		load(fp);
		fclose(fp);
	}

	void niw::load(FILE* fp) {
		read_data(fp, &_d, sizeof(int), 1);
		read_data(fp, &_kappa0, sizeof(double), 1);
		read_data(fp, &_nu0, sizeof(double), 1);
		if (_d < 1 || _nu0 <= _d + 1.0) {
			throw "niw: invalid loaded prior";
		}
		_mu0.resize(_d);
		_lambda0.resize(_d * _d);
		read_data(fp, _mu0.data(), sizeof(double), _d);
		read_data(fp, _lambda0.data(), sizeof(double), _d * _d);
		int states = 0;
		read_data(fp, &states, sizeof(int), 1);
		_n.clear();
		_s.clear();
		_ss.clear();
		_dirty.clear();
		_mu_n.clear();
		_chol.clear();
		_logdet.clear();
		_kappa_n.clear();
		_nu_n.clear();
		_mu_k.clear();
		_sigma_k.clear();
		resize(states);
		for (int k = 0; k < states; ++k) {
			read_data(fp, &_n[k], sizeof(int), 1);
			read_data(fp, _s[k].data(), sizeof(double), _d);
			read_data(fp, _ss[k].data(), sizeof(double), _d * _d);
		}
		int mode = 0;
		read_data(fp, &mode, sizeof(int), 1);
		_sample_mode = mode != 0;
		if (_sample_mode) {
			for (int k = 0; k < states; ++k) {
				_mu_k[k].resize(_d);
				_sigma_k[k].resize(_d * _d);
				read_data(fp, _mu_k[k].data(), sizeof(double), _d);
				read_data(fp, _sigma_k[k].data(), sizeof(double), _d * _d);
				// estimate() derives the Cholesky factor of the sampled Sigma and
				// its log-determinant, and lp() needs both.  They are not written
				// out -- they are a function of Sigma_k -- so rebuild them here,
				// or the density reads an empty _chol.
				std::vector<double> lower;
				if (!dense::chol(_sigma_k[k], _d, lower)) {
					throw "niw: loaded covariance is not positive definite";
				}
				_chol[k] = lower;
				_logdet[k] = dense::logdet(lower, _d);
			}
		}
		for (size_t k = 0; k < _dirty.size(); ++k) {
			_dirty[k] = true;
		}
	}

}
