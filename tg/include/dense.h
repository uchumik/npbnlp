#ifndef NPBNLP_DENSE_H
#define NPBNLP_DENSE_H
#include <vector>
namespace npbnlp { namespace dense {
	bool chol(const std::vector<double>& a, int d, std::vector<double>& l);
	void solve_lower(const std::vector<double>& l, int d, const std::vector<double>& b, std::vector<double>& x);
	void solve_upper(const std::vector<double>& u, int d, const std::vector<double>& b, std::vector<double>& x);
	void solve_rtri(const std::vector<double>& l, const std::vector<double>& a, int d, std::vector<double>& x);
	double logdet(const std::vector<double>& l, int d);
	double quad(const std::vector<double>& l, int d, const std::vector<double>& v);
	void syr(std::vector<double>& a, int d, const std::vector<double>& x, double alpha);
	void xxt(const std::vector<double>& x, int d, std::vector<double>& c);
}}
#endif
