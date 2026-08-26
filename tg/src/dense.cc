#include "dense.h"
#include <cmath>

namespace npbnlp { namespace dense {
	// Cholesky factorization: A = L L^T.
	bool chol(const std::vector<double>& a, int d, std::vector<double>& l) {
		if ((int)a.size() != d*d) return false;
		l.assign(d*d, 0.0);
		for (int i=0; i<d; ++i) for (int j=0; j<=i; ++j) {
			double z = a[i*d+j];
			for (int k=0; k<j; ++k) z -= l[i*d+k]*l[j*d+k];
			if (i == j) { if (!(z > 0.0) || !std::isfinite(z)) return false; l[i*d+j]=std::sqrt(z); }
			else { if (l[j*d+j] == 0.0) return false; l[i*d+j]=z/l[j*d+j]; }
		}
		return true;
	}
	// Solve Lx=b by forward substitution.
	void solve_lower(const std::vector<double>& l, int d, const std::vector<double>& b, std::vector<double>& x) {
		x.assign(d, 0.0);
		for (int i=0; i<d; ++i) { double z=b[i]; for (int j=0;j<i;++j) z-=l[i*d+j]*x[j]; x[i]=z/l[i*d+i]; }
	}
	// Solve Ux=b by backward substitution.
	void solve_upper(const std::vector<double>& u, int d, const std::vector<double>& b, std::vector<double>& x) {
		x.assign(d, 0.0);
		for (int i=d-1; i>=0; --i) { double z=b[i]; for (int j=i+1;j<d;++j) z-=u[i*d+j]*x[j]; x[i]=z/u[i*d+i]; }
	}
	// Solve X L^T = A by solving L X^T = A^T row by row.
	void solve_rtri(const std::vector<double>& l, const std::vector<double>& a, int d, std::vector<double>& x) {
		x.assign(d*d, 0.0);
		for (int i=0;i<d;++i) { std::vector<double> b(d); for (int j=0;j<d;++j) b[j]=l[i*d+j]; std::vector<double> r; solve_lower(a,d,b,r); for (int j=0;j<d;++j) x[i*d+j]=r[j]; }
	}
	// For A = L L^T, log|A| = 2 sum_i log L_ii.
	double logdet(const std::vector<double>& l, int d) { double z=0; for(int i=0;i<d;++i) z+=2.0*std::log(l[i*d+i]); return z; }
	// v^T (L L^T)^-1 v = ||L^-1 v||^2.
	double quad(const std::vector<double>& l, int d, const std::vector<double>& v) { std::vector<double> x; solve_lower(l,d,v,x); double z=0; for(double q:x) z+=q*q; return z; }
	// Symmetric rank-one update: A += alpha x x^T.
	void syr(std::vector<double>& a, int d, const std::vector<double>& x, double alpha) { for(int i=0;i<d;++i) for(int j=0;j<d;++j) a[i*d+j]+=alpha*x[i]*x[j]; }
	void xxt(const std::vector<double>& x, int d, std::vector<double>& c) { c.assign(d*d,0.0); for(int i=0;i<d;++i) for(int j=0;j<d;++j) for(int k=0;k<d;++k) c[i*d+j]+=x[i*d+k]*x[j*d+k]; }
}}
