#include <cstdlib>
#include "vhdp.h"
#include "math.h"
#include "convinience.h"
#include <cfloat>
#include <cmath>
#include <iostream>
#include <random>

using namespace std;
using namespace npbnlp;

namespace npbnlp { double ORD_score[8]={0},ORD_lpk[8]={0}; long long ORD_n[8]={0},ORD_have[8]={0},ORD_stop[8]={0},ORD_pass[8]={0},ORD_hn[8]={0}; }
namespace npbnlp { long long VHDP_N_cmax_d[8]={0,0,0,0,0,0,0,0}; long long VHDP_N_cmax=0, VHDP_N_lp=0, VHDP_N_pr=0, VHDP_N_pr_hit=0, VHDP_N_gamma=0, VHDP_N_gamma_hit=0, VHDP_N_pass=0, VHDP_N_pass_hit=0, VHDP_N_binf=0, VHDP_N_binf_hit=0; }

#define VHDP_ALPHA 5.
// The transition base is the stick-breaking distribution over an unbounded state set.
// pr(k,NULL) is only the missing-context sentinel; it is not a state probability.
#define VHDP_BASE 1.
#define VHDP_ZERO 1e-300
#define VHDP_LZERO -1e300

vhdp::vhdp(): _n(1), _a(1), _b(1), _c(1), _d(1), _h(new vhdp_context), _alpha(new vector<double>(1, VHDP_ALPHA)), _max(new vector<vector<double> >) {
}

vhdp::vhdp(int n, double a, double b): _n(n), _a(a), _b(b), _c(1), _d(1), _h(new vhdp_context), _alpha(new vector<double>(n, VHDP_ALPHA)), _max(new vector<vector<double> >) {
	if (n < 1 || a <= 0 || b <= 0) throw "invalid vhdp parameters";
	_h->set((int)a, (int)b);
}

vhdp::~vhdp() {}

// The root base is uniform over the distinct states already present: H(k)=1/v.
double vhdp::_pr_base() const {
	const char *m=getenv("VHDP_BASE_MODE");
	if (m && m[0]=='p') return 1e-4;  // the prototype's _H_
	int v=_h->v();
	return v>0 ? 1./v : 1.;
}

double vhdp::alpha(int n) const {
	if (n < 0 || n >= _n) throw "invalid vhdp alpha index";
	return (*_alpha)[n];
}

void vhdp::set_alpha(int n, double v) {
	if (n < 0 || n >= _n || v <= 0) throw "invalid vhdp alpha";
	lock_guard<mutex> l(_mutex);
	(*_alpha)[n] = v;
	clear_cache();
}

int vhdp::n() const { return _n; }
double vhdp::discount(int) const { return 0.; }
double vhdp::strength(int n) const { return alpha(n); }
context* vhdp::h() const { return _h.get(); }

// Stick-breaking weights for the transition.  A context c gives state k the
// break gamma_k(c), and the predictive is the product of the sticks not taken:
//
//   p(k | c) = gamma_k(c) * prod_{j<k} (1 - gamma_j(c))
//
// which is what lets the state set stay unbounded -- there is no normalisation
// over a fixed K.  gamma is the CRP posterior of the break at c, backing off to
// the parent, and _beta_inf is the mass the parent leaves to states at or above
// k, floored so a deep context can never claim more than its parent allows.
double vhdp::_gamma_base(int k, const context *c) {
	vhdp_context *v = const_cast<vhdp_context*>(static_cast<const vhdp_context*>(c));
	++VHDP_N_gamma;
	double old = v->cache_gamma(k);
	if (old != -DBL_MAX) { ++VHDP_N_gamma_hit; return old; }
	double g = (1. + v->stick_stop(k))/(1. + alpha(0) + v->stick_n(k));
	v->set_cache_gamma(k, g);
	return g;
}

double vhdp::_beta_inf(int k, const context *c) {
	if (k < 0) return 0.;
	vhdp_context *v = const_cast<vhdp_context*>(static_cast<const vhdp_context*>(c));
	++VHDP_N_binf;
	double old = v->cache_beta_inf(k);
	if (old != -DBL_MAX) { ++VHDP_N_binf_hit; return old; }
	double p = _beta_inf(k-1, c) + pr(k, c);
	v->set_cache_beta_inf(k, p);
	return p;
}

double vhdp::_gamma(int k, const context *c) {
	if (c->n() == 0) return _gamma_base(k, c);
	vhdp_context *v = const_cast<vhdp_context*>(static_cast<const vhdp_context*>(c));
	++VHDP_N_gamma;
	double old = v->cache_gamma(k);
	if (old != -DBL_MAX) { ++VHDP_N_gamma_hit; return old; }
	const context *p = c->parent();
	double beta_k = pr(k, p);
	double beta_inf = max(1. - _beta_inf(k-1, p), beta_k);
	double an = alpha(c->n());
	double g = (an*beta_k + v->stick_stop(k))/(an*beta_inf + v->stick_n(k));
	v->set_cache_gamma(k, g);
	return g;
}

double vhdp::_pr_pass(int k, const context *c) {
	if (k < 0) return 1.;
	vhdp_context *v = const_cast<vhdp_context*>(static_cast<const vhdp_context*>(c));
	++VHDP_N_pass;
	double old = v->cache_pass(k);
	if (old != -DBL_MAX) { ++VHDP_N_pass_hit; return old; }
	double p = _pr_pass(k-1, c) * (1. - _gamma(k, c));
	v->set_cache_pass(k, p);
	return p;
}

double vhdp::pr(int k, const context *c) {
	if (!c) return _pr_base(); // top of the hierarchy: _crp_add's new-table term
	if (k < 0) return 0.;
	++VHDP_N_pr;
	vhdp_context *v = const_cast<vhdp_context*>(static_cast<const vhdp_context*>(c));
	double old = v->cache_pr(k);
	if (old != -DBL_MAX) { ++VHDP_N_pr_hit; return old; }
	double p = max(VHDP_ZERO, _pr_pass(k-1, c) * _gamma(k, c));
	v->set_cache_pr(k, p);
	return p;
}

double vhdp::lp(int k, const context *c) {
	++VHDP_N_lp;
	double p = pr(k, c);
	if (p < VHDP_ZERO) return VHDP_LZERO;
	return log(p);
}

double vhdp::pr(word&, const context*) { throw "unsupported in vhdp"; }
double vhdp::pr(chunk&, const context*) { throw "unsupported in vhdp"; }
double vhdp::lp(word&, const context*) { throw "unsupported in vhdp"; }
double vhdp::lp(chunk&, const context*) { throw "unsupported in vhdp"; }

bool vhdp::add(int k, context *c) {
	lock_guard<mutex> l(_mutex);
	bool r = false;
	while (c && (r = static_cast<vhdp_context*>(c)->add(k, this))) c = c->parent();
	return r;
}

bool vhdp::remove(int k, context *c) {
	lock_guard<mutex> l(_mutex);
	bool r = false;
	while (c && (r = static_cast<vhdp_context*>(c)->remove(k))) c = c->parent();
	return r;
}

void vhdp::add(word&, context*) { throw "unsupported in vhdp"; }
void vhdp::remove(word&, context*) { throw "unsupported in vhdp"; }
void vhdp::add(chunk&, context*) { throw "unsupported in vhdp"; }
void vhdp::remove(chunk&, context*) { throw "unsupported in vhdp"; }

context* vhdp::find(sentence& s, int i) const { return find(s, i, _n); }
context* vhdp::find(sentence& s, int i, int order) const {
	context *c = h();
	for (int m=1; m<order; ++m) { context *d=static_cast<vhdp_context*>(c)->find(s.wd(i-m).pos); if (!d) break; c=d; }
	return c;
}
context* vhdp::find_exist(sentence& s, int i, int order) const {
	context *c = h();
	for (int m=1; m<order; ++m) { context *d=static_cast<vhdp_context*>(c)->find(s.wd(i-m).pos); if (!d) break; c=d; }
	return c;
}
context* vhdp::make(sentence& s, int i) { return make(s, i, _n); }
context* vhdp::make(sentence& s, int i, int order) {
	context *c=h();
	for (int m=1; m<order; ++m) c=static_cast<vhdp_context*>(c)->make(s.wd(i-m).pos);
	return c;
}
context* vhdp::find(word&, int) const { throw "unsupported in vhdp"; }
context* vhdp::find(chunk&, int) const { throw "unsupported in vhdp"; }
context* vhdp::find(nsentence&, int) const { throw "unsupported in vhdp"; }
context* vhdp::make(word&, int) { throw "unsupported in vhdp"; }
context* vhdp::make(chunk&, int) { throw "unsupported in vhdp"; }
context* vhdp::make(nsentence&, int) { throw "unsupported in vhdp"; }

double vhdp::lp_order(int k, const context *c, double& ln_pass, double& ln_pr) {
	int s=(int)_a, p=(int)_b;
	// Order posterior, as in VPYLM: depth d is weighted by its stop probability
	// times the passes of the depths above it, times the transition probability
	// at that depth.  The walk stops at the deepest context that exists; deeper
	// orders keep scoring against it rather than a base measure, so a missing
	// context cannot look better than the real one above it.
	// ln_pr and ln_pass carry the accumulated transition probability across depths.
	// stop() and pass() include the Beta(a,b) prior in the stop/pass masses.
	if (c) { ln_pr=lp(k,c); s=c->stop(); p=c->pass(); }
	double r=log(s)-log(s+p)+ln_pr+ln_pass;
	ln_pass += log(p)-log(s+p);
	return r;
}

int vhdp::draw_n(sentence& s, int i) {
	int k=s.wd(i).pos; const context *c=h(); double lp=0., ln_pr=log(_pr_base()); vector<double> table;
	static bool probe = getenv("VHDP_ORDPROBE")!=NULL;
	for (int j=1; j<=_n && i-j>=-2; ++j) {
		bool have = (c!=NULL);
		double lpk = have ? this->lp(k, const_cast<context*>(c)) : 0.; // probe only
		table.push_back(lp_order(k,c,lp,ln_pr));
		if (probe && j-1<8) {
			ORD_score[j-1]+=table.back(); ORD_lpk[j-1]+=lpk; ORD_n[j-1]+=1; ORD_have[j-1]+=have?1:0;
			if (have) { ORD_stop[j-1]+=c->stop(); ORD_pass[j-1]+=c->pass(); ORD_hn[j-1]+=1; }
		}
		if(c) c=static_cast<const vhdp_context*>(c)->find(s.wd(i-j).pos);
	}
	return 1+rd::ln_draw(table);
}
int vhdp::draw_n(word&, int) { throw "unsupported in vhdp"; }
int vhdp::draw_n(chunk&, int) { throw "unsupported in vhdp"; }
int vhdp::draw_n(nsentence&, int) { throw "unsupported in vhdp"; }
int vhdp::draw_k(context *c) { return c->sample(this); }

void vhdp::_estimate_gamma() {
	int m=_h->c(), tables=static_cast<vhdp_context*>(_h.get())->stick_size();
	if (m <= 0) return;
	double value=alpha(0); beta_distribution be; gamma_distribution<> gd(1.,1.); shared_ptr<generator> g=generator::create();
	for (int e=0;e<20;++e) {
		double eta=be(1.+value,m), d=_c+tables-1., q=m*(_d-log(eta));
		double pi=d/(d+q);
		gamma_distribution<>::param_type par(_c+tables-(pi<uniform_real_distribution<double>(0.,1.)((*g)())?1:0), 1./(_d-log(eta)));
		gd.param(par); value=gd((*g)());
	}
	(*_alpha)[0]=value;
}

void vhdp::estimate(int iter) {
	gamma_distribution<> gd;
	for (int e=0;e<iter;++e) {
		_estimate_gamma(); vector<double> aa(_n,0.), bb(_n,0.); _h->estimate_a(aa,bb,this); shared_ptr<generator> g=generator::create();
		for(int i=1;i<_n;++i) { gd.param(typename gamma_distribution<>::param_type(_c+aa[i],1./(_d-bb[i]))); (*_alpha)[i]=gd((*g)()); }
		if (e==iter-1 && getenv("VHDP_AB")) {
			for(int i=0;i<_n;++i)
				fprintf(stderr,"[ab] NEW depth=%d a=%f b=%f shape=%f scale=%f alpha=%f\n",
					i, aa[i], bb[i], _c+aa[i], 1./(_d-bb[i]), (*_alpha)[i]);

		}
	}
	clear_cache();
}

double vhdp::_cache_max(int order, context *c, int k) {
	++VHDP_N_cmax; if(order<8) ++VHDP_N_cmax_d[order];
	// Store the node bound at its depth and propagate it through missing descendants.
	double ln_pr_tr=lp(k,c);
	(*_max)[k][order]=max((*_max)[k][order],ln_pr_tr);
	vhdp_context *v=static_cast<vhdp_context*>(c);
	double smax=ln_pr_tr;
	if(order<_n-1) {
		if ((int)v->child().size() < _cache_kmax+1)
			for(int m=order;m<_n;++m) (*_max)[k][m]=max((*_max)[k][m],ln_pr_tr);
		for(auto& x : v->child()) smax=max(smax,_cache_max(order+1,x.second.get(),k));
	}
	v->set_subtree_max(k,smax);
	return smax;
}

void vhdp::cache_max(int k_max) {
	_cache_kmax=k_max;
	_max->assign(k_max+1, vector<double>(_n,-DBL_MAX));
	for(int k=0;k<=k_max;++k) _cache_max(0,_h.get(),k);
}

double vhdp::max_lp(int k, int order) const {
	if(k<0 || k>=(int)_max->size() || order<0 || order>=_n || order>=(int)(*_max)[k].size()) return const_cast<vhdp*>(this)->lp(k,_h.get());
	return (*_max)[k][order];
}

void vhdp::clear_cache() { _h->clear_cache(); }

void vhdp::save(const char *file) { FILE *fp=fopen(file,"wb"); if(!fp) throw "failed to open save file in vhdp::save"; save(fp); fclose(fp); }
void vhdp::load(const char *file) { FILE *fp=fopen(file,"rb"); if(!fp) throw "failed to open load file in vhdp::load"; load(fp); fclose(fp); }
void vhdp::save(FILE *fp) {
	if(!fp) throw "invalid file pointer in vhdp::save";
	if(fwrite(&_n,sizeof(int),1,fp)!=1 || fwrite(&_a,sizeof(double),1,fp)!=1 || fwrite(&_b,sizeof(double),1,fp)!=1 || fwrite(&_c,sizeof(double),1,fp)!=1 || fwrite(&_d,sizeof(double),1,fp)!=1) throw "failed to write vhdp parameters";
	int as=_alpha->size(); if(fwrite(&as,sizeof(int),1,fp)!=1 || (as && fwrite(&(*_alpha)[0],sizeof(double),as,fp)!=(size_t)as)) throw "failed to write vhdp alpha"; _h->save(fp);
}
void vhdp::load(FILE *fp) {
	if(!fp) throw "invalid file pointer in vhdp::load";
	if(fread(&_n,sizeof(int),1,fp)!=1 || fread(&_a,sizeof(double),1,fp)!=1 || fread(&_b,sizeof(double),1,fp)!=1 || fread(&_c,sizeof(double),1,fp)!=1 || fread(&_d,sizeof(double),1,fp)!=1) throw "failed to read vhdp parameters";
	int as=0; if(fread(&as,sizeof(int),1,fp)!=1 || as<1) throw "failed to read vhdp alpha size"; _alpha=shared_ptr<vector<double> >(new vector<double>); for(int i=0;i<as;++i) { double x; if(fread(&x,sizeof(double),1,fp)!=1) throw "failed to read vhdp alpha"; _alpha->push_back(x); } _h->load(fp); clear_cache();
}
