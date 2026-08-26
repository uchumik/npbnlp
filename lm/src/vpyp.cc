#include"vpyp.h"
#include"hpyp.h"
#include"math.h"
#include"convinience.h"
#include<random>

#define _H_ log(1e-4)

using namespace std;
using namespace npbnlp;

vpyp::vpyp():hpyp() {
}

vpyp::vpyp(int n, double a, double b):hpyp(n,a,b) {
}

vpyp::~vpyp() {
}

double vpyp::pr_component(int k, const context *h) {
	return hpyp::pr(k, h);
}

double vpyp::pr(int k, const context *h) {
	return exp(lp(k,h));
}

double vpyp::pr(word& w, const context *h) {
	return exp(lp(w,h));
}

double vpyp::pr(chunk& c, const context *h) {
	return exp(lp(c, h));
}

// VPYLM predictive: the depth is latent, so the probability is the mixture
//
//   p(w) = sum_d stop_d * prod_{j<d} pass_j * p_d(w)
//
// with p_d the plain HPYP probability at depth d.  Accumulated in logs, so the
// sum is a logsumexp and the normaliser z is the same mixture without p_d.
// hpyp::lp is called explicitly: the parent recursion inside it must stay in
// the HPYP chain rather than re-entering this mixture.
double vpyp::lp(int k, const context *h) {
	if (!h)
		return -log(_v);
	// Sum stop_j p(k|context_j) times the passes above j over all depths.
	double ln_pr = 0;
	/*
	double ln_pr = _cache.get(k, h, chk);
	if (chk)
		return ln_pr;
	ln_pr = 0;
		*/
	double z = 0;
	const context *c = h;
	while (c) {
		int s = c->stop();
		int p = c->pass();
		double ln_pr_stop = log(s) - log(s+p);
		double ln_pr_pass = log(p) - log(s+p);
		z = math::lse(z+ln_pr_pass, ln_pr_stop, (z==0));
		ln_pr = math::lse(ln_pr+ln_pr_pass,ln_pr_stop+hpyp::lp(k,c),(ln_pr==0));
		c = c->parent();
	}
	return max(-log(_v), ln_pr-z);
	//return _cache.set(k, h, ln_pr-z);
}

double vpyp::lp(word& w, const context *h) {
	if (!h)
		return _lpb(w);
	bool chk = false;
	double ln_pr = 0;
	double z = 0;
	const context *c = h;
	while (c) {
		int s = c->stop();
		int p = c->pass();
		double ln_pr_stop = log(s) - log(s+p);
		double ln_pr_pass = log(p) - log(s+p);
		z = math::lse(z+ln_pr_pass, ln_pr_stop, (z==0));
		ln_pr = math::lse(ln_pr+ln_pr_pass, ln_pr_stop+hpyp::lp(w,c),(ln_pr == 0));
		c = c->parent();
	}
	return max(_lpb(w), ln_pr-z);
	//return _cache.set(w, h, ln_pr-z);
}

double vpyp::lp(chunk& b, const context *h) {
	if (!h)
		return _lpb(b);
	double ln_pr = 0;
	double z = 0;
	const context *c = h;
	while (c) {
		int s = c->stop();
		int p = c->pass();
		double ln_pr_stop = log(s) - log(s+p);
		double ln_pr_pass = log(p) - log(s+p);
		z = math::lse(z+ln_pr_pass, ln_pr_stop, (z==0));
		ln_pr = math::lse(ln_pr+ln_pr_pass, ln_pr_stop+hpyp::lp(b,c),(ln_pr == 0));
		c = c->parent();
	}
	return max(_lpb(b), ln_pr-z);
}

double vpyp::_lpb(word& w) const {
	if (!_base)
		return -log(_v);
	double lp = 0;
	for (int i = 0; i < w.len+1; ++i) {
		int n = w.m[i];
		context *h = _base->h();
		for (int j = 1; j < n && i-j >= -1; ++j) {
			context *c = h->find(w[i-j]);
			if (!c)
				break;
			else
				h = c;
		}
		lp += _base->lp(w[i], h);
		if (lp < _H_)
			break;
	}
	return max(_H_,lp);
	//return lp;
}

double vpyp::_lpb(chunk& b) const {
	if (!_base)
		return -log(_v);
	double lp = 0;
	for (int i = 0;i < b.len+1; ++i) {
		int n = b.n[i];
		context *h = _base->h();
		for (int j = 1; j < n && i-j >= -1; ++j) {
			context *c = h->find(b[i-j]);
			if (!c)
				break;
			else
				h = c;
		}
		lp += _base->lp(b.wd(i), h);
		if (lp < _H_)
			break;
	}
	return max(_H_,lp);
}

double vpyp::_prb(chunk& c) const {
	return exp(_lpb(c));
}

double vpyp::_prb(word& w) const {
	return exp(_lpb(w));
}
