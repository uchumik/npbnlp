#include "vhdp_context.h"
#include "vhdp.h"
#include <cmath>

using namespace std;
using namespace npbnlp;

vhdp_context::vhdp_context(): context(), _cdp(new vector<stick>), _child(new children),
	_pr(new vector<double>), _pass(new vector<double>), _gamma(new vector<double>), _beta_inf(new vector<double>), _smax(new vector<double>) {
}

vhdp_context::vhdp_context(int k, context *h): context(k, h), _cdp(new vector<stick>), _child(new children),
	_pr(new vector<double>), _pass(new vector<double>), _gamma(new vector<double>), _beta_inf(new vector<double>), _smax(new vector<double>) {
}

vhdp_context::~vhdp_context() {
}

const children& vhdp_context::child() const { return *_child; }

double vhdp_context::subtree_max(int k) const {
	if (k < 0 || k >= (int)_smax->size())
		return DBL_MAX; // unknown -> no pruning
	return (*_smax)[k];
}

void vhdp_context::set_subtree_max(int k, double v) {
	// No lock: cache_max is a single-threaded bulk pass over the tree, and this
	// is a cache, not a customer count.  Taking the mutex here cost ~50s per run
	// at 2.4e9 calls.
	while ((int)_smax->size() <= k)
		_smax->push_back(-DBL_MAX);
	(*_smax)[k] = v;
}

void vhdp_context::_ensure(vector<stick>& v, int k) {
	while ((int)v.size() <= k)
		v.push_back({0, 0});
}

context* vhdp_context::find(int k) const {
	auto it = _child->find(k);
	if (it == _child->end())
		return NULL;
	return it->second.get();
}

context* vhdp_context::make(int k) {
	context *c = find(k);
	if (c)
		return c;
	lock_guard<mutex> m(_mutex);
	auto it = _child->find(k);
	if (it == _child->end())
		(*_child)[k] = shared_ptr<vhdp_context>(new vhdp_context(k, this));
	return (*_child)[k].get();
}

int vhdp_context::stick_size() const { return (int)_cdp->size(); }

int vhdp_context::stick_stop(int k) const {
	if (k < 0 || k >= (int)_cdp->size())
		return 0;
	return (*_cdp)[k].stop;
}

int vhdp_context::stick_n(int k) const {
	if (k < 0 || k >= (int)_cdp->size())
		return 0;
	return (*_cdp)[k].stop + (*_cdp)[k].pass;
}

static double cache_get(const vector<double>& v, int k) {
	if (k < 0 || k >= (int)v.size())
		return -DBL_MAX;
	return v[k];
}

double vhdp_context::cache_pr(int k) const { return cache_get(*_pr, k); }
double vhdp_context::cache_pass(int k) const { return cache_get(*_pass, k); }
double vhdp_context::cache_gamma(int k) const { return cache_get(*_gamma, k); }
double vhdp_context::cache_beta_inf(int k) const { return cache_get(*_beta_inf, k); }

static void cache_set(vector<double>& v, int k, double p) {
	while ((int)v.size() <= k)
		v.push_back(-DBL_MAX);
	v[k] = p;
}

void vhdp_context::set_cache_pr(int k, double p) { lock_guard<mutex> m(_mutex); cache_set(*_pr, k, p); }
void vhdp_context::set_cache_pass(int k, double p) { lock_guard<mutex> m(_mutex); cache_set(*_pass, k, p); }
void vhdp_context::set_cache_gamma(int k, double p) { lock_guard<mutex> m(_mutex); cache_set(*_gamma, k, p); }
void vhdp_context::set_cache_beta_inf(int k, double p) { lock_guard<mutex> m(_mutex); cache_set(*_beta_inf, k, p); }

void vhdp_context::clear_cache() {
	lock_guard<mutex> m(_mutex);
	_pr->clear();
	_pass->clear();
	_gamma->clear();
	_beta_inf->clear();
	_smax->clear();
	for (auto& p : *_child)
		static_cast<vhdp_context*>(p.second.get())->clear_cache();
}

bool vhdp_context::add(int k, lm *m) {
	// Invalidate this node's subtree only.  Everything cached below depends on
	// this node's counts through pr(k, parent); nothing above does until the CRP
	// propagates, and that walks up calling add() on each ancestor in turn.
	clear_cache();
	// Order statistics belong here, not at the caller: the CRP walks up to the
	// ancestors, and the prototype's Context::add runs _context_add() at every node
	// it reaches.  Counting only the caller's node leaves shallow contexts with
	// pass but no stop, which drives the order posterior toward the deepest order.
	incr_stop();
	for (context *p=parent(); p; p=p->parent()) p->incr_pass();
	{
		lock_guard<mutex> l(_mutex);
		_ensure(*_cdp, k);
		++(*_cdp)[k].stop;
	}
	{
		lock_guard<mutex> l(_mutex);
		for (int j = 0; j < k; ++j) {
			_ensure(*_cdp, j);
			++(*_cdp)[j].pass;
		}
	}
	return _crp_add(k, m);
}

bool vhdp_context::remove(int k) {
	clear_cache();
	decr_stop();
	for (context *p=parent(); p; p=p->parent()) p->decr_pass();
	{
		lock_guard<mutex> l(_mutex);
		if (k >= (int)_cdp->size() || (*_cdp)[k].stop <= 0)
			throw "invalid vhdp stick removal";
		--(*_cdp)[k].stop;
		for (int j = 0; j < k; ++j)
			--(*_cdp)[j].pass;
	}
	return _crp_remove(k);
}

bool vhdp_context::_crp_add(int k, lm *m) {
	lock_guard<mutex> l(_mutex);
	++_customer;
	auto it = _restaurant->find(k);
	if (it == _restaurant->end())
		it = _restaurant->emplace(k, shared_ptr<arrangements>(new arrangements)).first;
	vector<int>& r = *it->second->table;
	vector<double> p(r.size()+1, 0.);
	double z = 0.;
	for (int i = 0; i < (int)r.size(); ++i) { p[i] = r[i]; z += p[i]; }
	p[r.size()] = m->alpha(_n) * m->pr(k, _parent);
	z += p[r.size()];
	++it->second->n;
	int id = rd::draw(z, p);
	if (id < 0) id = 0;
	if (id >= (int)p.size()) id = p.size()-1;
	if (id == (int)r.size()) { r.push_back(1); ++_table; return true; }
	++r[id];
	return false;
}

bool vhdp_context::_crp_remove(int k) {
	lock_guard<mutex> l(_mutex);
	auto it = _restaurant->find(k);
	if (it == _restaurant->end() || it->second->table->empty())
		throw "invalid vhdp restaurant removal";
	vector<int>& r = *it->second->table;
	vector<double> p(r.begin(), r.end());
	double z = 0.;
	for (double x : p) z += x;
	int id = rd::draw(z, p);
	if (id < 0) id = 0;
	if (id >= (int)p.size()) id = p.size()-1;
	--_customer;
	--it->second->n;
	--r[id];
	if (r[id] == 0) {
		--_table;
		r.erase(r.begin()+id);
		if (r.empty()) _restaurant->erase(it);
		return true;
	}
	return false;
}

void vhdp_context::estimate_a(vector<double>& a, vector<double>& b, lm *m) {
	for (auto& p : *_child)
		static_cast<vhdp_context*>(p.second.get())->estimate_a(a, b, m);
	shared_ptr<generator> g = generator::create();
	bernoulli_distribution d;
	beta_distribution be;
	// One auxiliary pair per DISH, as Context::estimate_t does, not one per node.
	// Each (context, state) carries its own stick here, so each dish is its own
	// Beta process; drawing once per node with the node's total customer count
	// under-accumulates b by a factor of the dish count.  log Beta(.) is negative
	// and enters the Gamma rate as _d - b, so too few terms leave the rate small
	// and alpha pinned near its initial value: measured a1 = 25.4 after six epochs
	// against the prototype's 0.25, which is what pushed the order posterior deep.
	double al = m->alpha(_n);
	a[_n] += _table;
	for (auto& r : *_restaurant) {
		int cu = stick_stop(r.first);
		if (cu <= 0) // Beta(1+a, 0) is not a distribution; the prototype never
			continue; // hits this because a seated dish has stop > 0.
		bernoulli_distribution::param_type mu((double)cu/(al+cu));
		a[_n] -= d((*g)(), mu);
		b[_n] += log(be(1.+al, cu));
	}
}

void vhdp_context::save(FILE *fp) {
	if (!fp) throw "invalid file pointer in vhdp_context::save";
#define VW(x) if (fwrite(&(x), sizeof(x), 1, fp) != 1) throw "failed to write vhdp_context";
	VW(_k); VW(_n); VW(_customer); VW(_table); VW(_a); VW(_b); VW(_stop); VW(context::_pass);
	int nr = _restaurant->size(); VW(nr);
	for (auto& q : *_restaurant) {
		int key=q.first, cn=q.second->n, nt=q.second->table->size(); VW(key); VW(cn); VW(nt);
		if (nt && fwrite(&(*q.second->table)[0], sizeof(int), nt, fp) != (size_t)nt) throw "failed to write vhdp restaurant";
	}
	int nc = _cdp->size(); VW(nc);
	for (auto& x : *_cdp) { VW(x.stop); VW(x.pass); }
	int child = _child->size(); VW(child);
	for (auto& q : *_child) { int key=q.first; VW(key); static_cast<vhdp_context*>(q.second.get())->save(fp); }
#undef VW
}

void vhdp_context::load(FILE *fp) {
	if (!fp) throw "invalid file pointer in vhdp_context::load";
#define VR(x) if (fread(&(x), sizeof(x), 1, fp) != 1) throw "failed to read vhdp_context";
	VR(_k); VR(_n); VR(_customer); VR(_table); VR(_a); VR(_b); VR(_stop); VR(context::_pass);
	_restaurant->clear();
	int nr=0; VR(nr);
	for (int i=0; i<nr; ++i) { int key,cn,nt; VR(key); VR(cn); VR(nt); auto q=shared_ptr<arrangements>(new arrangements); q->n=cn; q->table->resize(nt); if (nt && fread(&(*q->table)[0],sizeof(int),nt,fp)!=(size_t)nt) throw "failed to read vhdp restaurant"; (*_restaurant)[key]=q; }
	_cdp->clear(); int nc=0; VR(nc); for (int i=0;i<nc;++i) { stick x; VR(x.stop); VR(x.pass); _cdp->push_back(x); }
	_child->clear(); int child=0; VR(child); for (int i=0;i<child;++i) { int key; VR(key); shared_ptr<vhdp_context> q(new vhdp_context(key,this)); q->load(fp); (*_child)[key]=q; }
#undef VR
	clear_cache();
}
