#include"ipcfg.h"
#include"cyk.h"
#include"rd.h"
#include"convinience.h"
#include"generator.h"
#include<queue>
#include<cassert>

#ifdef _OPENMP
#include<omp.h>
#endif

#define C 50000
#define K 1000

using namespace std;
using namespace npbnlp;

static unordered_map<int, int> tfreq;

ipcfg::ipcfg():_m(20), _k(20),_K(K), _v(C), _a(1), _b(1), _nonterm(new hpyp(3)),_word(new vector<shared_ptr<hpyp> >),_letter(new vector<shared_ptr<vpyp> >) {
	_nonterm->set_v(_K);
	for (auto i = 0; i < _k+1; ++i) {
		_word->push_back(shared_ptr<hpyp>(new hpyp(1)));
		_letter->push_back(shared_ptr<vpyp>(new vpyp(_m)));
		(*_letter)[i]->set_v(_v);
		(*_word)[i]->set_base((*_letter)[i].get());
	}
}

ipcfg::ipcfg(int m):_m(m), _k(20), _K(K), _v(C), _a(1), _b(1), _nonterm(new hpyp(3)), _word(new vector<shared_ptr<hpyp> >), _letter(new vector<shared_ptr<vpyp> >) {
	_nonterm->set_v(_K);
	for (auto i = 0; i < _k+1; ++i) {
		_word->push_back(shared_ptr<hpyp>(new hpyp(1)));
		_letter->push_back(shared_ptr<vpyp>(new vpyp(_m)));
		(*_letter)[i]->set_v(_v);
		(*_word)[i]->set_base((*_letter)[i].get());
	}
}

ipcfg::~ipcfg() {
}

void ipcfg::save(const char *f) {
	FILE *fp = NULL;
	if ((fp = fopen(f, "wb")) == NULL)
		throw "failed to open save file in ipcfg::save";
	try {
		if (fwrite(&_m, sizeof(int), 1, fp) != 1)
			throw "failed to write _m in ipcfg::save";
		if (fwrite(&_k, sizeof(int), 1, fp) != 1)
			throw "failed to write _k in ipcfg::save";
		if (fwrite(&_K, sizeof(int), 1, fp) != 1)
			throw "failed to write _K in ipcfg::save";
		if (fwrite(&_v, sizeof(int), 1, fp) != 1)
			throw "failed to write _v in ipcfg::save";
		_nonterm->save(fp);
		for (auto i = 0; i < _k+1; ++i) {
			(*_word)[i]->save(fp);
			(*_letter)[i]->save(fp);
		}
	} catch (const char *ex) {
		throw ex;
	}
	fclose(fp);
}

void ipcfg::load(const char *f) {
	FILE *fp = NULL;
	if ((fp = fopen(f, "rb")) == NULL)
		throw "failed to open save file in ipcfg::load";
	try {
		if (fread(&_m, sizeof(int), 1, fp) != 1)
			throw "failed to read _m in ipcfg::load";
		if (fread(&_k, sizeof(int), 1, fp) != 1)
			throw "failed to read _k in ipcfg::load";
		if (fread(&_K, sizeof(int), 1, fp) != 1)
			throw "failed to read _K in ipcfg::load";
		if (fread(&_v, sizeof(int), 1, fp) != 1)
			throw "failed to read _v in ipcfg::load";
		_nonterm->load(fp);
		while ((int)_word->size() < _k+1) {
			_word->push_back(shared_ptr<hpyp>(new hpyp(1)));
			_letter->push_back(shared_ptr<vpyp>(new vpyp(_m)));
			(*_word)[_word->size()-1]->set_base((*_letter)[_word->size()-1].get());
			(*_letter)[_word->size()-1]->set_v(_v);
		}
		for (auto i = 0; i < _k+1; ++i) {
			(*_word)[i]->load(fp);
			(*_letter)[i]->load(fp);
		}
	} catch (const char *ex) {
		throw ex;
	}
	fclose(fp);
}

tree ipcfg::sample(io& f, int i) {
	return sample(f, i, nullptr);
}

tree ipcfg::sample(io& f, int i, tree *cur) {
	cyk c(f, i);
	if (c.s.size() == 1) {
		tree t(c.s);
		t[0].k = 0;
		t[0].i = 0;
		t[0].j = 0;
		return t;
	}
	vt dp;
	_slice(c, cur);
	// inside
	int size = c.s.size();
	for (auto j = 0; j < size; ++j) {
		_calc_preterm(c, j, dp[j][j]);
	}
	for (auto l = 1; l < size; ++l) {
		for (auto j = 0; j < size-l; ++j) {
			//double mu = c.mu[j][j+l];
			_calc_nonterm(c, j, j+l, dp);
		}
	}
	// tree sampling
	tree t(c.s);
	int k = 0; // root
	int id = t.s.size()-1; // root id
	node& root = t[id];
	root.k = k;
	root.i = 0;
	root.j = size-1;
	_traceback(c, 0, t.s.size()-1, k, dp, t);
	return t;
}

tree ipcfg::parse(io& f, int i) {
	cyk c(f, i);
	if (c.s.size() == 1) {
		tree t(c.s);
		t[0].k = 0;
		t[0].i = 0;
		t[0].j = 0;
		return t;
	}
	vt dp;
	_slice(c, nullptr);
	// inside
	int size = c.s.size();
	for (auto j = 0; j < size; ++j) {
		_calc_preterm(c, j, dp[j][j]);
	}
	for (auto l = 1; l < size; ++l) {
		for (auto j = 0; j < size-l; ++j) {
			//double mu = c.mu[j][j+l];
			_calc_nonterm(c, j, j+l, dp);
		}
	}
	// tree sampling
	tree t(c.s);
	int k = 0; // root
	int id = t.s.size()-1; // root id
	node& root = t[id];
	root.k = k;
	root.i = 0;
	root.j = size-1;
	_traceback(c, 0, t.s.size()-1, k, dp, t, true);
	return t;
}

void ipcfg::add(tree& t) {
	lock_guard<mutex> m(_mutex);
	_add(t, t.s.size()-1);
	if (tfreq[_k] > 0)
		_resize();
}

void ipcfg::_add(tree& t, int i) {
	node& z = t[i];
	tfreq[z.k]++;
	if (z.i != z.j) { // nonterminal
		node& left = t[t.s.size()*z.i+z.b-z.i*(1.+z.i)/2];
		node& right = t[t.s.size()*(z.b+1)+z.j-(1.+z.b)*(z.b+2)/2];
		context *h = _nonterm->h();
		h = h->make(right.k);
		h = h->make(left.k);
		_nonterm->add(z.k, h);
		h = _nonterm->h();
		_nonterm->add(left.k, h);
		h = h->make(left.k);
		_nonterm->add(right.k, h);
		_add(t, t.s.size()*z.i+z.b-z.i*(1.+z.i)/2);
		_add(t, t.s.size()*(z.b+1)+z.j-(1.+z.b)*(z.b+2)/2);
	} else if (z.k > 0) { // preterminal
		word& w = t.wd(z.i);
		context *h = (*_word)[z.k]->h();
		(*_word)[z.k]->add(w, h);
		_nonterm->add(z.k, _nonterm->h());
	}
}

void ipcfg::remove(tree& t) {
	lock_guard<mutex> m(_mutex);
	_remove(t, t.s.size()-1);
	for (int k = _k-1; tfreq[k] == 0; --k) {
		_shrink();
	}
}

void ipcfg::_remove(tree& t, int i) {
	node& z = t[i];
	tfreq[z.k]--;
	if (z.i != z.j) { // nonterminal
		node& left = t[t.s.size()*z.i+z.b-z.i*(1.+z.i)/2];
		node& right = t[t.s.size()*(z.b+1)+z.j-(1.+z.b)*(z.b+2)/2];
		context *h = _nonterm->h();
		h = h->find(right.k);
		h = h->find(left.k);
		_nonterm->remove(z.k, h);
		h = _nonterm->h();
		_nonterm->remove(left.k, h);
		h = h->find(left.k);
		_nonterm->remove(right.k, h);
		_remove(t, t.s.size()*z.i+z.b-z.i*(1.+z.i)/2);
		_remove(t, t.s.size()*(z.b+1)+z.j-(1.+z.b)*(z.b+2)/2);
	} else if (z.k > 0) { // preterminal
		word& w = t.wd(z.i);
		context *h = (*_word)[z.k]->h();
		(*_word)[z.k]->remove(w, h);
		_nonterm->remove(z.k, _nonterm->h());
	}
}

void ipcfg::estimate(int iter) {
	for (int i = 1; i < _k+1; ++i) {
		(*_word)[i]->gibbs(iter);
		(*_word)[i]->estimate(iter);
		(*_letter)[i]->estimate(iter);
	}
	_nonterm->estimate(iter);
}

void ipcfg::poisson_correction(int n) {
	for (int i = 1; i < _k+1; ++i) {
		(*_word)[i]->poisson_correction(n);
	}
}

void ipcfg::set(int v, int k) {
	_v = v;
	_K = k;
	_k = min(_k, _K);
	for (auto it = _letter->begin(); it != _letter->end(); ++it) {
		(*it)->set_v(_v);
	}
	_nonterm->set_v(k);
}

void ipcfg::slice(double a, double b) {
	if (a <= 0 || b <= 0) {
		return;
	}
	_a = a;
	_b = b;
}

void ipcfg::_traceback(cyk& c, int i, int j, int z, vt& a, tree& tr, bool best) {
	double mu = c.mu[i][j];
	if (i == j) { // pre-terminal
		return;
	} else { // non-terminal
		vector<double> table;
		vector<int> left;
		vector<int> right;
		vector<int> brp; // break point
		for (auto k = i; k < j; ++k) {
			for (auto l = c.begin(i,k); l != c.end(i,k); ++l) {
				double lp_l = _nonterm->lp(*l, _nonterm->h());
				context *h = _nonterm->h();
				context *t = h->find(*l);
				if (t)
					h = t;
				for (auto r = c.begin(k+1,j); r != c.end(k+1,j); ++r) {
					if (z > max(*l, *r))
						continue;
					double lp_r = _nonterm->lp(*r, h);
					context *s = _nonterm->h();
					context *u = s->find(*r);
					if (u) {
						s = u;
						u = s->find(*l);
						if (u)
							s = u;
					}
					double lp = _nonterm->lp(z,s)+lp_l+lp_r;
					if (lp < mu)
						continue;
					table.push_back(lp+a[i][k][*l].v+a[k+1][j][*r].v);
					left.push_back(*l);
					right.push_back(*r);
					brp.push_back(k);
				}
			}
		}
		int id = 0;
		if (best)
			id = rd::best(table);
		else
			id = rd::ln_draw(table);
		int b = brp[id];
		node& n = tr[tr.s.size()*i+j-i*(1.+i)/2];
		n.b = b;
		node& ln = tr[tr.s.size()*i+b-i*(1.+i)/2];
		node& rn = tr[tr.s.size()*(b+1)+j-(b+1.)*(2.+b)/2];
		ln.k = left[id];
		ln.i = i;
		ln.j = b;
		rn.k = right[id];
		rn.i = b+1;
		rn.j = j;
		_traceback(c, i, b, ln.k, a, tr);
		_traceback(c, b+1, j, rn.k, a, tr);
	}
}

void ipcfg::_calc_preterm(cyk& c, int j, vt& a) {
	word& w = c.wd(j);
	double mu = c.mu[j][j];
	for (auto k = c.begin(j,j); k != c.end(j,j); ++k) {
		double lp = (*_word)[*k]->lp(w, (*_word)[*k]->h())+_nonterm->lp(*k, _nonterm->h());
		if (lp >= mu) {
			a[*k].v = math::lse(a[*k].v, lp, true);
			a[*k].set(true);
		}
	}
}

void ipcfg::_calc_nonterm(cyk& c, int i, int j, vt& a) {
	double mu = c.mu[i][j];
	for (auto k = i; k < j; ++k) {
		for (auto l = c.begin(i,k); l != c.end(i,k); ++l) {
			double lp_l = _nonterm->lp(*l, _nonterm->h());
			context *h = _nonterm->h();
			context *t = h->find(*l);
			if (t)
				h = t;
			for (auto r = c.begin(k+1,j); r != c.end(k+1,j); ++r) {
				double lp_r = _nonterm->lp(*r, h);
				context *s = _nonterm->h();
				context *u = s->find(*r);
				if (u) {
					s = u;
					u = s->find(*l);
					if (u)
						s = u;
				}
				for (auto z = c.begin(i,j); z != c.end(i,j); ++z) {
					if (*z > max(*l, *r))
						continue;
					double lp = _nonterm->lp(*z,s)+lp_l+lp_r;
					if (lp < mu)
						continue;
					a[i][j][*z].v = math::lse(a[i][j][*z].v, lp+a[i][k][*l].v+a[k+1][j][*r].v, !a[i][j][*z].is_init());
					if (!a[i][j][*z].is_init())
						a[i][j][*z].set(true);
				}
			}
		}
	}
}

// Recursively walk the current tree T from the root and record, for every
// span (i,j) that appears in T, a pointer to its node. Node index layout is
// idx(i,j) = N*i + j - i*(i+1)/2 (see _add/_remove). on-path nodes carry
// their assigned label in node.k (>=0); off-path cells keep the ctor default
// node.k == -1 and are never touched here.
void ipcfg::_collect_spans(tree& t, int idx, vector<vector<const node*> >& on) {
	node& z = t[idx];
	if (z.k < 0)
		return;
	on[z.i][z.j] = &z;
	if (z.i == z.j) // pre-terminal
		return;
	int N = t.s.size();
	int b = z.b;
	_collect_spans(t, N*z.i+b-z.i*(1.+z.i)/2, on);
	_collect_spans(t, N*(b+1)+z.j-(1.+b)*(b+2)/2, on);
}

void ipcfg::_slice(cyk& l, tree *cur) {
	int size = l.s.size();
	bool cond = (cur != nullptr && cur->s.size() == size && size > 1);
	if (cond) {
		// --- WO-005: span-independent mu conditioned on the current tree T ---
		// Deviation from the note's unscaled beta(a,b): off-path spans keep the
		// current per-span fresh draw (log(be)+table[id]) instead of beta(a,b),
		// because the unscaled threshold with the present a,b would over-prune
		// and wreck the search space. mu depends only on the fixed post-remove
		// scores, so slice-sampling validity (marginalizing mu recovers P) holds.
		vector<vector<const node*> > on(size, vector<const node*>(size, nullptr));
		_collect_spans(*cur, size-1, on); // root idx = N-1
		// pre-terminals
		for (int i = 0; i < size; ++i) {
			const node *z = on[i][i];
			if (z != nullptr && z->k >= 1 && z->k <= _k)
				_slice_preterm_cond(l, i, z->k); // on-path: mu = log(be)+score_cur
			else
				_slice_preterm(l, i);            // off-path: fresh draw
		}
		// non-terminals, span-independent (no pivot walk / shared nu)
		for (int m = 1; m < size-1; ++m) {
			for (int i = 0; i < size-m; ++i) {
				int j = i+m;
				const node *z = on[i][j];
				bool onpath = false;
				int lc = 0, rc = 0, kc = 0, mc = 0;
				if (z != nullptr && z->i == i && z->j == j && z->i != z->j) {
					int b = z->b;
					const node *ln = on[i][b];
					const node *rn = on[b+1][j];
					if (ln != nullptr && rn != nullptr) {
						lc = ln->k; rc = rn->k; kc = b; mc = z->k;
						if (mc >= 1 && mc <= _k && lc >= 1 && lc <= _k &&
						    rc >= 1 && rc <= _k && mc <= max(lc, rc))
							onpath = true;
					}
				}
				if (onpath)
					_slice_nonterm_cond(l, i, j, lc, rc, kc, mc);
				else
					_draw(l, i, j); // off-path: per-span fresh draw
			}
		}
		// root
		const node *r = on[0][size-1];
		if (r != nullptr && r->i == 0 && r->j == size-1 && r->i != r->j) {
			int b = r->b;
			const node *ln = on[0][b];
			const node *rn = on[b+1][size-1];
			if (ln != nullptr && rn != nullptr &&
			    ln->k >= 1 && ln->k <= _k && rn->k >= 1 && rn->k <= _k)
				_slice_root_cond(l, ln->k, rn->k);
			else
				_slice_root(l);
		} else {
			_slice_root(l);
		}
		return;
	}
	// --- cur == nullptr: original path (parse), unchanged incl. RNG usage ---
	// terminal
	for (auto i = 0; i < l.s.size(); ++i) {
		_slice_preterm(l, i);
	}
	vector<double> table;
	for (auto i = 0; i < l.s.size()-1; ++i) {
		table.push_back(_marginalize(l,i,i+1));
	}
	int p = rd::ln_draw(table);
	//shared_ptr<generator> g = generator::create();
	//uniform_int_distribution<> u(0, l.s.size()-2);
	//int p = u((*g)());
	//non terminal
	for (auto m = 1; m < l.s.size()-1; ++m) {
		//int len = l.s.size()-m;
		//uniform_int_distribution<> v(0, len-1);
		//int p = v((*g)());
		// draw nu for slice non-terminal
		//double nu = _draw(l, p, p+m);
		double nu = _draw(l, p, p+m);
		// slice non-terminals
		for (auto i = 0; i < l.s.size()-m; ++i) {
			if (i == p)
				continue;
			_slice_nonterm(l, i, i+m, nu);
		}
		vector<double> cand;
		if (p > 0) {
			cand.push_back(_marginalize(l,p-1,p+m));
		}
		if (p+m < l.s.size()-1) {
			cand.push_back(_marginalize(l,p,p+m+1));
		}
		if (cand.size() > 1) {
			int id = rd::ln_draw(cand);
			if (id == 0) {
				p -= 1;
			}
		} else if (p > 0) {
			p -= 1;
		}
	}
	// root
	_slice_root(l);
}

double ipcfg::_marginalize(cyk& c, int i, int j) {
	double z = 0;
	for (auto k = i; k < j; ++k) {
		for (auto l = c.begin(i,k); l != c.end(i,k); ++l) {
			double lp_l = _nonterm->lp(*l, _nonterm->h());
			context *h = _nonterm->h();
			context *t = h->find(*l);
			if (t)
				h = t;
			for (auto r = c.begin(k+1,j); r != c.end(k+1,j); ++r) {
				double lp_r = _nonterm->lp(*r, h);
				context *s = _nonterm->h();
				context *u = s->find(*r);
				if (u) {
					s = u;
					u = s->find(*l);
					if (u)
						s = u;
				}
				for (auto m = max(*l,*r); m > 0; --m) {
					double lp =_nonterm->lp(m,s)+lp_l+lp_r;
					math::lse(z,lp,(z==0.));
				}
			}
		}
	}
	return z;
}

double ipcfg::_draw(cyk& c, int i, int j) {
	beta_distribution be;
	vector<double> table;
	vector<int> z;
	for (auto k = i; k < j; ++k) {
		for (auto l = c.begin(i,k); l != c.end(i,k); ++l) {
			double lp_l = _nonterm->lp(*l, _nonterm->h());
			context *h = _nonterm->h();
			context *t = h->find(*l);
			if (t)
				h = t;
			for (auto r = c.begin(k+1,j); r != c.end(k+1,j); ++r) {
				double lp_r = _nonterm->lp(*r, h);
				context *s = _nonterm->h();
				context *u = s->find(*r);
				if (u) {
					s = u;
					u = s->find(*l);
					if (u)
						s = u;
				}
				for (auto m = max(*l,*r); m > 0; --m) {
					double lp = _nonterm->lp(m,s)+lp_l+lp_r;
					table.push_back(lp);
					z.push_back(m);
				}
			}
		}
	}
	int id = rd::ln_draw(table);
	double mu = log(be(_a,_b))+table[id];
	c.mu[i][j] = mu;
	for (auto m = 0; m < (int)table.size(); ++m) {
		if (table[m] >= mu)
			c.k[i][j].insert(z[m]);
	}
	return mu;
}

void ipcfg::_slice_nonterm(cyk& c, int i, int j, double mu) {
	vector<double> table;
	vector<int> z;
	for (auto k = i; k < j; ++k) {
		for (auto l = c.begin(i,k); l != c.end(i,k); ++l) {
			double lp_l = _nonterm->lp(*l, _nonterm->h());
			context *h = _nonterm->h();
			context *t = h->find(*l);
			if (t)
				h = t;
			for (auto r = c.begin(k+1,j); r != c.end(k+1,j); ++r) {
				double lp_r = _nonterm->lp(*r, h);
				context *s = _nonterm->h();
				context *u = s->find(*r);
				if (u) {
					s = u;
					u = s->find(*l);
					if (u)
						s = u;
				}
				for (auto m = max(*l, *r); m > 0; --m) {
					double lp = _nonterm->lp(m,s)+lp_l+lp_r;
					table.push_back(lp);
					z.push_back(m);
				}
			}
		}
	}
	// P(B,C|A) := P(A->B C|A)
	// P(B,C|A) \propto P(B,C,A) = P(A|B,C)P(B,C)
	c.mu[i][j] = mu;
	for (auto m = 0; m < (int)table.size(); ++m) {
		if (table[m] >= mu) {
			c.k[i][j].insert(z[m]);
		}
	}
}

// WO-005 on-path non-terminal: condition mu on the current rule
// A(mc) -> B(lc) C(rc) split at kc. score_cur = log P(A|B,C) + log P(B) +
// log P(C|B) reproduces exactly what _draw/_slice_nonterm push to the table
// (_nonterm->lp(mc,s)+lp_l+lp_r). mu = log(be)+score_cur with log(be)<=0
// guarantees score_cur >= mu, so the current rule always survives the slice.
void ipcfg::_slice_nonterm_cond(cyk& c, int i, int j, int lc, int rc, int kc, int mc) {
	(void)kc; // split point is implied by lc/rc contexts; kept for clarity
	beta_distribution be;
	// score of the current on-path rule (same context construction as _draw)
	double lp_l = _nonterm->lp(lc, _nonterm->h());
	context *h = _nonterm->h();
	context *t = h->find(lc);
	if (t)
		h = t;
	double lp_r = _nonterm->lp(rc, h);
	context *s = _nonterm->h();
	context *u = s->find(rc);
	if (u) {
		s = u;
		u = s->find(lc);
		if (u)
			s = u;
	}
	double score_cur = _nonterm->lp(mc, s)+lp_l+lp_r;
	double mu = log(be(_a, _b))+score_cur;
	c.mu[i][j] = mu;
	// permitted set: enumerate every rule of this span exactly as _draw does
	for (auto k = i; k < j; ++k) {
		for (auto l = c.begin(i,k); l != c.end(i,k); ++l) {
			double lpl = _nonterm->lp(*l, _nonterm->h());
			context *hh = _nonterm->h();
			context *tt = hh->find(*l);
			if (tt)
				hh = tt;
			for (auto r = c.begin(k+1,j); r != c.end(k+1,j); ++r) {
				double lpr = _nonterm->lp(*r, hh);
				context *ss = _nonterm->h();
				context *uu = ss->find(*r);
				if (uu) {
					ss = uu;
					uu = ss->find(*l);
					if (uu)
						ss = uu;
				}
				for (auto mm = max(*l,*r); mm > 0; --mm) {
					double lp = _nonterm->lp(mm, ss)+lpl+lpr;
					if (lp >= mu)
						c.k[i][j].insert(mm);
				}
			}
		}
	}
	assert(c.k[i][j].count(mc) > 0); // current parent label must survive
}
/*
   void ipcfg::_slice(cyk& l) {
// terminal
for (auto i = 0; i < l.s.size(); ++i) {
_slice_preterm(l, i);
}
// non terminal
for (auto m = 1; m < l.s.size()-1; ++m) {
for (auto i = 0; i < l.s.size()-m; ++i) {
_slice_nonterm(l, i, i+m);
}
}
// root
_slice_root(l);
}
*/

void ipcfg::_slice_preterm(cyk& l, int i) {
	beta_distribution be;
	//shared_ptr<generator> g = generator::create();
	word& w = l.wd(i);
	vector<double> table;
	for (auto k = 1; k < _k+1; ++k) {
		double lp = (*_word)[k]->lp(w, (*_word)[k]->h())+_nonterm->lp(k, _nonterm->h());
		table.push_back(lp);
	}
	int id = rd::ln_draw(table);
	double mu = log(be(_a, _b))+table[id];
	//double mu = table[id];
	l.mu[i][i] = mu;
	for (auto j = 0; j < (int)table.size(); ++j) {
		if (table[j] >= mu) {
			l.k[i][i].insert(j+1);
		}
	}
}

// WO-005 on-path pre-terminal: condition mu on the current label.
// table[label-1] is the score _slice_preterm/_calc_preterm assign to this
// label; mu = log(be)+score_cur keeps that label (and any richer ones) alive.
void ipcfg::_slice_preterm_cond(cyk& l, int i, int label) {
	beta_distribution be;
	word& w = l.wd(i);
	vector<double> table;
	for (auto k = 1; k < _k+1; ++k) {
		double lp = (*_word)[k]->lp(w, (*_word)[k]->h())+_nonterm->lp(k, _nonterm->h());
		table.push_back(lp);
	}
	double score_cur = table[label-1];
	double mu = log(be(_a, _b))+score_cur;
	l.mu[i][i] = mu;
	for (auto j = 0; j < (int)table.size(); ++j) {
		if (table[j] >= mu) {
			l.k[i][i].insert(j+1);
		}
	}
	assert(l.k[i][i].count(label) > 0); // current pre-terminal must survive
}

/*
   void ipcfg::_slice_nonterm(cyk& c, int i, int j) {
   beta_distribution be;
//shared_ptr<generator> g = generator::create();
vector<double> table;
vector<int> z;
for (auto k = i; k < j; ++k) {
for (auto l = c.begin(i,k); l != c.end(i,k); ++l) {
double lp_l = _nonterm->lp(*l, _nonterm->h());
context *h = _nonterm->h();
context *t = h->find(*l);
if (t)
h = t;
for (auto r = c.begin(k+1,j); r != c.end(k+1,j); ++r) {
double lp_r = _nonterm->lp(*r, h);
context *s = _nonterm->h();
context *u = s->find(*r);
if (u) {
s = u;
u = s->find(*l);
if (u)
s = u;
}
//for (auto m = 1; m < _k+1; ++m) {
for (auto m = max(*l,*r); m > 0; --m) {
double lp = _nonterm->lp(m, s)+lp_l+lp_r;
table.push_back(lp);
z.push_back(m);
}
}
}
}
// P(B,C|A) := P(A->B C|A)
// P(B,C|A) \propto P(B,C,A) = P(A|B,C)P(B,C)
// draw A ~ P(A,B,C) for a threshold at cell_{i,j}
int id = rd::ln_draw(table);
double mu = log(be(_a, _b))+table[id];
//double mu = table[id];
c.mu[i][j] = mu;
for (auto m = 0; m < (int)table.size(); ++m) {
if (table[m] >= mu) {
c.k[i][j].insert(z[m]);
}
}
}
*/

void ipcfg::_slice_root(cyk& c) {
	beta_distribution be;
	//shared_ptr<generator> g = generator::create();
	int size = c.s.size();
	vector<double> table;
	for (auto k = 0; k < size-1; ++k) {
		for (auto l = c.begin(0, k); l != c.end(0, k); ++l) {
			double lp_l = _nonterm->lp(*l, _nonterm->h());
			context *h = _nonterm->h();
			context *t = h->find(*l);
			if (t)
				h = t;
			for (auto r = c.begin(k+1, size-1); r != c.end(k+1, size-1); ++r) {
				double lp_r = _nonterm->lp(*r, h);
				context *s = _nonterm->h();
				context *u = s->find(*r);
				if (u) {
					s = u;
					u = s->find(*l);
					if (u)
						s = u;
				}
				double lp = _nonterm->lp(0, s)+lp_l+lp_r;
				table.push_back(lp);
			}
		}
	}
	int id = rd::ln_draw(table);
	// table[id] is a log-domain score (== _nonterm->lp(...)+lp_l+lp_r); the
	// old form log(be(_a,_b)+table[id]) took log of (0,1)+negative => NaN,
	// which silenced root pruning entirely. Match the other slice helpers:
	// mu = log(be) + score, so the current root decomposition (score >= mu)
	// always survives while lower-scoring ones are pruned.
	double mu = log(be(_a, _b))+table[id];
	c.mu[0][size-1] = mu;
	c.k[0][size-1].insert(0);
	/*
	   for (auto m = 0; m < table.size(); ++m) {
	   if (table[m] >= mu)
	   c.k[0][size-1][0]+=1;
	   */
}

// WO-005 on-path root: root label is always 0; condition mu on the current
// root split into children lc/rc. score_cur = log P(0|lc,rc)+log P(lc)+
// log P(rc|lc) matches _slice_root's table entry; mu = log(be)+score_cur
// ensures the current root decomposition survives traceback pruning.
void ipcfg::_slice_root_cond(cyk& c, int lc, int rc) {
	beta_distribution be;
	int size = c.s.size();
	double lp_l = _nonterm->lp(lc, _nonterm->h());
	context *h = _nonterm->h();
	context *t = h->find(lc);
	if (t)
		h = t;
	double lp_r = _nonterm->lp(rc, h);
	context *s = _nonterm->h();
	context *u = s->find(rc);
	if (u) {
		s = u;
		u = s->find(lc);
		if (u)
			s = u;
	}
	double score_cur = _nonterm->lp(0, s)+lp_l+lp_r;
	double mu = log(be(_a, _b))+score_cur;
	c.mu[0][size-1] = mu;
	c.k[0][size-1].insert(0);
}

void ipcfg::_resize() {
	if (_k+1 > _K)
		return;
	++_k;
	_word->resize(_k+1, shared_ptr<hpyp>(new hpyp(1)));
	_letter->resize(_k+1, shared_ptr<vpyp>(new vpyp(_m)));
	(*_word)[_k]->set_base((*_letter)[_k].get());
	(*_letter)[_k]->set_v(_v);
}

void ipcfg::_shrink() {
	--_k;
	_word->pop_back();
	_letter->pop_back();
}
