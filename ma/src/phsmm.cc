#include"phsmm.h"
#include"convinience.h"
#include"rd.h"
#include"wordtype.h"
#include"vtable.h"
#include"lattice.h"
#include"generator.h"
#include"negative_binomial.h"
#include<random>
#include<cstdlib>
#include<cstdio>
#include<cmath>
#include<atomic>
#include<chrono>
#ifdef _OPENMP
#include<omp.h>
#endif

#define C 1
#define K 1000
#define ZERO 1e-36

#define A 1.
#define B 1.

using namespace std;
using namespace npbnlp;

static unordered_map<int, int> wfreq;
static unordered_map<int, int> pfreq;
static negative_binomial nb;

// diagnostics (env gated): accumulated _slice wall time, reported at exit
static std::atomic<long long> ph_slice_us(0), ph_slice_sent(0);
struct phsmm_diag {
	~phsmm_diag() {
		if (getenv("NPBNLP_PHASE_TIME") && ph_slice_sent > 0)
			fprintf(stderr, "[phsmm slice ms] %lld sentences=%lld\n",
					ph_slice_us.load()/1000, ph_slice_sent.load());
	}
};
static phsmm_diag phsmm_diag_reporter;

// debug: print the forward marginal likelihood of a sentence for
// numerical comparison between the original and marginalized paths
static void dbg_lk(vector<double>& t) {
	static const bool dbg = (getenv("NPBNLP_DEBUG_LK") != NULL);
	if (!dbg || t.empty())
		return;
	double z = 0;
	for (auto i = 0; i < (int)t.size(); ++i)
		z = math::lse(z, t[i], (i == 0));
	fprintf(stderr, "lk %.10f\n", z);
}

phsmm::phsmm():_n(1),_m(10),_l(2),_k(20),_v(C),_K(K),_a(1),_b(1),_original(false),_pos(new hpyp(_l)),_word(new vector<shared_ptr<hpyp> >),_letter(new vector<shared_ptr<vpyp> >),_lprior(new vector<double>(chartype::n, 0)),_cprior(new vector<double>(chartype::n, 0)),_num(new vector<int>(chartype::n, 0)),_change(new vector<int>(chartype::n, 0)),_len(new vector<int>(chartype::n, 0)) {
	//_pos->set_v(_K);
	for (auto i = 0; i < _k+1; ++i) {
		_word->push_back(shared_ptr<hpyp>(new hpyp(_n)));
		_letter->push_back(shared_ptr<vpyp>(new vpyp(_m)));
		//(*_letter)[i]->set_v(_v);
		(*_word)[i]->set_base((*_letter)[i].get());
	}
	beta_distribution be;
	// duration prior
	for (auto& p : *_lprior)
		p = 1.-be(A, B);

	// type change prior
	for (auto& p : *_cprior)
		p = 1.-be(A, B);
	//_prior = 1.-be(_change, _len);
}

phsmm::phsmm(int n, int m, int l, int k):_n(n),_m(m),_l(l),_k(k),_v(C),_K(K),_a(1),_b(1),_original(false),_pos(new hpyp(_l)),_word(new vector<shared_ptr<hpyp> >),_letter(new vector<shared_ptr<vpyp> >),_lprior(new vector<double>(chartype::n,0)),_cprior(new vector<double>(chartype::n, 0)),_num(new vector<int>(chartype::n,0)),_change(new vector<int>(chartype::n, 0)),_len(new vector<int>(chartype::n, 0)) {
	//_pos->set_v(_K);
	for (auto i = 0; i < _k+1; ++i) {
		_word->push_back(shared_ptr<hpyp>(new hpyp(_n)));
		_letter->push_back(shared_ptr<vpyp>(new vpyp(_m)));
		//(*_letter)[i]->set_v(_v);
		(*_word)[i]->set_base((*_letter)[i].get());
	}
	beta_distribution be;
	// duration prior
	for (auto& p : *_lprior)
		p = 1.-be(A, B);
	
	// type change prior
	for (auto& p : *_cprior)
		p = 1.-be(A, B);
	//_prior = 1.-be(_change, _len);
}

phsmm::~phsmm() {
}

void phsmm::set_k(int k) {
	if (k > 0)
		_K = k;
}

void phsmm::set(int v, int k) {
	_v = v;
	_K = k;
	_k = min(_k, _K);
	for (auto it = _letter->begin(); it != _letter->end(); ++it) {
		(*it)->set_v(_v);
	}
	_pos->set_v(k);
}

int phsmm::n() {
	return _n;
}

int phsmm::m() {
	return _m;
}

int phsmm::l() {
	return _l;
}

int phsmm::k() {
	return _k;
}

double phsmm::lexlp(word& w, int p) {
	if (p < 1 || p > _k)
		return -log((double)_v);
	return (*_word)[p]->lp(w, (*_word)[p]->h());
}

double phsmm::poslp(int p) {
	if (p < 1 || p > _k)
		return -log((double)_k);
	return _pos->lp(p, _pos->h());
}

void phsmm::slice(double a, double b) {
	if (a <= 0 || b <= 0) {
		return;
	}
	_a = a;
	_b = b;
}

void phsmm::set_original(bool f) {
	_original = f;
}

void phsmm::save(const char *file) {
	FILE *fp = NULL;
	if ((fp = fopen(file, "wb")) == NULL)
		throw "failed to open save file in phsmm::save";
	try {
		if (fwrite(&_n, sizeof(int), 1, fp) != 1)
			throw "failed to write _n in phsmm::save";
		if (fwrite(&_m, sizeof(int), 1, fp) != 1)
			throw "failed to write _m in phsmm::save";
		if (fwrite(&_l, sizeof(int), 1, fp) != 1)
			throw "failed to write _l in phsmm::save";
		if (fwrite(&_k, sizeof(int), 1, fp) != 1)
			throw "failed to write _k in phsmm::save";
		if (fwrite(&_v, sizeof(int), 1, fp) != 1)
			throw "failed to write _v in phsmm::save";
		if (fwrite(&_K, sizeof(int), 1, fp) != 1)
			throw "failed to write _K in phsmm::save";
		_pos->save(fp);
		for (auto i = 0; i < _k+1; ++i) {
			(*_word)[i]->save(fp);
			(*_letter)[i]->save(fp);
		}
		int n = _cprior->size();
		if (fwrite(&n, sizeof(int), 1, fp) != 1)
			throw "failed to save prior class num in phsmm::save";
		if (fwrite(_lprior->data(), sizeof(double), n, fp) != n)
			throw "failed to save duration prior in phsmm::save";
		if (fwrite(_cprior->data(), sizeof(double), n, fp) != n)
			throw "failed to save type change prior in phsmm::save";
	} catch (const char *ex) {
		throw ex;
	}
	fclose(fp);
}

void phsmm::load(const char *file) {
	FILE *fp = NULL;
	if ((fp = fopen(file, "rb")) == NULL)
		throw "failed to open save file in phsmm::load";
	try {
		if (fread(&_n, sizeof(int), 1, fp) != 1)
			throw "failed to read _n in phsmm::load";
		if (fread(&_m, sizeof(int), 1, fp) != 1)
			throw "failed to read _m in phsmm::load";
		if (fread(&_l, sizeof(int), 1, fp) != 1)
			throw "failed to read _l in phsmm::load";
		if (fread(&_k, sizeof(int), 1, fp) != 1)
			throw "failed to read _k in phsmm::load";
		if (fread(&_v, sizeof(int), 1, fp) != 1)
			throw "failed to read _v in phsmm::load";
		if (fread(&_K, sizeof(int), 1, fp) != 1)
			throw "failed to read _K in phsmm::load";
		_pos->load(fp);
		while ((int)_word->size() < _k+1) {
			_word->push_back(shared_ptr<hpyp>(new hpyp(_n)));
			_letter->push_back(shared_ptr<vpyp>(new vpyp(_m)));
			(*_word)[_word->size()-1]->set_base((*_letter)[_word->size()-1].get());
			//(*_letter)[_word->size()-1]->set_v(_v);
		}
		for (auto i = 0; i < _k+1; ++i) {
			(*_word)[i]->load(fp);
			(*_letter)[i]->load(fp);
		}
		int n = 0;
		if (fread(&n, sizeof(int), 1, fp) != 1)
			throw "failed to read prior class num in phsmm::load";
		_lprior->resize(n);
		_cprior->resize(n);
		if (fread(_lprior->data(), sizeof(double), n, fp) != n)
			throw "failed to read duration prior in phsmm::load";
		if (fread(_cprior->data(), sizeof(double), n, fp) != n)
			throw "failed to read type change prior in phsmm::load";
	} catch (const char *ex) {
		throw ex;
	}
	fclose(fp);
}

/*
// random init
void phsmm::init(sentence& s) {
lock_guard<mutex> m(_mutex);
shared_ptr<wid> dic = wid::create();
uniform_int_distribution<> u(1, _k-1);
shared_ptr<generator> g = generator::create();
for (int i = 0; i < s.size(); ++i) {
word& x = s.wd(i);
x.pos = u((*g)());
if ((*dic)[x] == 1) { // unk
x.id = dic->index(x);
} else {
x.id = (*dic)[x];
}
wfreq[x.id]++;
context *p = _pos->h();
for (int j = 1; j < _l; ++j) {
word& w = s.wd(i-j);
p = p->make(w.pos);
}
context *h = (*_word)[x.pos]->make(s, i);
(*_word)[x.pos]->add(x, h);
_pos->add(x.pos, p);
pfreq[x.pos]++;
}
// eos
context *h = (*_word)[0]->make(s, s.size());
(*_word)[0]->add(s.wd(s.size()),h);
context *p = _pos->h();
int eos = s.size();
for (int j = 1; j < _l; ++j) {
word& w = s.wd(eos-j);
p = p->make(w.pos);
}
_pos->add(s.wd(s.size()).pos, p);
}
*/

// initialization by model
void phsmm::init(sentence& s) {
	lock_guard<mutex> m(_mutex);
	shared_ptr<wid> dic = wid::create();
	for (int i = 0; i < s.size(); ++i) {
		word& x = s.wd(i);
		if ((*dic)[x] == 1) { // unk
			x.id = dic->index(x);
		} else {
			x.id = (*dic)[x];
		}
		wfreq[x.id]++;
		context *p = _pos->h();
		for (int j = 1; j < _l; ++j) {
			word& w = s.wd(i-j);
			p = p->make(w.pos);
			//context *q = p->make(w.pos);
			//if (!q)
			//	break;
			//p = q;
		}
		vector<double> table;
		for (int k = 1; k < _k+1; ++k) {
			const context *c = (*_word)[k]->h();
			for (int j = 1; j < _n; ++j) {
				word& w = s.wd(i-j);
				context *d = c->find(w.id);
				if (!d)
					break;
				c = d;
			}
			double lp = _pos->lp(k, p)+(*_word)[k]->lp(x, c);
			//cout << k << ":" << lp << endl;
			table.push_back(lp);
		}
		int pos = 1+rd::ln_draw(table);
		x.pos = pos;
		if (pos == _k) {
			_resize();
		}
		context *h = (*_word)[pos]->make(s, i);
		(*_word)[pos]->add(x, h);
		_pos->add(pos, p);
		pfreq[pos]++;
		type wt = wordtype::get(x);
		type t = chartype::get(x[0]);
		for (int j = 1; j < x.len; ++j) {
			type u = chartype::get(x[j]);
			if (t != u)
				++(*_change)[wt];
			t = u;
		}
		(*_len)[wt] += x.len;
		(*_num)[wt] += x.len-1;
	}
	// eos
	context *h = (*_word)[0]->make(s, s.size());
	(*_word)[0]->add(s.wd(s.size()),h);
	context *p = _pos->h();
	int eos = s.size();
	for (int j = 1; j < _l; ++j) {
		word& w = s.wd(eos-j);
		p = p->make(w.pos);
	}
	_pos->add(s.wd(s.size()).pos, p);
}

void phsmm::add(sentence& s) {
	lock_guard<mutex> m(_mutex);
	shared_ptr<wid> dic = wid::create();
	for (auto i = 0; i < s.size(); ++i) {
		word& w = s.wd(i);
		if ((*dic)[w] == 1) {
			w.id = dic->index(w);
		} else {
			w.id = (*dic)[w];
		}
		wfreq[w.id]++;
		pfreq[w.pos]++;
	}
	int rd[s.size()+1] = {0};
	rd::shuffle(rd, s.size()+1);
	for (int i = 0; i < s.size()+1; ++i) {
		word& w = s.wd(rd[i]);
		while (_k <= w.pos && _k < _K)
			_resize();
		context *h = (*_word)[w.pos]->make(s, rd[i]);
		(*_word)[w.pos]->add(w, h);
		// update pos arrangement
		context *p = _pos->h();
		for (int j = 1; j < _l; ++j) {
			word& x = s.wd(rd[i]-j);
			p = p->make(x.pos);
		}
		_pos->add(w.pos, p);
		if (!w.id) // skip bos/eos
			continue;
		type wt = wordtype::get(w);
		type t = chartype::get(w[0]);
		for (auto j = 1; j < w.len; ++j) {
			type u = chartype::get(w[j]);
			if (t != u)
				++(*_change)[wt];
			t = u;
		}
		(*_len)[wt] += w.len;
		(*_num)[wt] += w.len-1;
	}
}

void phsmm::remove(sentence& s) {
	lock_guard<mutex> m(_mutex);
	shared_ptr<wid> dic = wid::create();
	for (auto i = 0; i < s.size(); ++i) {
		word& w = s.wd(i);
		wfreq[w.id]--;
		pfreq[w.pos]--;
		if (wfreq[w.id] == 0) {
			dic->remove(w);
			wfreq.erase(w.id);
		}
	}
	for (int i = 0; i < s.size()+1; ++i) {
		word& w = s.wd(i);
		context *h = (*_word)[w.pos]->find(s, i);
		(*_word)[w.pos]->remove(w, h);
		// update pos arrangement
		context *p = _pos->h();
		for (int j = 1; j < _l; ++j) {
			word& x = s.wd(i-j);
			p = p->find(x.pos);
		}
		_pos->remove(w.pos, p);
		if (!w.id) // skip bos/eos
			continue;
		type wt = wordtype::get(w);
		type t = chartype::get(w[0]);
		for (auto j = 1; j < w.len; ++j) {
			type u = chartype::get(w[j]);
			if (t != u)
				--(*_change)[wt];
			t = u;
		}
		(*_len)[wt] -= w.len;
		(*_num)[wt] -= w.len-1;
	}
	for (int k = _k-1; pfreq[k] == 0; --k) {
		_shrink();
	}
}

void phsmm::estimate(int iter) {
	for (int i = 1; i < _k+1; ++i) {
		(*_word)[i]->gibbs(iter);
		(*_word)[i]->estimate(iter);
		(*_letter)[i]->estimate(iter);
	}
	_pos->estimate(iter);
	// type change prior and duration prior
	beta_distribution be;
	for (auto t = 0; t < chartype::n; ++t) {
		(*_lprior)[t] = 1.-be(A+(*_num)[t], B+(*_len)[t]);
		(*_cprior)[t] = 1.-be(A+(*_change)[t], B+(*_len)[t]);
	}
	//_prior = 1.-be(_change, _len);
}

void phsmm::poisson_correction(int n) {
	for (int i = 1; i < _k+1; ++i) {
		(*_word)[i]->poisson_correction(n);
	}
}

sentence phsmm::parse(io& f, int i) {
	if (!_original)
		return _minfer(f, i, true);
	lattice l(f, i);
	vt dp;
	// slice
	_slice(l);
	_type_prior(l);
	// forward filtering
	for (auto t = 0; t < (int)l.w.size(); ++t) {
		for (auto j = 0; j < l.size(t); ++j) {
			/*
			   if (l.skip(t,j))
			   continue;
			   */
			word& w = l.wd(t, j+1);
			double ln_prior = l.prior[t][j];
			for (auto pt = l.sbegin(t, j); pt != l.send(t, j); ++pt) {
				int p = *pt;
				const context *c = (*_word)[p]->h();
				const context *z = _pos->h();
				for (auto k = 0; k < l.size(t-w.len); ++k) {
					/*
					   if (l.skip(t-w.len,k))
					   continue;
					   */
					const context *h = NULL;
					word& prev = l.wd(t-w.len, k+1);
					if (_n > 1)
						h = c->find(prev.id);
					// pos transition
					for (auto it = l.sbegin(t-w.len, k); it != l.send(t-w.len, k); ++it) {
						int q = *it;
						const context *u = NULL;
						if (_l > 1)
							u = z->find(q);
						if (h && u)
							_forward(l, t-w.len-prev.len, h, u, ln_prior, w, p, prev, q, dp[t][j][p], dp[t-w.len][k][q], _n-1, _l-1, false, false);
						else if (h)
							_forward(l, t-w.len-prev.len, h, z, ln_prior, w, p, prev, q, dp[t][j][p], dp[t-w.len][k][q], _n-1, _l-1, false, true);
						else if (u)
							_forward(l, t-w.len-prev.len, c, u, ln_prior, w, p, prev, q, dp[t][j][p], dp[t-w.len][k][q], _n-1, _l-1, true, false);
						else
							_forward(l, t-w.len-prev.len, c, z, ln_prior, w, p, prev, q, dp[t][j][p], dp[t-w.len][k][q], _n-1, _l-1, true, true);
					}
				}
			}
		}
	}
	// backward sampling
	sentence s;
	word *w = l.wp(l.w.size(), 1);
	int t = (int)l.w.size()-w->len;
	bool dbg_first = true;
	while (t >= 0) {
		const context *c = (*_word)[w->pos]->h();
		const context *z = _pos->h();
		vector<double> table;
		vector<int> len;
		vector<int> pos;
		for (auto k = 0; k < l.size(t); ++k) {
			/*
			   if (l.skip(t, k))
			   continue;
			   */
			const context *h = NULL;
			word& prev = l.wd(t, k+1);
			if (_n > 1)
				h = c->find(prev.id);
			for (auto qt = l.sbegin(t, k); qt != l.send(t, k); ++qt) {
				int q = *qt;
				// prev slice
				/*
				   if (l.u(t) && (*_word)[q]->lp(prev, (*_word)[q]->h())+_pos->lp(q, _pos->h()) < l.u(t))
				   continue;
				   */
				const context *u = NULL;
				int i = table.size();
				table.push_back(1.);
				len.push_back(k+1);
				pos.push_back(q);
				if (_l > 1)
					u = z->find(q);
				if (h && u)
					_backward(l, t-prev.len, h, u, *w, w->pos, prev, q, table[i], dp[t][k][q], _n-1, _l-1, false, false);
				else if (h)
					_backward(l, t-prev.len, h, z, *w, w->pos, prev, q, table[i], dp[t][k][q], _n-1, _l-1, false, true);
				else if (u)
					_backward(l, t-prev.len, c, u, *w, w->pos, prev, q, table[i], dp[t][k][q], _n-1, _l-1, true, false);
				else
					_backward(l, t-prev.len, c, z, *w, w->pos, prev, q, table[i], dp[t][k][q], _n-1, _l-1, true, true);
			}
		}
		if (dbg_first) {
			dbg_lk(table);
			dbg_first = false;
		}
		int id = rd::best(table);
		w = l.wp(t, len[id]);
		w->pos = pos[id];
		s.w.push_back(*w);
		t -= w->len;
	}
	reverse(s.w.begin(), s.w.end());
	s.n.resize(s.w.size(), 0);
	return s;
}

sentence phsmm::sample(io& f, int i) {
	if (!_original)
		return _minfer(f, i, false);
	lattice l(f, i);
	vt dp;
	// slice
	_slice(l);
	_type_prior(l);
	// forward filtering
	for (auto t = 0; t < (int)l.w.size(); ++t) {
		for (auto j = 0; j < l.size(t); ++j) {
			/*
			   if (l.skip(t,j))
			   continue;
			   */
			word& w = l.wd(t, j+1);
			double ln_prior = l.prior[t][j];
			for (auto pt = l.sbegin(t, j); pt != l.send(t, j); ++pt) {
				int p = *pt;
				const context *c = (*_word)[p]->h();
				const context *z = _pos->h();
				for (auto k = 0; k < l.size(t-w.len); ++k) {
					/*
					   if (l.skip(t-w.len,k))
					   continue;
					   */
					const context *h = NULL;
					word& prev = l.wd(t-w.len, k+1);
					if (_n > 1)
						h = c->find(prev.id);
					// pos transition
					for (auto it = l.sbegin(t-w.len, k); it != l.send(t-w.len, k); ++it) {
						int q = *it;
						const context *u = NULL;
						if (_l > 1)
							u = z->find(q);
						if (h && u)
							_forward(l, t-w.len-prev.len, h, u, ln_prior, w, p, prev, q, dp[t][j][p], dp[t-w.len][k][q], _n-1, _l-1, false, false);
						else if (h)
							_forward(l, t-w.len-prev.len, h, z, ln_prior, w, p, prev, q, dp[t][j][p], dp[t-w.len][k][q], _n-1, _l-1, false, true);
						else if (u)
							_forward(l, t-w.len-prev.len, c, u, ln_prior, w, p, prev, q, dp[t][j][p], dp[t-w.len][k][q], _n-1, _l-1, true, false);
						else
							_forward(l, t-w.len-prev.len, c, z, ln_prior, w, p, prev, q, dp[t][j][p], dp[t-w.len][k][q], _n-1, _l-1, true, true);
					}
				}
			}
		}
	}
	// backward sampling
	sentence s;
	word *w = l.wp(l.w.size(), 1);
	int t = (int)l.w.size()-w->len;
	while (t >= 0) {
		const context *c = (*_word)[w->pos]->h();
		const context *z = _pos->h();
		vector<double> table;
		vector<int> len;
		vector<int> pos;
		for (auto k = 0; k < l.size(t); ++k) {
			/*
			   if (l.skip(t, k))
			   continue;
			   */
			const context *h = NULL;
			word& prev = l.wd(t, k+1);
			if (_n > 1)
				h = c->find(prev.id);
			for (auto qt = l.sbegin(t, k); qt != l.send(t, k); ++qt) {
				int q = *qt;
				const context *u = NULL;
				int j = table.size();
				table.push_back(1.);
				len.push_back(k+1);
				pos.push_back(q);
				if (_l > 1)
					u = z->find(q);
				if (h && u)
					_backward(l, t-prev.len, h, u, *w, w->pos, prev, q, table[j], dp[t][k][q], _n-1, _l-1, false, false);
				else if (h)
					_backward(l, t-prev.len, h, z, *w, w->pos, prev, q, table[j], dp[t][k][q], _n-1, _l-1, false, true);
				else if (u)
					_backward(l, t-prev.len, c, u, *w, w->pos, prev, q, table[j], dp[t][k][q], _n-1, _l-1, true, false);
				else
					_backward(l, t-prev.len, c, z, *w, w->pos, prev, q, table[j], dp[t][k][q], _n-1, _l-1, true, true);
			}
		}
		int id = rd::ln_draw(table);
		w = l.wp(t, len[id]);
		w->pos = pos[id];
		s.w.push_back(*w);
		t -= w->len;
	}
	reverse(s.w.begin(), s.w.end());
	s.n.resize(s.w.size(), 0);
	return s;
}

void phsmm::_forward(lattice& l, int i, const context *c, const context *t, double& ln_prior, word& w, int p, word& prev, int q, vt& a, vt& b, int n, int m, bool unk, bool not_exist) {
	if (n <= 1 && m <= 1) {
		a.v = math::lse(a.v, b.v+(*_word)[p]->lp(w, c)+_pos->lp(p, t)+ln_prior, !a.is_init());
		if (!a.is_init())
			a.set(true);
	} else {
		for (auto j = 0; j < l.size(i); ++j) {
			/*
			   if (l.skip(i, j))
			   continue;
			   */
			word& y = l.wd(i, j+1);
			const context *h = NULL;
			if (!unk && n > 1)
				h = c->find(y.id);
			for (auto pt = l.sbegin(i, j); pt != l.send(i, j); ++pt) {
				int r = *pt;
				const context *u = NULL;
				if (!not_exist && m > 1)
					u = t->find(r);
				if (h && u)
					_forward(l, i-y.len, h, u, ln_prior, w, p, y, r, a[prev.len-1][q], b[j][r], n-1, m-1, false, false);
				else if (h)
					_forward(l, i-y.len, h, t, ln_prior, w, p, y, r, a[prev.len-1][q], b[j][r], n-1, m-1, false, true);
				else if (u)
					_forward(l, i-y.len, c, u, ln_prior, w, p, y, r, a[prev.len-1][q], b[j][r], n-1, m-1, true, false);
				else
					_forward(l, i-y.len, c, t, ln_prior, w, p, y, r, a[prev.len-1][q], b[j][r], n-1, m-1, true, true);
			}
		}
	}
}

void phsmm::_backward(lattice& l, int i, const context *c, const context *t, word& w, int p, word& prev, int q, double& lpr, vt& b, int n, int m, bool unk, bool not_exist) {
	if (n <= 1 && m <= 1) {
		lpr = math::lse(lpr, b.v+(*_word)[p]->lp(w, c)+_pos->lp(p, t), (lpr == 1.));
	} else {
		for (auto j = 0; j < l.size(i); ++j) {
			/*
			   if (l.skip(i, j))
			   continue;
			   */
			word& y = l.wd(i, j+1);
			const context *h = NULL;
			if (!unk && n > 1)
				h = c->find(y.id);
			for (auto pt = l.sbegin(i, j); pt != l.send(i, j); ++pt) {
				int r = *pt;
				const context *u = NULL;
				if (!not_exist && m > 1)
					u = t->find(r);
				if (h && u)
					_backward(l, i-y.len, h, u, w, p, y, r, lpr, b[j][r], n-1, m-1, false, false);
				else if (h)
					_backward(l, i-y.len, h, t, w, p, y, r, lpr, b[j][r], n-1, m-1, false, true);
				else if (u)
					_backward(l, i-y.len, c, u, w, p, y, r, lpr, b[j][r], n-1, m-1, true, false);
				else
					_backward(l, i-y.len, c, t, w, p, y, r, lpr, b[j][r], n-1, m-1, true, true);
			}
		}
	}
}

void phsmm::_slice(lattice& l) {
	static const bool noslice = (getenv("NPBNLP_NOSLICE") != NULL);
	static const bool slcheck = (getenv("NPBNLP_SLICE_CHECK") != NULL);
	static const bool naive = (getenv("NPBNLP_NAIVE_SLICE") != NULL); // A/B: pre-memoization path
	static const bool pht = (getenv("NPBNLP_PHASE_TIME") != NULL);
	auto _t0 = std::chrono::steady_clock::now();
	beta_distribution be;
	shared_ptr<generator> g = generator::create();
	int T = (int)l.w.size();
	int M = _m; // letter (char) n-gram order; base-measure context depths 0..M-1
	// Memoize the word-base (=_lpb(word), the char-level base measure) per
	// (class, absolute char position, context depth). Overlapping word
	// candidates share these char log-probs, replacing O(segments*K*len) letter
	// -LM calls with O(T*K*M); each word base is then a sum of memoized values.
	static thread_local vector<int> cid;
	static thread_local vector<double> WL, WLE;
	static thread_local vector<int> prev;
	cid.resize(T);
	for (int p = 0; p < T; ++p)
		cid[p] = l.wd(p, 1)[0]; // char id at absolute position p
	// only depths actually reachable by some word candidate are needed; words
	// are short (<= maxlen) while the letter context M can be deep, so cap the
	// dense precompute at the longest word length to avoid wasted letter-LM calls.
	int maxlen = 1;
	for (int t = 0; t < T; ++t)
		if ((int)l.w[t].size() > maxlen)
			maxlen = (int)l.w[t].size();
	int Dcap = min(M-1, maxlen); // fill depths 0..Dcap; accesses stay within this
	size_t stride = (size_t)T*M;
	WL.assign((size_t)(_k+1)*stride, 0);
	WLE.assign((size_t)(_k+1)*stride, 0);
	prev.assign(M > 1 ? M-1 : 1, 0);
	if (!naive) {
		for (int k = 1; k < _k+1; ++k) {
			for (int p = 0; p < T; ++p) {
				for (int d = 0; d <= Dcap; ++d) {
					// real char at p with d preceding within-word chars (then BOS)
					for (int j = 0; j < M-1; ++j)
						prev[j] = (j < d && p-1-j >= 0) ? cid[p-1-j] : 0;
					WL[(size_t)k*stride + (size_t)p*M + d] = (*_letter)[k]->wlp(cid[p], prev.data(), M-1);
					// eos char (id 0) for a word ending at p with depth d last chars
					for (int j = 0; j < M-1; ++j)
						prev[j] = (j < d && p-j >= 0) ? cid[p-j] : 0;
					WLE[(size_t)k*stride + (size_t)p*M + d] = (*_letter)[k]->wlp(0, prev.data(), M-1);
				}
			}
		}
	}
	double maxdiff = 0;
	l.emit.resize(l.w.size());
	for (auto t = 0; t < (int)l.w.size(); ++t) {
		// marginarize \sum_k p(c_{t-j+1}^t, k)
		//vector<double> lpw;
		l.emit[t].resize(l.w[t].size());
		for (auto w = l.w[t].begin(); w != l.w[t].end(); ++w) {
			int len = w->len;
			int s = t - len + 1;
			int de = min(len, M-1);
			double z = 0; // p(c_{t-j+1}^t)
			vector<double> table;
			l.emit[t][w->len-1].resize(_k+1, 0);
			for (auto k = 1; k < _k+1; ++k) {
				double wlp;
				if (naive) {
					wlp = (*_word)[k]->lp(*w, (*_word)[k]->h());
				} else {
					double base = 0;
					for (int i = 0; i < len; ++i)
						base += WL[(size_t)k*stride + (size_t)(s+i)*M + min(i, M-1)];
					base += WLE[(size_t)k*stride + (size_t)t*M + de];
					wlp = (*_word)[k]->lp_root_base(*w, base);
				}
				if (slcheck) {
					double ref = (*_word)[k]->lp(*w, (*_word)[k]->h());
					double diff = fabs(wlp-ref);
					if (diff > maxdiff)
						maxdiff = diff;
				}
				l.emit[t][w->len-1][k] = wlp;
				double lp = wlp+_pos->lp(k, _pos->h());
				z = math::lse(z, lp, (z==0));
				table.push_back(lp);
			}
			if (noslice) {
				for (auto k = 1; k < _k+1; ++k)
					l.k[t][w->len-1].push_back(k);
				continue;
			}
			// p(k|c_{t-j+1}~t)
			for (auto i = table.begin(); i != table.end(); ++i) {
				*i -= z;
			}
			//lpw.push_back(z);
			// for slice pos
			/*
			   for (auto k = 1; k < _k+1; ++k) {
			// p(k|c_{t-j+1}~t)
			double lp = (*_word)[k]->lp(*w, (*_word)[k]->h())+_pos->lp(k, _pos->h())-z;
			table.push_back(lp);
			}
			*/
			//w->pos = rd::ln_draw(table)+1;
			int id = rd::ln_draw(table);
			double mu = log(be(_a, _b))+table[id];
			for (auto i = 0; i < (int)table.size(); ++i) {
				if (table[i] >= mu)
					l.k[t][w->len-1].push_back(i+1);
			}

		}
		/*
		//uniform_int_distribution<> v(1, l.size(t));
		//int len = v((*g)()); // for slice words
		int len = 1+rd::ln_draw(lpw);
		double nu = log(be(_a, _b))+lpw[len-1];
		for (auto j = 0; j < lpw.size(); ++j) {
		if (lpw[j] < nu) {
		l.check[t][j] = 1;
		}
		}
		*/
	}
	if (slcheck)
		fprintf(stderr, "[slice_check] max|clp-ref|=%.3e\n", maxdiff);
	if (pht) {
		ph_slice_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-_t0).count();
		++ph_slice_sent;
	}
}

/*
 * efficient forward filtering with marginalized forward prob
 * dp keys: [t][len][pos][λ1..λ_{n-1}][κ1..κ_{l-2}]
 *   λ_i: lengths of the previous i-th words (0 means BOS padding)
 *   κ_i: classes of the previous (i+1)-th words
 * am keys: [t][len][λ1..λ_{n-2}][pos][κ1..κ_{l-2}]
 *   = lse over the deepest length λ_{n-1} of dp (marginalized connection target)
 */
sentence phsmm::_minfer(io& f, int i, bool best) {
	lattice l(f, i);
	vt dp;
	vt am;
	vt trm;
	vt bos;
	_slice(l);
	_type_prior(l);
	int nw = _n-1;
	int nc = max(_l-1, 1);
	// build BOS pseudo table: length keys 0 x nw, class keys 0 x nc
	{
		vt *n = &bos;
		for (auto d = 0; d < nw; ++d)
			n = &(*n)[0];
		for (auto d = 0; d < nc; ++d)
			n = &(*n)[0];
		n->v = 0;
		n->set(true);
	}
	// forward filtering
	_mfill(l, dp, am, bos, trm);
	// backward sampling
	sentence s;
	word *w = l.wp(l.w.size(), 1);
	int t = (int)l.w.size()-w->len;
	if (w->pos < 0 || w->pos > _k)
		w->pos = 0;
	vector<int> lam; // window of context word lengths (λ1..λ_{nw})
	vector<int> rcs; // window of context classes (r1..r_{nc})
	{
		// joint draw of the last word chain from eos
		vector<double> tbl;
		vector<vector<int> > lpath;
		vector<vector<int> > rpath;
		vector<int> cl;
		vector<int> cr;
		const context *c = (*_word)[w->pos]->h();
		_mtable(l, t, nw, nc, c, false, *w, am[t], trm, cl, cr, tbl, lpath, rpath);
		if (tbl.empty())
			throw "failed to construct backward table in phsmm::_minfer";
		dbg_lk(tbl);
		int id = (best) ? rd::best(tbl) : rd::ln_draw(tbl);
		lam = lpath[id];
		rcs = rpath[id];
	}
	while (t >= 0) {
		int P = rcs[0];
		int J = 0;
		if (nw == 0) {
			// draw the length of the word ending at t
			vector<double> tb;
			vector<int> cand;
			for (auto it = dp[t].begin(); it != dp[t].end(); ++it) {
				vt *leaf = &(*(it->second))[P];
				for (auto d = 1; d < nc; ++d)
					leaf = &(*leaf)[rcs[d]];
				if (!leaf->is_init())
					continue;
				cand.push_back(it->first);
				tb.push_back(leaf->v);
			}
			if (tb.empty())
				throw "failed to draw a word length in phsmm::_minfer";
			int id = (best) ? rd::best(tb) : rd::ln_draw(tb);
			J = cand[id];
		} else {
			J = lam[0];
		}
		word *cur = l.wp(t, J);
		cur->pos = P;
		s.w.push_back(*cur);
		int tn = t-cur->len;
		if (tn < 0)
			break;
		int lnew = 0;
		if (nw >= 1) {
			// draw the deepest context word length of the dp entry at t
			vt *node = &dp[t];
			node = &(*node)[J];
			node = &(*node)[P];
			for (auto d = 1; d < nw; ++d)
				node = &(*node)[lam[d]];
			vector<double> tb;
			vector<int> cand;
			for (auto it = node->begin(); it != node->end(); ++it) {
				vt *leaf = it->second.get();
				for (auto d = 1; d < nc; ++d)
					leaf = &(*leaf)[rcs[d]];
				if (!leaf->is_init())
					continue;
				cand.push_back(it->first);
				tb.push_back(leaf->v);
			}
			if (tb.empty())
				throw "failed to draw a context length in phsmm::_minfer";
			int id = (best) ? rd::best(tb) : rd::ln_draw(tb);
			lnew = cand[id];
		}
		// draw the deepest context class from trans and marginalized forward prob
		vt *anode = &am[tn];
		for (auto d = 1; d < nw; ++d)
			anode = &(*anode)[lam[d]];
		if (nw >= 1)
			anode = &(*anode)[lnew];
		for (auto d = 1; d < nc; ++d)
			anode = &(*anode)[rcs[d]];
		vector<double> tb;
		vector<int> cand;
		vector<int> rc(rcs.begin()+1, rcs.end());
		for (auto it = anode->begin(); it != anode->end(); ++it) {
			vt& child = *(it->second);
			if (!child.is_init())
				continue;
			rc.push_back(it->first);
			double tr = _mtr(P, rc, trm);
			rc.pop_back();
			cand.push_back(it->first);
			tb.push_back(tr+child.v);
		}
		if (tb.empty())
			throw "failed to draw a class in phsmm::_minfer";
		int id = (best) ? rd::best(tb) : rd::ln_draw(tb);
		int rnew = cand[id];
		// shift context windows
		for (auto d = 0; d+1 < nw; ++d)
			lam[d] = lam[d+1];
		if (nw >= 1)
			lam[nw-1] = lnew;
		for (auto d = 0; d+1 < nc; ++d)
			rcs[d] = rcs[d+1];
		rcs[nc-1] = rnew;
		t = tn;
	}
	reverse(s.w.begin(), s.w.end());
	s.n.resize(s.w.size(), 0);
	return s;
}

void phsmm::_mfill(lattice& l, vt& dp, vt& am, vt& bos, vt& trm) {
	int nw = _n-1;
	for (auto t = 0; t < (int)l.w.size(); ++t) {
		for (auto j = 0; j < l.size(t); ++j) {
			word& w = l.wd(t, j+1);
			double pi = l.prior[t][j];
			int s = t-w.len;
			vt& as = (s < 0) ? bos : am[s];
			if (!as.is_init())
				continue;
			for (auto pt = l.sbegin(t, j); pt != l.send(t, j); ++pt) {
				int p = *pt;
				const context *c = (*_word)[p]->h();
				double lnp = pi+((_n == 1) ? l.emit[t][j][p] : 0);
				// own length is a kept key of am only when the emission context is non-empty;
				// for nw == 0 it is the deepest length and is marginalized out
				_mchain(l, s, nw, c, false, w, p, lnp, as, dp[t][w.len][p], (nw >= 1) ? am[t][w.len] : am[t], trm);
			}
		}
	}
}

void phsmm::_mchain(lattice& l, int pos, int d, const context *c, bool unk, word& w, int p, double lnp, vt& as, vt& dpn, vt& an, vt& trm) {
	if (d <= 0) {
		double base = lnp+((_n > 1) ? (*_word)[p]->lp(w, c) : 0);
		vector<int> rc;
		_mcls(max(_l-1, 1), rc, as, dpn, an[p], trm, p, base);
	} else {
		for (auto it = as.begin(); it != as.end(); ++it) {
			int lam = it->first;
			vt& child = *(it->second);
			if (!child.is_init())
				continue;
			word& y = (lam > 0 && pos >= 0) ? l.wd(pos, lam) : l.wd(-1, 1);
			const context *h = (!unk) ? c->find(y.id) : NULL;
			_mchain(l, pos-y.len, d-1, (h) ? h : c, (unk || !h), w, p, lnp, child, dpn[lam], (d > 1) ? an[lam] : an, trm);
		}
	}
}

void phsmm::_mcls(int e, vector<int>& rc, vt& as, vt& dpn, vt& an, vt& trm, int p, double base) {
	if (e <= 1) {
		double x = 0;
		bool init = false;
		for (auto it = as.begin(); it != as.end(); ++it) {
			vt& child = *(it->second);
			if (!child.is_init())
				continue;
			rc.push_back(it->first);
			double tr = _mtr(p, rc, trm);
			rc.pop_back();
			x = math::lse(x, tr+child.v, !init);
			init = true;
		}
		if (!init)
			return;
		dpn.v = base+x;
		dpn.set(true);
		an.v = math::lse(an.v, dpn.v, !an.is_init());
		an.set(true);
	} else {
		for (auto it = as.begin(); it != as.end(); ++it) {
			vt& child = *(it->second);
			if (!child.is_init())
				continue;
			rc.push_back(it->first);
			_mcls(e-1, rc, child, dpn[it->first], an[it->first], trm, p, base);
			rc.pop_back();
		}
	}
}

double phsmm::_mtr(int p, vector<int>& rc, vt& trm) {
	vt *n = &trm;
	for (auto r = rc.begin(); r != rc.end(); ++r)
		n = &(*n)[*r];
	vt& leaf = (*n)[p];
	if (!leaf.is_init()) {
		const context *u = _pos->h();
		int d = 0;
		for (auto r = rc.begin(); r != rc.end() && d < _l-1; ++r, ++d) {
			const context *f = u->find(*r);
			if (!f)
				break;
			u = f;
		}
		leaf.v = _pos->lp(p, u);
		leaf.set(true);
	}
	return leaf.v;
}

void phsmm::_mtable(lattice& l, int pos, int d, int e, const context *c, bool unk, word& w, vt& as, vt& trm, vector<int>& cl, vector<int>& cr, vector<double>& tbl, vector<vector<int> >& lpath, vector<vector<int> >& rpath) {
	if (d > 0) {
		for (auto it = as.begin(); it != as.end(); ++it) {
			int lam = it->first;
			vt& child = *(it->second);
			if (!child.is_init())
				continue;
			word& y = (lam > 0 && pos >= 0) ? l.wd(pos, lam) : l.wd(-1, 1);
			const context *h = (!unk && _n > 1) ? c->find(y.id) : NULL;
			cl.push_back(lam);
			_mtable(l, pos-y.len, d-1, e, (h) ? h : c, (unk || !h), w, child, trm, cl, cr, tbl, lpath, rpath);
			cl.pop_back();
		}
	} else if (e > 1) {
		for (auto it = as.begin(); it != as.end(); ++it) {
			vt& child = *(it->second);
			if (!child.is_init())
				continue;
			cr.push_back(it->first);
			_mtable(l, pos, 0, e-1, c, unk, w, child, trm, cl, cr, tbl, lpath, rpath);
			cr.pop_back();
		}
	} else {
		double em = (*_word)[w.pos]->lp(w, (_n > 1) ? c : (*_word)[w.pos]->h());
		for (auto it = as.begin(); it != as.end(); ++it) {
			vt& child = *(it->second);
			if (!child.is_init())
				continue;
			cr.push_back(it->first);
			double tr = _mtr(w.pos, cr, trm);
			tbl.push_back(em+tr+child.v);
			lpath.push_back(cl);
			rpath.push_back(cr);
			cr.pop_back();
		}
	}
}

void phsmm::_resize() {
	if (_k+1 > _K)
		return;
	++_k;
	_word->resize(_k+1, shared_ptr<hpyp>(new hpyp(_n)));
	_letter->resize(_k+1, shared_ptr<vpyp>(new vpyp(_m)));
	(*_word)[_k]->set_base((*_letter)[_k].get());
	//(*_letter)[_k]->set_v(_v);
}

void phsmm::_shrink() {
	--_k;
	_word->pop_back();
	_letter->pop_back();
}

void phsmm::_type_prior(lattice& l) {
	l.prior.resize(l.w.size());
	for (auto t = 0; t < (int)l.w.size(); ++t) {
		l.prior[t].resize(l.size(t), 0);
		for (auto j = 0; j < l.size(t); ++j) {
			word& w = l.wd(t, j+1);
			type wt = wordtype::get(w);
			type tp = chartype::get(w[0]);
			int change = 0;
			for (auto k = 1; k < w.len; ++k) {
				type u = chartype::get(w[k]);
				if (tp != u)
					++change;
				tp = u;
			}
			l.prior[t][j] = log(nb.density((*_cprior)[wt], w.len-change, change)) + log(nb.density((*_lprior)[wt], 1, w.len-1));
		}
	}
}
