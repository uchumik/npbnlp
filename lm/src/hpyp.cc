#include"hpyp.h"
#include"math.h"
#include"beta.h"
#include"poisson.h"
#include"convinience.h"
#include"chartype.h"
#include"wordtype.h"
#include<random>
#ifdef _OPENMP
#include<omp.h>
#endif

using namespace std;
using namespace npbnlp;
using gamma_dist = gamma_distribution<>;

#define VOCAB 1
#define STRENGTH 1
#define DISCOUNT 0.5
#define POISSON_A 0.2
#define POISSON_B 0.1
#define MAXLEN 100

hpyp::hpyp():_n(1),_a(1),_b(1),_base(NULL),_v(VOCAB),_h(new context),_discount(new vector<double>(_n, DISCOUNT)),_strength(new vector<double>(_n, STRENGTH)),_bc(nullptr),_cbc(nullptr),/*_poisson(nullptr),*/_lambda(nullptr),/*_w(nullptr),*/_f(0)/*_f(nullptr)*/,_length(nullptr)/*,_lambda(nullptr)*/  {
}

hpyp::hpyp(int n, double a, double b):_n(n),_a(a),_b(b),_base(NULL),_v(VOCAB),_h(new context),_discount(new vector<double>(_n, DISCOUNT)),_strength(new vector<double>(_n, STRENGTH)),_bc(nullptr),_cbc(nullptr),/*_poisson(nullptr),*/_lambda(nullptr),/*_w(nullptr),*/_f(0)/*_f(nullptr)*/,_length(nullptr)/*,_lambda(nullptr)*/  {
}

hpyp::hpyp(const hpyp& lm):_n(lm._n),_a(lm._a),_b(lm._b),_v(lm._v),_h(lm._h),_discount(lm._discount),_strength(lm._strength),_bc(lm._bc),_cbc(lm._cbc),_lambda(lm._lambda),_f(lm._f),_length(lm._length) {
}

hpyp::hpyp(hpyp&& lm):_n(lm._n),_a(lm._a),_b(lm._b),_v(lm._v),_h(lm._h),_discount(lm._discount),_strength(lm._strength),_bc(lm._bc),_cbc(lm._cbc),_lambda(lm._lambda),_f(lm._f),_length(lm._length) {
}

hpyp& hpyp::operator=(const hpyp& lm) {
	_n = lm._n;
	_a = lm._a;
	_b = lm._b;
	_v = lm._v;
	_h = lm._h;
	_bc = lm._bc;
	_cbc = lm._cbc;
	_discount = lm._discount;
	_strength = lm._strength;
	_lambda = lm._lambda;
	_length = lm._length;
	_f = lm._f;
	_base = lm._base;
	_cbase = lm._cbase;
	_cbase_add = lm._cbase_add;
	_cbase_remove = lm._cbase_remove;
	return *this;
}

hpyp& hpyp::operator=(const hpyp&& lm) noexcept {
	_n = lm._n;
	_a = lm._a;
	_b = lm._b;
	_v = lm._v;
	_h = lm._h;
	_bc = lm._bc;
	_cbc = lm._cbc;
	_discount = lm._discount;
	_strength = lm._strength;
	_lambda = lm._lambda;
	_length = lm._length;
	_f = lm._f;
	_base = lm._base;
	_cbase = lm._cbase;
	_cbase_add = lm._cbase_add;
	_cbase_remove = lm._cbase_remove;
	return *this;
}

hpyp::~hpyp() {
}

void hpyp::save(const char *file) {
	FILE *fp = NULL;
	if ((fp = fopen(file, "wb")) == NULL)
		throw "failed to open save file in hpyp::save";
	save(fp);
	fclose(fp);
}

void hpyp::save(FILE *fp) {
	if (fwrite(&_n, sizeof(int), 1, fp) != 1)
		throw "failed to write _n in hpyp::save";
	if (fwrite(&_a, sizeof(double), 1, fp) != 1)
		throw "failed to write _a in hpyp::save";
	if (fwrite(&_b, sizeof(double), 1, fp) != 1)
		throw "failed to write _b in hpyp::save";
	if (fwrite(&_v, sizeof(int), 1, fp) != 1)
		throw "failed to write _v in hpyp::save";
	if (fwrite(&(*_discount)[0], sizeof(double), _discount->size(), fp) != _discount->size())
		throw "failed to write _discount in hpyp::save";
	if (fwrite(&(*_strength)[0], sizeof(double), _strength->size(), fp) != _strength->size())
		throw "failed to write _strength in hpyp::save";
	int poisson_size = (_lambda)? _lambda->size() : 0;
	if (fwrite(&poisson_size, sizeof(int), 1, fp) != 1)
		throw "failed to write size of poisson_dist in hpyp::save";
	if (poisson_size != 0) {
		if (fwrite(&(*_lambda)[0], sizeof(double), _lambda->size(), fp) != _lambda->size())
			throw "failed to write _poisson in hpyp::save";
		if (fwrite(&_f, sizeof(int), 1, fp) != 1)
			throw "failed to write length_dist denominator in hpyp::save";
		int n = _length->size();
		if (fwrite(&n, sizeof(int), 1, fp) != 1)
			throw "failed to write number of length_dist in hpyp::save";
		if (fwrite(&(*_length)[0], sizeof(double), _length->size(), fp) != _length->size())
			throw "failed to write _length in hpyp::save";

	}
	_h->save(fp);
}

void hpyp::load(const char *file) {
	FILE *fp = NULL;
	if ((fp = fopen(file, "rb")) == NULL)
		throw "failed to open model file in hpyp::load";
	load(fp);
	fclose(fp);
}

void hpyp::load(FILE *fp) {
	if (fread(&_n, sizeof(int), 1, fp) != 1)
		throw "failed to read _n in hpyp::load";
	if (fread(&_a, sizeof(double), 1, fp) != 1)
		throw "failed to read _a in hpyp::load";
	if (fread(&_b, sizeof(double), 1, fp) != 1)
		throw "failed to read _b in hpyp::load";
	if (fread(&_v, sizeof(int), 1, fp) != 1)
		throw "failed to read _v in hpyp::load";
	_discount->resize(_n);
	_strength->resize(_n);
	if (fread(&(*_discount)[0], sizeof(double), _n, fp) != (size_t)_n)
		throw "failed to read _discount in hpyp::load";
	if (fread(&(*_strength)[0], sizeof(double), _n, fp) != (size_t)_n)
		throw "failed to read _strength in hpyp::load";
	int poisson_size = 0;
	if (fread(&poisson_size, sizeof(int), 1, fp) != 1)
		throw "failed to read size of poisson_ddist in hpyp::load";
	if (poisson_size) {
		_lambda = shared_ptr<vector<double> >(new vector<double>(poisson_size, 0));
		if (fread(&(*_lambda)[0], sizeof(double), poisson_size, fp) != (size_t)poisson_size)
			throw "failed to read _poisson in hpyp::load";
		if (fread(&_f, sizeof(int),1,fp) != 1)
			throw "failed to read length_dist denominator in hpyp::load";
		int n = 0;
		if (fread(&n, sizeof(int), 1, fp) != 1)
			throw "failed to number of length_dist in hpyp::load";
		_length = shared_ptr<vector<double> >(new vector<double>(n, 0.));
		if (fread(&(*_length)[0], sizeof(double), n, fp) != (size_t)n)
			throw "failed to read _length in hpyp::load";

	}
	_h->load(fp);
}

int hpyp::n() const {
	return _n;
}

long long hpyp::base_customers() const {
	long long n = 0;
	if (_bc) {
		for (auto it = _bc->cbegin(); it != _bc->cend(); ++it)
			n += (long long)it->second.size();
	}
	return n;
}

double hpyp::discount(int n) const {
	if (n < 0 || n >= _n) {
		throw "found access to out of range n >= _n: at discount param";
	}
	return (*_discount)[n];
}

double hpyp::strength(int n) const {
	if (n < 0 || n >= _n) {
		throw "found access to out of range n >= _n: at strength param";
	}
	return (*_strength)[n];
}

double hpyp::alpha(int n) const {
	return 0;
}

context* hpyp::find(nsentence& s, int i) const {
	context *h = _h.get();
	for (int m = 1; m < _n; ++m) {
		context *c = h->find(s[i-m]);
		if (c)
			h = c;
		else
			break;
	}
	return h;
}

context* hpyp::find(sentence& s, int i) const {
	context *h = _h.get(); // root
	for (int m = 1; m < _n; ++m) {
		context *c = h->find(s[i-m]);
		if (c)
			h = c;
		else
			break;
	}
	return h;
}

context* hpyp::find(chunk& b, int i) const {
	context *h = _h.get();
	for (int m = 1; m < _n; ++m) {
		context *c = h->find(b[i-m]);
		if (c)
			h = c;
		else
			break;
	}
	return h;
}

context* hpyp::find(word& w, int i) const {
	context *h = _h.get();
	for (int m = 1; m < _n; ++m) {
		context *c = h->find(w[i-m]);
		if (c)
			h = c;
		else
			break;
	}
	return h;
}

context* hpyp::make(nsentence& s, int i) {
	context *h = _h.get();
	for (int m = 1; m < _n; ++m) {
		h = h->make(s[i-m]);
	}
	return h;
}

context* hpyp::make(sentence& s, int i) {
	context *h = _h.get();//&(*_h);
	for (int m = 1; m < _n; ++m) {
		h = h->make(s[i-m]);
	}
	return h;
}

context* hpyp::make(chunk& b, int i) {
	context *h = _h.get();
	for (int m = 1; m < _n; ++m) {
		h = h->make(b[i-m]);
	}
	return h;
}

context* hpyp::make(word& w, int i) {
	context *h = _h.get();//&(*_h);
	for (int m = 1; m < _n; ++m) {
		h = h->make(w[i-m]);
	}
	return h;
}

void hpyp::set_base(lm *b) {
	_base = b;
}

void hpyp::set_cbase(std::function<double(chunk&)> f) {
	_cbase = f;
}

void hpyp::set_cbase_add(std::function<void(chunk&)> f) {
	_cbase_add = f;
}

void hpyp::set_cbase_remove(std::function<void(chunk&)> f) {
	_cbase_remove = f;
}

void hpyp::set_v(int v) {
	if (v < 1)
		throw "found invalid vocab setting";
	_v = v;
}

int hpyp::v() const {
	return _v;
}

double hpyp::pr(chunk& c, const context *h) {
	return exp(lp(c, h));
}

double hpyp::pr(word& w, const context *h) {
	return exp(lp(w, h));
}

double hpyp::pr(int k, const context *h) {
	return exp(lp(k, h));
}

double hpyp::lp(chunk& ch, const context *h) {
	if (!h) {
		return _lpb(ch);
	}
	//bool chk = false;
	//double lpr = _cache.get(ch, h, chk);
	double lpr = 0;
	/*
	if (chk)
		return lpr;
		*/
	double c = h->c();
	double t = h->t();
	double cu = h->cu(ch.id);
	double tu = h->tu(ch.id);
	int n = h->n();
	double a = cu-(*_discount)[n]*tu;
	double b = (*_strength)[n]+(*_discount)[n]*t;
	double d = (*_strength)[n]+c;
	if (a == 0.)
		lpr = log(b)+lp(ch,h->parent())-log(d);
	else
		lpr = math::lse(log(a)-log(d), log(b)+lp(ch, h->parent())-log(d));
	return lpr;
	//return _cache.set(ch, h, lpr);

}

double hpyp::lp_root_base(chunk& ch, double base) {
	// equivalent to lp(ch, _h.get()) but uses the supplied base as the
	// parent fallback (the root's parent is null -> _lpb(ch) == base).
	const context *h = _h.get();
	double c = h->c();
	double t = h->t();
	double cu = h->cu(ch.id);
	double tu = h->tu(ch.id);
	int n = h->n();
	double a = cu-(*_discount)[n]*tu;
	double b = (*_strength)[n]+(*_discount)[n]*t;
	double d = (*_strength)[n]+c;
	if (a == 0.)
		return log(b)+base-log(d);
	return math::lse(log(a)-log(d), log(b)+base-log(d));
}

double hpyp::lp_root_base(word& w, double base) {
	// word variant: base is the base-measure sum WITHOUT the poisson
	// correction; add _correct here so this matches _lpb(w) exactly, then
	// interpolate at the root (parent is null -> _lpb(w) == corrected base).
	double bs = base;
	if (_lambda != nullptr)
		bs += _correct(w);
	const context *h = _h.get();
	double c = h->c();
	double t = h->t();
	double cu = h->cu(w.id);
	double tu = h->tu(w.id);
	int n = h->n();
	double a = cu-(*_discount)[n]*tu;
	double b = (*_strength)[n]+(*_discount)[n]*t;
	double d = (*_strength)[n]+c;
	if (a == 0.)
		return log(b)+bs-log(d);
	return math::lse(log(a)-log(d), log(b)+bs-log(d));
}

double hpyp::wlp(word& w, const int *prev, int np) {
	// context walk identical to _lpb: follow prev[0], prev[1], ... down the
	// trie from the root, stopping at the first miss or after n-1 steps.
	const context *h = _h.get();
	for (int j = 0; j < np && j < _n-1; ++j) {
		const context *c = h->find(prev[j]);
		if (!c)
			break;
		h = c;
	}
	return lp(w, h);
}

double hpyp::wlp(int k, const int *prev, int np) {
	// int/letter variant of wlp: same context walk, virtual lp(int, ctx)
	// (dispatches to vpyp::lp for a variable-order letter model).
	const context *h = _h.get();
	for (int j = 0; j < np && j < _n-1; ++j) {
		const context *c = h->find(prev[j]);
		if (!c)
			break;
		h = c;
	}
	return lp(k, h);
}

double hpyp::lp(word& w, const context *h) {
	if (!h) {
		return _lpb(w);
	}
	bool chk = false;
	double lpr = 0;
	/*
	double lpr = _cache.get(w, h, chk);
	if (chk)
		return lpr;
		*/
	double c = h->c();
	double t = h->t();
	double cu = h->cu(w.id);
	double tu = h->tu(w.id);
	int n = h->n();
	double a = cu-(*_discount)[n]*tu;
	double b = (*_strength)[n]+(*_discount)[n]*t;
	double d = (*_strength)[n]+c;
	if (a == 0.)
		lpr = log(b)+lp(w,h->parent())-log(d);
	else
		lpr = math::lse(log(a)-log(d), log(b)+lp(w, h->parent())-log(d));
	//cout << "c:" << c << " t:" << t << " cu:" << cu << " tu:" << tu << " n:" << n << " stlength:" << (*_strength)[n] << " discount:" << (*_discount)[n] << " v:" << _v << " lp:" << lpr << endl;
	return lpr;
	//return _cache.set(w, h, lpr);
}

double hpyp::lp(int k, const context *h) {
	if (!h)
		return -log(_v);
	// log(1-h->v()/V) = log((V-h->v())/V) = log(V-h->v()) - log(V)
	/*
	if (!h && _h->cu(k) == 0 && _h->v() < _v) {
		return log(_v-_h->v())-log(_v);
	} else if (!h) {
		return -log(_v);
		//return log(_v-_h->v())-log(_v);
	}
	*/
	//bool chk = false;
	double lpr = 0;
	/*
	double lpr = _cache.get(k, h, chk);
	if (chk)
		return lpr;
		*/
	double c = h->c();
	double t = h->t();
	double cu = h->cu(k);
	double tu = h->tu(k);
	int n = h->n();
	double a = cu-(*_discount)[n]*tu;
	double b = (*_strength)[n]+(*_discount)[n]*t;
	double d = (*_strength)[n]+c;
	// log((a+b*pr)/d) = log(a+b*pr) - log d
	// a == 0
	// log(a+b*pr) - log d = log(b) + log(pr) - log d
	// otherwise
	// log(a/d + b*pr/d)
	// = logsumexp(log(a/d) + log(b*pr/d))
	// = logsumexp(log a - log d + log b + log pr - log d)
	if (a == 0.)
		lpr = log(b)+lp(k,h->parent())-log(d);
	else
		lpr = math::lse(log(a)-log(d), log(b)+lp(k, h->parent())-log(d));
	//cout << "c:" << c << " t:" << t << " cu:" << cu << " tu:" << tu << " n:" << n << " stlength:" << (*_strength)[n] << " discount:" << (*_discount)[n] << " v:" << _v << " lp:" << lpr << endl;
	return lpr;
	//return _cache.set(k, h, lpr);
}

context* hpyp::h() const {
	return _h.get();
}

double hpyp::_prb(chunk& c) const {
	return exp(_lpb(c));
}

double hpyp::_prb(word& w) const {
	return exp(_lpb(w));
}

double hpyp::_lpb(chunk& b) const {
	if (_cbase)
		return _cbase(b);
	if (!_base)
		return -log(_v);
	double lp = 0;
	for (int i = 0; i < b.len+1; ++i) {
		int n = _base->n();
		const context *h = _base->h();
		for (int j = 1; j < n; ++j) {
			const context *c = h->find(b[i-j]);
			if (!c)
				break;
			else
				h = c;
		}
		//lp += _base->lp(b[i], h);
		lp += _base->lp(b.wd(i), h);
	}
	return lp;
}

double hpyp::_lpb(word& w) const {
	if (!_base)
		return -log(_v);
	/*
	   if (!_base && _h->cu(w.id) == 0 && _h->v() < _v) {
	   return log(_v-_h->v())-log(_v);
	   } else if (!_base) {
	   return -log(_v);
	//return log(_v-_h->v())-log(_v);
	}
	   */
	double lp = 0;
	for (int i = 0; i < w.len+1; ++i) {
		int n = _base->n();
		const context *h = _base->h();
		for (int j = 1; j < n; ++j) {
			const context *c = h->find(w[i-j]);
			if (!c)
				break;
			else
				h = c;
		}
		lp += _base->lp(w[i], h);
	}
	// poisson correction
	if (_lambda != nullptr) {
		lp += _correct(w);
	}
	return lp;
}

double hpyp::_correct(word& w) const {
	/* use poisson correction for each word type */
	type t = wordtype::get(w);
	poisson_distribution po;
	double lp = po.lp((*_lambda)[t], w.len);
	if (w.len < (int)_length->size())
		lp -= (*_length)[w.len];
	else if (_f)
		lp += log(_f);
	return lp;
}

void hpyp::_sample(vector<unsigned int>& w) {
	unsigned int k = w[0];
	do {
		context *h = _base->h();
		for (int m = 1; m < _base->n() && (int)w.size()-m >= 0; ++m) {
			context *c = h->find(w[(int)w.size()-m]);
			if (c)
				h = c;
			else
				break;
		}
		k = h->sample(_base);
		if (k)
			w.push_back(k);
	} while ((w.size() == 1 && k == 0) || (k != 0 && w.size() < MAXLEN));
}

void hpyp::_estimate_length(int n) {
	if (_length == nullptr) {
		_length = shared_ptr<vector<double> >(new vector<double>);
		_f = 0;
	} else {
		_length->clear();
		_f = 0;
	}
	vector<vector<unsigned int> > k(n,vector<unsigned int>(1, 0));
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (auto i = 0; i < n; ++i) {
		_sample(k[i]);
	}
	for (auto i = 0; i < n; ++i) {
		word w(k[i], 1, (int)k[i].size()-1);
		if ((int)_length->size() <= w.len) {
			_length->resize(w.len+1, 1.); // init by 1: add one smoothing
			++_f;
		}
		(*_length)[w.len] += 1.;
		++_f;
	}
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (auto i = 0; i < (int)_length->size(); ++i) {
		if ((*_length)[i])
			(*_length)[i] = log((*_length)[i])-log(_f);
		else if (_f)
			(*_length)[i] = -log(_f);
	}
}

void hpyp::_estimate_poisson() {
	if (_lambda == nullptr) {
		_lambda = shared_ptr<vector<double> >(new vector<double>(chartype::n, 0));
		//_w = shared_ptr<vector<int> >(new vector<int>(chartype::n, 0));
	} //else {
	  //fill(_w->begin(), _w->end(), 0);
	  //}
	vector<double> a(chartype::n, POISSON_A);
	vector<double> b(chartype::n, POISSON_B);
	//_h->estimate_l(a, b, *_w, _bc.get());
	_h->estimate_l(a, b, _bc.get());
	shared_ptr<generator> g = generator::create();
	gamma_dist gm;
	//int z = 0;
	/*
#ifdef _OPENMP
#pragma omp parallel for reduction(+:z)
#endif
*/
	for (auto i = 0; i < chartype::n; ++i) {
		gamma_dist::param_type param(a[i], 1./b[i]);
		gm.param(param);
		(*_lambda)[i] = gm((*g)());
	}
}

bool hpyp::add(int k, context *h) {
	bool add_to_parent = false;
	while (h && (add_to_parent = h->add(k, this))) {
		h = h->parent();
		if (!h)
			++_v;
	}
	//_cache.clear();
	return add_to_parent;
}

void hpyp::_seat_base(word& w) {
	wrap::add_a(w, _base);
}

void hpyp::_seat_base(chunk& c) {
	if (_cbase_add)
		_cbase_add(c);
	else
		wrap::add_a(c, _base);
}

void hpyp::_unseat_base(word& w) {
	wrap::remove_a(w, _base);
}

void hpyp::_unseat_base(chunk& c) {
	if (_cbase_remove)
		_cbase_remove(c);
	else
		wrap::remove_a(c, _base);
}

template<class T>
void hpyp::_bc_add(shared_ptr<bcorpus<T> >& bc, T& x) {
	if (bc == nullptr)
		bc = shared_ptr<bcorpus<T> >(new bcorpus<T>);
	_seat_base(x);
	(*bc)[x.id].push_back(x);
}

template<class T>
void hpyp::_bc_remove(shared_ptr<bcorpus<T> >& bc, T& x) {
	int size = (*bc)[x.id].size();
	int id = (*generator::create())()()%size;
	T& b = (*bc)[x.id][id];
	_unseat_base(b);
	(*bc)[x.id].erase((*bc)[x.id].begin()+id);
	if ((*bc)[x.id].empty())
		bc->erase(x.id);
	if (bc && bc->empty())
		bc = nullptr;
}

template<class T>
void hpyp::_gibbs_impl(bcorpus<T>& bc, int iter) {
	for (int i = 0; i < iter; ++i) {
		for (auto it = bc.begin(); it != bc.end(); ++it) {
			int size = it->second.size();
			vector<int> rd(size, 0);
			rd::shuffle(rd.data(), size);
			for (int j = 0; j < size; ++j) {
				lock_guard<mutex> m(_mutex);
				_unseat_base(it->second[rd[j]]);
				_seat_base(it->second[rd[j]]);
			}
		}
	}
}

void hpyp::add(word& w, context *h) {
	if (add(w.id, h) && _base) {
		lock_guard<mutex> m(_mutex);
		_bc_add(_bc, w);
	}
	//_cache.clear();
}

void hpyp::add(chunk& c, context *h) {
	if (add(c.id, h) && _base) {
		lock_guard<mutex> m(_mutex);
		_bc_add(_cbc, c);
	}
	//_cache.clear();
}

bool hpyp::remove(int k, context *h) {
	bool remove_from_parent = false;
	while (h && (remove_from_parent = h->remove(k))){
		h = h->parent();
		if (!h)
			--_v;
	}
	//_cache.clear();
	return remove_from_parent;
}

bool hpyp::valid() const {
	return _h && _h->valid();
}

bool hpyp::empty() const {
	return _h && _h->empty();
}

void hpyp::remove(word& w, context *h) {
	if (remove(w.id, h) && _base) {
		lock_guard<mutex> m(_mutex);
		_bc_remove(_bc, w);
	}
	//_cache.clear();
}

void hpyp::remove(chunk& c, context *h) {
	if (remove(c.id, h) && _base) {
		lock_guard<mutex> m(_mutex);
		_bc_remove(_cbc, c);
	}
	//_cache.clear();
}

void hpyp::estimate(int iter) {
	_h->cleanup();
	beta_distribution be;
	gamma_dist gm;
	shared_ptr<generator> g = generator::create();
	for (int i = 0; i < iter; ++i) {
		lock_guard<mutex> m(_mutex);
		vector<double> discount_a(_n);
		vector<double> discount_b(_n);
		vector<double> strength_a(_n);
		vector<double> strength_b(_n);
		_h->estimate_d(discount_a, discount_b, this);
		_h->estimate_t(strength_a, strength_b, this);
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int j = 0; j < _n; ++j) {
			gamma_dist::param_type param(_a+strength_a[j], 1./(_b-strength_b[j]));
			gm.param(param);
			(*_discount)[j] = be(_a+discount_a[j],_b+discount_b[j]);
			(*_strength)[j] = gm((*g)());
		}
	}
	//_cache.clear();
}

void hpyp::poisson_correction(int n) {
	if (_bc != nullptr) {
		lock_guard<mutex> m(_mutex);
		_estimate_length(n);
		_estimate_poisson();
	}
}

void hpyp::gibbs(int iter) {
	if (!_base)
		return;
	if (_bc)
		_gibbs_impl(*_bc, iter);
	if (_cbc)
		_gibbs_impl(*_cbc, iter);
}

int hpyp::draw_k(context *h) {
	return h->sample(this);
}

int hpyp::draw_n(word& w, int i) {
	vector<double> table;
	int j = 1;
	double ln_pr_pass = 0;
	double ln_pr_stop = 0;
	double lp_cache = 0;
	context *c = h();
	do {
		if (c) {
			int s = c->stop();
			int p = c->pass();
			lp_cache = lp(w[i], c);
			ln_pr_stop = log(s) - log(s+p);
			ln_pr_pass += log(p) - log(s+p);
			c = c->find(w[i-j]);
		} else {
			ln_pr_stop = -log(2);
			ln_pr_pass += -log(2);
		}
		table.push_back(lp_cache+ln_pr_stop+ln_pr_pass);
	} while (i-j >= -1 && j++ < _n);
	return 1+rd::ln_draw(table);
}

int hpyp::draw_n(chunk& b, int i) {
	vector<double> table;
	int j = 1;
	double ln_pr_pass = 0;
	double ln_pr_stop = 0;
	double lp_cache = 0;
	context *c = h();
	do {
		if (c) {
			int s = c->stop();
			int p = c->pass();
			lp_cache = lp(b[i], c);
			ln_pr_stop = log(s) - log(s+p);
			ln_pr_pass += log(p) - log(s+p);
			c = c->find(b[i-j]);
		} else {
			ln_pr_stop = -log(2);
			ln_pr_pass += -log(2);
		}
		table.push_back(lp_cache+ln_pr_stop+ln_pr_pass);
	} while (i-j >= -1 && j++ < _n);
	return 1+rd::ln_draw(table);
}

int hpyp::draw_n(sentence& sq, int i) {
	vector<double> table;
	int j = 1;
	double ln_pr_pass = 0;
	double ln_pr_stop = 0;
	double lp_cache = 0;
	context *c = h();
	do {
		if (c) {
			int s = c->stop();
			int p = c->pass();
			lp_cache = lp(sq[i], c);
			ln_pr_stop = log(s) - log(s+p);
			ln_pr_pass += log(p) - log(s+p);
			c = c->find(sq[i-j]);
		} else {
			ln_pr_stop = -log(2);
			ln_pr_pass += -log(2);
		}
		table.push_back(lp_cache+ln_pr_stop+ln_pr_pass);
	} while (i-j >= -1 && j++ < _n);
	return 1+rd::ln_draw(table);
}

int hpyp::draw_n(nsentence& sq, int i) {
	vector<double> table;
	int j = 1;
	double ln_pr_pass = 0;
	double ln_pr_stop = 0;
	double lp_cache = 0;
	context *c = h();
	do {
		if (c) {
			int s = c->stop();
			int p = c->pass();
			lp_cache = lp(sq[i], c);
			ln_pr_stop = log(s) - log(s+p);
			ln_pr_pass += log(p) - log(s+p);
			c = c->find(sq[i-j]);
		} else {
			ln_pr_stop = -log(2);
			ln_pr_pass += -log(2);
		}
		table.push_back(lp_cache+ln_pr_stop+ln_pr_pass);
	} while (i-j >= -1 && j++ < _n);
	return 1+rd::ln_draw(table);
}

context* hpyp::find(word& w, int i, int n) const {
	context *c = h();
	for (int m = 1; m < n; ++m) {
		context *d = c->find(w[i-m]);
		if (d)
			c = d;
		else
			break;
	}
	return c;
}

context* hpyp::find(chunk& b, int i, int n) const {
	context *c = h();
	for (int m = 1; m < n; ++m) {
		context *d = c->find(b[i-m]);
		if (d)
			c = d;
		else
			break;
	}
	return c;
}

context* hpyp::find(sentence& s, int i, int n) const {
	context *c = h();
	for (int m = 1; m < n; ++m) {
		context *d = c->find(s[i-m]);
		if (d)
			c = d;
		else
			break;
	}
	return c;
}

context* hpyp::find(nsentence& s, int i, int n) const {
	context *c = h();
	for (int m = 1; m < n; ++m) {
		context *d = c->find(s[i-m]);
		if (d)
			c = d;
		else
			break;
	}
	return c;
}

context* hpyp::make(word& w, int i, int n) {
	context *c = h();
	for (int m = 1; m < n; ++m) {
		c = c->make(w[i-m]);
	}
	return c;
}

context* hpyp::make(chunk& b, int i, int n) {
	context *c = h();
	for (int m = 1; m < n; ++m) {
		c = c->make(b[i-m]);
	}
	return c;
}

context* hpyp::make(sentence& s, int i, int n) {
	context *c = h();
	for (int m = 1; m < n; ++m) {
		c = c->make(s[i-m]);
	}
	return c;
}

context* hpyp::make(nsentence& s, int i, int n) {
	context *c = h();
	for (int m = 1; m < n; ++m) {
		c = c->make(s[i-m]);
	}
	return c;
}
