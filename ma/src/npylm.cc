#include"npylm.h"
#include"convinience.h"
#include"vtable.h"
#include"rd.h"
#include"negative_binomial.h"
#include"wordtype.h"
#ifdef _OPENMP
#include<omp.h>
#endif

#define C 1
// _v is learned from the data, so no vocabulary seed is applied.
using namespace std;
using namespace npbnlp;

static unordered_map<int, int> freq;
static negative_binomial nb;

npylm::npylm():_n(2),_word(new hpyp(_n)),_letter(new vpyp(10))/*, _prior(0), _change(1), _len(1)*/ {
	_word->set_base(_letter.get());
	// type change prior
	/*
	beta_distribution be;
	_prior = 1.-be(_change, _len);
	*/
}

npylm::npylm(int n, int m): _n(n),_word(new hpyp(n)),_letter(new vpyp(m))/*, _prior(0), _change(1), _len(1)*/ {
	_word->set_base(_letter.get());
	// type change prior
	/*
	beta_distribution be;
	_prior = 1.-be(_change, _len);
	*/
}

npylm::~npylm() {
}

/*
void npylm::set(int v) {
	if (v > 0)
		_letter->set_v(v);
}
*/

int npylm::n() {
	return _n;
}

int npylm::m() {
	return _letter->n();
}

void npylm::add(sentence& s) {
	lock_guard<mutex> m(_mutex);
	shared_ptr<wid> dic = wid::create();
	for (auto i = 0; i < s.size(); ++i) {
		word& w = s.wd(i);
		if ((*dic)[w] == 1) { // unk
			w.id = dic->index(w);
		} else {
			w.id = (*dic)[w];
		}
		freq[w.id]++;
		/*
		type t = chartype::get(w[0]);
		for (auto j = 0; j < w.len; ++j) {
			type u = chartype::get(w[j]);
			if (t != u)
				++_change;
			t = u;
		}
		_len += w.len;
		*/
	}
	wrap::add_a(s, _word.get());
}

void npylm::remove(sentence& s) {
	lock_guard<mutex> m(_mutex);
	shared_ptr<wid> dic = wid::create();
	for (auto i = 0; i < s.size(); ++i) {
		word& w = s.wd(i);
		freq[w.id]--;
		if (freq[w.id] == 0) {
			dic->remove(w);
			freq.erase(w.id);
		}
		/*
		type t = chartype::get(w[0]);
		for (auto j = 0; j < w.len; ++j) {
			type u = chartype::get(w[j]);
			if (t != u)
				--_change;
			t = u;
		}
		_len -= w.len;
		*/
	}
	wrap::remove_a(s, _word.get());
}

void npylm::estimate(int iter) {
	// type change prior
	/*
	beta_distribution be;
	_prior = 1.-be(_change, _len);
	*/
	_word->gibbs(iter);
	_word->estimate(iter);
	_letter->estimate(iter);
}

void npylm::poisson_correction(int n) {
	_word->poisson_correction(n);
}

void npylm::save(const char *file) {
	FILE *fp = NULL;
	if ((fp = fopen(file, "wb")) == NULL)
		throw "failed to open save file in npylm::save";
	try {
		_word->save(fp);
		_letter->save(fp);
		/*
		if (fwrite(&_prior, sizeof(double), 1, fp) != 1)
			throw "failed to save type change prior in npylm::save";
			*/
	} catch (const char *ex) {
		throw ex;
	}
	fclose(fp);
}

void npylm::load(const char *file) {
	FILE *fp = NULL;
	if ((fp = fopen(file, "rb")) == NULL)
		throw "failed to open model file in npylm::load";
	try {
		_word->load(fp);
		_letter->load(fp);
		_n = _word->n();
		/*
		if (fread(&_prior, sizeof(double), 1, fp) != 1)
			throw "failed to read type change prior in npylm::load";
			*/
	} catch (const char *ex) {
		throw ex;
	}
	fclose(fp);
}

sentence npylm::sample(io& f, int i) {
	lattice l(f, i);
	//_type_prior(l);
	vt dp;
	// forward filtering backward sampling
	for (auto t = 0; t < (int)l.w.size(); ++t) {
		for (auto j = 0; j < l.size(t); ++j) {
			const context *c = _word->h();
			word& w = l.wd(t, j+1);
			//double ln_prior = l.prior[t][j];
			for (auto k = 0; k < l.size(t-w.len); ++k) {
				word& prev = l.wd(t-w.len, k+1);
				const context *h = NULL;
				if (_n > 1)
					h = c->find(prev.id);
				if (h)
					_forward(l, t-w.len-prev.len, h, /*ln_prior,*/ w, prev, dp[t][j], dp[t-w.len][k], _n-1, false);
				else
					_forward(l, t-w.len-prev.len, c, /*ln_prior,*/ w, prev, dp[t][j], dp[t-w.len][k], _n-1, true);
			}
		}
	}
	sentence s;
	word *w = l.wp(l.w.size(), 1); // eos
	int t = (int)l.w.size()-w->len;
	while (t >= 0) {
		const context *c = _word->h();
		vector<double> table(l.size(t), 1.);
		for (auto k = 0; k < l.size(t); ++k) {
			word& prev = l.wd(t, k+1);
			const context *h = NULL;
			if (_n > 1)
				h = c->find(prev.id);
			if (h)
				_backward(l, t-prev.len, h, *w, table[k], dp[t][k], _n-1, false);
			else
				_backward(l, t-prev.len, c, *w, table[k], dp[t][k], _n-1, true);
		}
		int len = 1+rd::ln_draw(table);
		w = l.wp(t, len);
		s.w.push_back(*w);//l.wd(t, len));
		t -= len;
	}
	reverse(s.w.begin(), s.w.end());
	s.n.resize(s.w.size(), 0);
	return s;
}

sentence npylm::parse(io& f, int i) {
	lattice l(f, i);
	vt dp;
	//_type_prior(l);
	for (auto t = 0; t < (int)l.w.size(); ++t) {
		for (auto j = 0; j < l.size(t); ++j) {
			const context *c = _word->h();
			word& w = l.wd(t, j+1);
			//double ln_prior = l.prior[t][j];
			for (auto k = 0; k < l.size(t-w.len); ++k) {
				word& prev = l.wd(t-w.len, k+1);
				const context *h = NULL;
				if (_n > 1)
					h = c->find(prev.id);
				if (h)
					_forward(l, t-w.len-prev.len, h, /*ln_prior,*/ w, prev, dp[t][j], dp[t-w.len][k], _n-1, false);
				else
					_forward(l, t-w.len-prev.len, c, /*ln_prior,*/ w, prev, dp[t][j], dp[t-w.len][k], _n-1, true);
			}
		}
	}
	sentence s;
	word *w = l.wp(l.w.size(), 1);
	int t = (int)l.w.size()-w->len;
	while (t >= 0) {
		const context *c = _word->h();
		vector<double> table(l.size(t), 1.);
		for (auto k = 0; k < l.size(t); ++k) {
			word& prev = l.wd(t, k+1);
			const context *h = NULL;
			if (_n > 1)
				h = c->find(prev.id);
			if (h)
				_backward(l, t-prev.len, h, *w, table[k], dp[t][k], _n-1, false);
			else
				_backward(l, t-prev.len, c, *w, table[k], dp[t][k], _n-1, true);
		}
		int len = 1+rd::best(table);
		w = l.wp(t, len);
		s.w.push_back(*w);
		t -= len;
	}
	reverse(s.w.begin(), s.w.end());
	s.n.resize(s.w.size(), 0);
	return s;
}

void npylm::_forward(lattice& l, int i, const context *c, /*double& ln_prior,*/ word& w, word& p, vt& a, vt& b, int n, bool unk) {
	if (n <= 1) {
		a.v = math::lse(a.v, b.v+_word->lp(w, c)/*+ln_prior*/, !a.is_init());
		if (!a.is_init()) // initialized
			a.set(true);
	} else {
		for (auto j = 0; j < l.size(i); ++j) {
			word& prev = l.wd(i, j+1);
			const context *h = NULL;
			if (!unk)
				h = c->find(prev.id);
			if (h)
				_forward(l, i-prev.len, h, /*ln_prior,*/ w, prev, a[p.len-1], b[j], n-1, false);
			else
				_forward(l, i-prev.len, c, /*ln_prior,*/ w, prev, a[p.len-1], b[j], n-1, true);
		}
	}
}

void npylm::_backward(lattice& l, int i, const context *c, word& w, double& lpr, vt& b, int n, bool unk) {
	if (n <= 1) {
		lpr = math::lse(lpr, b.v+_word->lp(w, c), (lpr == 1.));
	} else {
		for (auto j = 0; j < l.size(i); ++j) {
			word& prev = l.wd(i, j+1);
			const context *h = NULL;
			if (!unk)
				h = c->find(prev.id);
			if (h)
				_backward(l, i-prev.len, h, w, lpr, b[j], n-1, false);
			else
				_backward(l, i-prev.len, c, w, lpr, b[j], n-1, true);
		}
	}
}

/*
void npylm::_type_prior(lattice& l) {
	l.prior.resize(l.w.size());
	for (auto t = 0; t < (int)l.w.size(); ++t) {
		l.prior[t].resize(l.size(t), 0);
		for (auto j = 0; j < l.size(t); ++j) {
			word& w = l.wd(t, j+1);
			type tp = chartype::get(w[0]);
			int change = 0;
			for (auto k = 1; k < w.len; ++k) {
				type u = chartype::get(w[k]);
				if (tp != u)
					++change;
				tp = u;
			}
			//double pr = nb.density(_prior, w.len-change, change);
			l.prior[t][j] = log(nb.density(_prior, w.len-change, change));
			//assert(l.prior[t][j] <= 0.);
		}
	}
}
*/
