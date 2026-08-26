#include"nnpylm.h"
#include"convinience.h"
#include"vtable.h"
#include"rd.h"
#include"wordtype.h"
#include"chunktype.h"
#include"negative_binomial.h"
#ifdef _OPENMP
#include<omp.h>
#endif

#define C 1
// _v is learned from the data, so no vocabulary seed is applied.
#define A 1.
#define B 1.
#define CHUNK_CDF_TH 0.999
#define L 50
#define ZERO 1e-16
#define LZERO log(ZERO)

using namespace std;
using namespace npbnlp;

static unordered_map<int, int> freq;
static negative_binomial nb;

//nnpylm::nnpylm(int n, npylm *lm): _n(n), _lm(lm), _chunk(new hpyp(_n)), _word(lm->_word), _letter(lm->_letter) {
nnpylm::nnpylm(int n, int m, int l): _n(n),  _chunk(new hpyp(_n)), _word(new hpyp(m)), _letter(new vpyp(l)), _num(new vector<int>(chunktype2::n, 0)), _change(new vector<int>(chunktype2::n, 0)), _len(new vector<int>(chunktype2::n, 0)), _lprior(new vector<double>(chunktype2::n, 0)), _cprior(new vector<double>(chunktype2::n, 0)), _chsize(new vector<int>(chunktype2::n, 0)) {
	_chunk->set_base(_word.get());
	_word->set_base(_letter.get());
	beta_distribution be;
	for (auto& p : *_lprior) {
		p = 1.-be(A, B);
	}
	for (auto& p : *_cprior) {
		p = 1.-be(A, B);
	}
	for (auto k = 0; k < chunktype2::n; ++k) {
		double cdf = 0;
		int l = 1;
		for (; cdf < CHUNK_CDF_TH && l < L; ++l) {
			cdf = nb.cdf((*_lprior)[k], 1, l-1);
		}
		(*_chsize)[k] = l;
	}
	//_prior = 1.-be(_change, _len);
}

nnpylm::nnpylm():_n(1), _chunk(new hpyp(_n)), _word(new hpyp(2)), _letter(new vpyp(10)), _num(new vector<int>(chunktype2::n, 0)), _change(new vector<int>(chunktype2::n, 0)), _len(new vector<int>(chunktype2::n, 0)), _lprior(new vector<double>(chunktype2::n, 0)), _cprior(new vector<double>(chunktype2::n, 0)), _chsize(new vector<int>(chunktype2::n, 0)) {
	_chunk->set_base(_word.get());
	_word->set_base(_letter.get());
	beta_distribution be;
	for (auto& p : *_lprior) {
		p = 1.-be(A, B);
	}
	for (auto& p : *_cprior) {
		p = 1.-be(A, B);
	}
	for (auto k = 0; k < chunktype2::n; ++k) {
		double cdf = 0;
		int l = 1;
		for (; cdf < CHUNK_CDF_TH && l < L; ++l) {
			cdf = nb.cdf((*_lprior)[k], 1, l-1);
		}
		(*_chsize)[k] = l;
	}
	//_prior = 1.-be(_change, _len);
}

nnpylm::~nnpylm() {
}

void nnpylm::add(nsentence& s) {
	lock_guard<mutex> m(_mutex);
	shared_ptr<cid> dic = cid::create();
	for (auto i = 0; i < s.size(); ++i) {
		chunk& c = s.ch(i);
		if ((*dic)[c] == 1) { // unk
			c.id = dic->index(c);
		} else {
			c.id = (*dic)[c];
		}
		freq[c.id]++;
		if (c.type < 0)
			continue;
		int change = 0;
		type t = wordtype::get(c.wd(0));
		for (auto j = 0; j < c.len; ++j) {
			type u = wordtype::get(c.wd(j));
			if (t != u)
				++change;
			t = u;
		}
		(*_num)[c.type] += c.len-1;
		(*_len)[c.type] += c.len;
		(*_change)[c.type] += change;
	}
	wrap::add_a(s, _chunk.get());
}

void nnpylm::remove(nsentence& s) {
	lock_guard<mutex> m(_mutex);
	shared_ptr<cid> dic = cid::create();
	for (auto i = 0; i < s.size(); ++i) {
		chunk& c = s.ch(i);
		freq[c.id]--;
		if (freq[c.id] == 0) {
			dic->remove(c);
			freq.erase(c.id);
		}
		if (c.type < 0)
			continue;
		int change = 0;
		type t = wordtype::get(c.wd(0));
		for (auto j = 0; j < c.len; ++j) {
			type u = wordtype::get(c.wd(j));
			if (t != u)
				++change;
			t = u;
		}
		(*_num)[c.type] -= c.len-1;
		(*_len)[c.type] -= c.len;
		(*_change)[c.type] -= change;
	}
	wrap::remove_a(s, _chunk.get());
}

void nnpylm::estimate(int iter) {
	_chunk->gibbs(iter);
	_word->gibbs(iter);
	_chunk->estimate(iter);
	_word->estimate(iter);
	_letter->estimate(iter);
	beta_distribution be;
	for (auto t = 0; t < chunktype2::n; ++t) {
		(*_lprior)[t] = 1.-be(A+(*_num)[t], B+(*_len)[t]);
		(*_cprior)[t] = 1.-be(A+(*_change)[t], B+(*_len)[t]);
	}
	//_prior = 1.-be(_change, _len);
	for (auto k = 0; k < chunktype2::n; ++k) {
		double cdf = 0;
		int l = 1;
		for (; cdf < CHUNK_CDF_TH && l < L; ++l) {
			cdf = nb.cdf((*_lprior)[k], 1, l-1);
		}
		(*_chsize)[k] = l;
	}
}

void nnpylm::poisson_correction(int n) {
	_word->poisson_correction(n);
}

/*
void nnpylm::set(int v) {
	if (v > 0)
		_letter->set_v(v);
}
*/

int nnpylm::n() {
	return _n;
}

int nnpylm::m() {
	return _word->n();
}

int nnpylm::l() {
	return _letter->n();
}

void nnpylm::save(const char *file) {
	FILE *fp = NULL;
	if ((fp = fopen(file, "wb")) == NULL)
		throw "failed to open save file in nnpylm::save";
	try {
		_chunk->save(fp);
		_word->save(fp);
		_letter->save(fp);
		int n = _lprior->size();
		if (fwrite(&n, sizeof(int), 1, fp) != 1)
			throw "failed to save prior class num in nnpylm::save";
		if (fwrite(&_lprior, sizeof(double), n, fp) != n)
			throw "failed to save type duration prior in nnpylm::save";
		if (fwrite(&_cprior, sizeof(double), n, fp) != n)
			throw "failed to save type change prior in nnpylm::save";
	} catch (const char *ex) {
		throw ex;
	}
	fclose(fp);
}

void nnpylm::load(const char *file) {
	FILE *fp = NULL;
	if ((fp = fopen(file, "rb")) == NULL)
		throw "failed to open model file in nnpylm::load";
	try {
		_chunk->load(fp);
		_word->load(fp);
		_letter->load(fp);
		_n = _chunk->n();
		int n = 0;
		if (fread(&n, sizeof(int), 1, fp) != 1)
			throw "failed to read prior class num in nnpylm::load";
		if (fread(&_lprior, sizeof(double), n, fp) != n)
			throw "failed to load type duration prior in nnpylm::load";
		if (fread(&_cprior, sizeof(double), n, fp) != n)
			throw "failed to load type change prior in nnpylm::load";
		for (auto k = 0; k < chunktype2::n; ++k) {
			double cdf = 0;
			int l = 1;
			for (; cdf < CHUNK_CDF_TH && l < L; ++l) {
				cdf = nb.cdf((*_lprior)[k], 1, l-1);
			}
			(*_chsize)[k] = l;
		}
	} catch (const char *ex) {
		throw ex;
	}
}

nsentence nnpylm::sample(nio& f, int i) {
	clattice2 l(f, i, *_chsize);
	vt dp;
	_type_prior(l);
	for (auto t = 0; t < (int)l.c.size(); ++t) {
		for (auto j = 0; j < l.size(t); ++j) {
			const context *c = _chunk->h();
			chunk& ch = l.ch(t, j+1);
			double ln_prior = l.prior[t][j];
			for (auto k = 0; k < l.size(t-ch.len); ++k) {
				chunk& prev = l.ch(t-ch.len, k+1);
				const context *h = NULL;
				if (_n > 1)
					h = c->find(prev.id);
				if (h)
					_forward(l, t-ch.len-prev.len, h, ln_prior, ch, prev, dp[t][j], dp[t-ch.len][k], _n-1, false);
				else
					_forward(l, t-ch.len-prev.len, c, ln_prior, ch, prev, dp[t][j], dp[t-ch.len][k], _n-1, true);
			}
		}
	}
	nsentence s;
	chunk *ch = l.cp(l.c.size(), 1); // eos
	int t = (int)l.c.size()-ch->len;
	while (t >= 0) {
		const context *c = _chunk->h();
		vector<double> table(l.size(t), 1.);
		for (auto k = 0; k < l.size(t); ++k) {
			chunk& prev = l.ch(t, k+1);
			const context *h = NULL;
			if (_n > 1)
				h = c->find(prev.id);
			if (h)
				_backward(l, t-prev.len, h, *ch, table[k], dp[t][k], _n-1, false);
			else
				_backward(l, t-prev.len, c, *ch, table[k], dp[t][k], _n-1, true);
		}
		int len = 1+rd::ln_draw(table);
		ch = l.cp(t, len);
		s.c.push_back(*ch);
		t -= len;
	}
	reverse(s.c.begin(), s.c.end());
	s.n.resize(s.c.size(), 0);
	return s;
}

nsentence nnpylm::parse(nio& f, int i) {
	clattice2 l(f, i, *_chsize);
	vt dp;
	_type_prior(l);
	for (auto t = 0; t < (int)l.c.size(); ++t) {
		for (auto j = 0; j < l.size(t); ++j) {
			const context *c = _chunk->h();
			chunk& ch = l.ch(t, j+1);
			double ln_prior = l.prior[t][j];
			for (auto k = 0; k < l.size(t-ch.len); ++k) {
				chunk& prev = l.ch(t-ch.len, k+1);
				const context *h = NULL;
				if (_n > 1)
					h = c->find(prev.id);
				if (h)
					_forward(l, t-ch.len-prev.len, h, ln_prior, ch, prev, dp[t][j], dp[t-ch.len][k], _n-1, false);
				else
					_forward(l, t-ch.len-prev.len, c, ln_prior, ch, prev, dp[t][j], dp[t-ch.len][k], _n-1, true);
			}
		}
	}
	nsentence s;
	chunk *ch = l.cp(l.c.size(), 1); // eos
	int t = (int)l.c.size()-ch->len;
	while (t >= 0) {
		const context *c = _chunk->h();
		vector<double> table(l.size(t), 1.);
		for (auto k = 0; k < l.size(t); ++k) {
			chunk& prev = l.ch(t, k+1);
			const context *h = NULL;
			if (_n > 1)
				h = c->find(prev.id);
			if (h)
				_backward(l, t-prev.len, h, *ch, table[k], dp[t][k], _n-1, false);
			else
				_backward(l, t-prev.len, c, *ch, table[k], dp[t][k], _n-1, true);
		}
		int len = 1+rd::best(table);
		ch = l.cp(t, len);
		s.c.push_back(*ch);
		t -= len;
	}
	reverse(s.c.begin(), s.c.end());
	s.n.resize(s.c.size(), 0);
	return s;
}

void nnpylm::_forward(clattice2& l, int i, const context *c, double& ln_prior, chunk& ch, chunk& p, vt& a, vt& b, int n, bool unk) {
	if (n <= 1) {
		a.v = math::lse(a.v, b.v+_chunk->lp(ch, c)+ln_prior, !a.is_init());
		if (!a.is_init())
			a.set(true);
	} else {
		for (auto j = 0; j < l.size(i); ++j) {
			chunk& prev = l.ch(i, j+1);
			const context *h = NULL;
			if (!unk)
				h = c->find(prev.id);
			if (h)
				_forward(l, i-prev.len, h, ln_prior, ch, prev, a[p.len-1], b[j], n-1, false);
			else
				_forward(l, i-prev.len, c, ln_prior, ch, prev, a[p.len-1], b[j], n-1, true);
		}
	}
}

void nnpylm::_backward(clattice2& l, int i, const context *c, chunk& ch, double& lpr, vt& b, int n, bool unk) {
	if (n <= 1) {
		lpr = math::lse(lpr, b.v+_chunk->lp(ch, c), (lpr == 1.));
	} else {
		for (auto j = 0; j < l.size(i); ++j) {
			chunk& prev = l.ch(i, j+1);
			const context *h = NULL;
			if (!unk)
				h = c->find(prev.id);
			if (h)
				_backward(l, i-prev.len, h, ch, lpr, b[j], n-1, false);
			else
				_backward(l, i-prev.len, c, ch, lpr, b[j], n-1, true);
		}
	}
}

void nnpylm::_type_prior(clattice2& l) {
	l.prior.resize(l.c.size());
	for (auto t = 0; t < (int)l.c.size(); ++t) {
		l.prior[t].resize(l.c[t].size(), 0);
		for (auto i = 0; i < (int)l.c[t].size(); ++i) {
			chunk& c = l.c[t][i];
			int change = 0;
			int len = 0;
			type tp = wordtype::get(c.wd(0));
			for (auto j = 0; j < c.len; ++j) {
				type u = wordtype::get(c.wd(j));
				if (tp != u)
					++change;
				tp = u;
			}
			double pr_dur = max(ZERO, nb.density((*_lprior)[c.type], 1, c.len-1));
			double pr_chg = max(ZERO, nb.density((*_cprior)[c.type], c.len-change, change));
			l.prior[t][i] = log(pr_dur)+log(pr_chg);
			//l.prior[t][i] = log(nb.density((*_lprior)[c.type], 1, c.len-1)) + log(nb.density((*_cprior)[c.type], c.len-change, change));
			// duration only
			//l.prior[t][i] = log(nb.density((*_lprior)[c.type], 1, c.len-1));
			// type change only
			//l.prior[t][i] = log(nb.density((*_cprior)[c.type], len-change, change));
		}
	}
}
