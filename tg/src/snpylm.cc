#include"snpylm.h"
#include"convinience.h"
#include"rd.h"
#include"beta.h"
#include"generator.h"
#include<cmath>
#include<cstdlib>
#include<cstdio>
#include<deque>
#include<random>

#define SVOCAB 3000
#define SGAMMA 10.0
#define SALPHA 1.0
#define LAMBDA_A 2.0
#define LAMBDA_B 1.0

using namespace std;
using namespace npbnlp;

using gamma_dist = gamma_distribution<double>;

// Persistent backing store for the synthetic NE-symbol spellings. The wid
// dictionary keeps word keys whose _doc points into these buffers, so the
// storage must outlive the dictionary; a deque never relocates its elements,
// keeping every _doc pointer valid for the program's lifetime.
static deque<vector<unsigned int> > ne_bufs;

snpylm::snpylm(): snpylm(2, 1, 8, 10) {
}

snpylm::snpylm(int n, int hn, int hl, int k):
	_n(n < 2 ? 2 : n), _hn(hn < 1 ? 1 : hn), _hl(hl), _k(k), _v(SVOCAB),
	_gamma(SGAMMA), _alpha(SALPHA), _pi(1.0/(1.0+SGAMMA)),
	_bg(new hpyp(_n)), _spell(new vpyp(_hl)),
	_H(new vector<shared_ptr<hpyp> >), _Hletter(new vector<shared_ptr<vpyp> >),
	_pine(0), _piw(0), _pieos(0) {
	_spell->set_v(_v);
	_nek.assign(1, 0);
	_rho.assign(1, 0);
	_lambda.assign(1, 0);
	_necnt.assign(1, 0);
	_nelen.assign(1, 0);
	for (int i = 0; i <= _k; ++i) {
		_Hletter->push_back(shared_ptr<vpyp>(new vpyp(_hl)));
		(*_Hletter)[i]->set_v(_v);
		_H->push_back(shared_ptr<hpyp>(new hpyp(_hn)));
		(*_H)[i]->set_base((*_Hletter)[i].get());
		if (i > 0) {
			_rho.push_back(0);
			_lambda.push_back(LAMBDA_A);
			_necnt.push_back(0);
			_nelen.push_back(0);
		}
	}
	_install_cbase();
	_register_symbols();
}

snpylm::~snpylm() {
}

// G^bg is a chunk-keyed hpyp whose base is the G0 mixture. _spell is passed as
// the non-null _base only to satisfy hpyp's `&& _base` guard on chunk seating;
// with _cbase installed hpyp never walks it (see _lpb(chunk)/_seat_base(chunk)).
void snpylm::_install_cbase() {
	_bg->set_base(_spell.get());
	_bg->set_cbase([this](chunk& c) { return _g0_lp(c); });
	_bg->set_cbase_add([this](chunk& c) { _g0_add(c); });
	_bg->set_cbase_remove([this](chunk& c) { _g0_remove(c); });
}

void snpylm::_register_symbols() {
	shared_ptr<wid> d = wid::create();
	if ((int)_nek.size() < _k+1)
		_nek.resize(_k+1, 0);
	for (int k = 1; k <= _k; ++k) {
		if (_nek[k] > 0)
			continue;
		ne_bufs.emplace_back();
		vector<unsigned int>& buf = ne_bufs.back();
		buf.push_back(0x01u);
		buf.push_back((unsigned int)'N');
		buf.push_back((unsigned int)'E');
		buf.push_back(0x0E0000u + (unsigned int)k); // private high plane, unique per k
		word w(buf, 0, (int)buf.size());
		int id = d->index(w);
		_nek[k] = id;
		_id2k[id] = k;
	}
}

void snpylm::_resize(int k) {
	int target = k+1;
	while ((int)_H->size() < target) {
		int idx = (int)_H->size();
		_Hletter->push_back(shared_ptr<vpyp>(new vpyp(_hl)));
		(*_Hletter)[idx]->set_v(_v);
		_H->push_back(shared_ptr<hpyp>(new hpyp(_hn)));
		(*_H)[idx]->set_base((*_Hletter)[idx].get());
	}
	while ((int)_rho.size() < target) _rho.push_back(0);
	while ((int)_lambda.size() < target) _lambda.push_back(LAMBDA_A);
	while ((int)_necnt.size() < target) _necnt.push_back(0);
	while ((int)_nelen.size() < target) _nelen.push_back(0);
	if (_k < k)
		_k = k;
	_register_symbols();
}

void snpylm::set(int v, int k) {
	_v = v;
	_spell->set_v(_v);
	for (auto it = _Hletter->begin(); it != _Hletter->end(); ++it)
		(*it)->set_v(_v);
	if (k > _k)
		_resize(k);
}

void snpylm::set_gamma(double g) {
	if (g > 0) {
		_gamma = g;
		_pi = 1.0/(1.0+_gamma);
	}
}

void snpylm::set_alpha(double a) {
	if (a > 0)
		_alpha = a;
}

int snpylm::n() const {
	return _n;
}

int snpylm::k() const {
	return _k;
}

// >0 : NE class id; 0 : normal (spelling) word; -1 : reserved BOS/EOS/unk.
int snpylm::_kind(int id) const {
	if (id <= 1)
		return -1;
	auto it = _id2k.find(id);
	if (it != _id2k.end())
		return it->second;
	return 0;
}

int snpylm::_tvid(chunk& ch) {
	if (ch.k >= 1)
		return _nek[ch.k];
	word& w = ch.wd(0);
	if (w.id <= 1) {
		shared_ptr<wid> d = wid::create();
		w.id = d->index(w);
	}
	return w.id;
}

// log G0^spell(w): the NPYLM word base measure evaluated on the shared letter
// VPYP _spell (mirrors hpyp::_lpb(word) for a vpyp base measure exactly).
double snpylm::_spell_lp(word& w) {
	double lp = 0;
	int nn = _spell->n();
	for (int i = 0; i < w.len+1; ++i) {
		const context *h = _spell->h();
		for (int j = 1; j < nn; ++j) {
			const context *c = h->find(w[i-j]);
			if (!c)
				break;
			h = c;
		}
		lp += _spell->lp((int)w[i], h);
	}
	return lp;
}

// log G0(v). NE weight uses the DP predictive rho_k = (m_k + alpha/K)/(m. + alpha)
// which sums to 1 over the K active classes, so G0 stays a proper mixture.
double snpylm::_g0_lp(chunk& tv) {
	int kind = _kind(tv.id);
	if (kind > 0) {
		double denom = (double)_pine + _alpha;
		double num = (double)_rho[kind] + _alpha/(double)(_k > 0 ? _k : 1);
		return log(_pi) + log(num) - log(denom);
	} else if (kind == 0) {
		return log(1.0 - _pi) + _spell_lp(tv.wd(0));
	}
	return -log((double)_v); // reserved token: neutral uniform
}

// base escape (new root table in G^bg): seat the spelling / bump the switching
// counters. Fires exactly once per base customer, the correct counting site.
void snpylm::_g0_add(chunk& tv) {
	int kind = _kind(tv.id);
	if (kind > 0) {
		if (kind >= (int)_rho.size())
			_rho.resize(kind+1, 0);
		++_pine;
		++_rho[kind];
	} else if (kind == 0) {
		wrap::add_v(tv.wd(0), _spell.get());
		++_piw;
	} else {
		++_pieos;
	}
}

void snpylm::_g0_remove(chunk& tv) {
	int kind = _kind(tv.id);
	if (kind > 0) {
		--_pine;
		--_rho[kind];
	} else if (kind == 0) {
		wrap::remove_v(tv.wd(0), _spell.get());
		--_piw;
	} else {
		--_pieos;
	}
}

void snpylm::init(nsentence& s) {
	_seat(s, true);
}

void snpylm::add(nsentence& s) {
	_seat(s, true);
}

void snpylm::remove(nsentence& s) {
	_seat(s, false);
}

// Symmetric seat/un-seat of a sentence. add and remove touch exactly the same
// multiset of (template token, context) pairs on G^bg and the same NE spans on
// each H_k, so a full add-all / remove-all roundtrip returns every restaurant
// to zero customers/tables (WO-006 phase-1 acceptance).
void snpylm::_seat(nsentence& s, bool add) {
	lock_guard<mutex> m(_mutex);
	shared_ptr<cid> cdic = cid::create();
	int M = s.size();
	for (int j = 0; j < M; ++j) {
		chunk& ch = s.ch(j);
		if (ch.k >= (int)_nek.size() || (ch.k >= 1 && _nek[ch.k] == 0))
			_resize(ch.k);
	}
	vector<int> tvid(M, 0);
	for (int j = 0; j < M; ++j)
		tvid[j] = _tvid(s.ch(j));
	if (add) {
		for (int j = 0; j < M; ++j) {
			chunk tv(s.ch(j));
			tv.id = tvid[j];
			context *h = _bg->h();
			for (int d = 1; d < _n; ++d) {
				int pid = (j-d >= 0) ? tvid[j-d] : 0;
				h = h->make(pid);
			}
			_bg->add(tv, h);
			chunk& ch = s.ch(j);
			if (ch.k >= 1) {
				ch.id = cdic->index(ch);
				(*_H)[ch.k]->add(ch, (*_H)[ch.k]->h());
				int clen = 0;
				for (int w = 0; w < ch.len; ++w)
					clen += ch.wd(w).len;
				++_necnt[ch.k];
				_nelen[ch.k] += clen;
			}
		}
		// EOS template token (reserved id 0); default chunk => wd() is safe eos.
		chunk eos;
		eos.id = 0;
		context *h = _bg->h();
		for (int d = 1; d < _n; ++d) {
			int pid = (M-d >= 0) ? tvid[M-d] : 0;
			h = h->make(pid);
		}
		_bg->add(eos, h);
	} else {
		chunk eos;
		eos.id = 0;
		context *h = _bg->h();
		for (int d = 1; d < _n && h; ++d) {
			int pid = (M-d >= 0) ? tvid[M-d] : 0;
			h = h->find(pid);
		}
		if (!h)
			throw "bg eos context not found in snpylm::remove";
		_bg->remove(eos, h);
		for (int j = 0; j < M; ++j) {
			chunk tv(s.ch(j));
			tv.id = tvid[j];
			context *c = _bg->h();
			for (int d = 1; d < _n && c; ++d) {
				int pid = (j-d >= 0) ? tvid[j-d] : 0;
				c = c->find(pid);
			}
			if (!c)
				throw "bg context not found in snpylm::remove";
			_bg->remove(tv, c);
			chunk& ch = s.ch(j);
			if (ch.k >= 1) {
				context *r = (*_H)[ch.k]->h();
				(*_H)[ch.k]->remove(ch, r);
				int clen = 0;
				for (int w = 0; w < ch.len; ++w)
					clen += ch.wd(w).len;
				--_necnt[ch.k];
				_nelen[ch.k] -= clen;
			}
		}
	}
}

void snpylm::estimate(int iter) {
	for (int i = 1; i <= _k; ++i) {
		(*_H)[i]->gibbs(iter);
		(*_H)[i]->estimate(iter);
		(*_Hletter)[i]->estimate(iter);
	}
	_bg->gibbs(iter);
	_bg->estimate(iter);
	_spell->estimate(iter);
	// pi ~ Beta(1 + #NE base tables, gamma + #normal base tables)
	beta_distribution be;
	_pi = be(1.0+(double)_pine, _gamma+(double)_piw);
	if (_pi <= 0)
		_pi = 1e-12;
	if (_pi >= 1)
		_pi = 1.0-1e-12;
	// lambda_k ~ Gamma(a0 + sum char-length, b0 + span count) (Gamma-Poisson)
	gamma_dist gm;
	shared_ptr<generator> g = generator::create();
	for (int i = 1; i <= _k; ++i) {
		gamma_dist::param_type p(LAMBDA_A+_nelen[i], 1.0/(LAMBDA_B+(double)_necnt[i]));
		gm.param(p);
		_lambda[i] = gm((*g)());
	}
	if (getenv("NPBNLP_SNPYLM_STATS"))
		stats();
}

void snpylm::poisson_correction(int n) {
	_spell->poisson_correction(n);
	for (int i = 1; i <= _k; ++i)
		(*_Hletter)[i]->poisson_correction(n);
}

void snpylm::stats() const {
	int active = 0;
	for (int i = 1; i <= _k; ++i)
		if (_necnt[i] > 0)
			++active;
	fprintf(stderr, "[snpylm] k=%d active=%d pi=%.6f pine=%d piw=%d pieos=%d\n",
			_k, active, _pi, _pine, _piw, _pieos);
	for (int i = 1; i <= _k; ++i) {
		if (_necnt[i] == 0 && _rho[i] == 0)
			continue;
		fprintf(stderr, "  [class %d] ne_id=%d spans=%d chars=%.0f rho=%d lambda=%.3f\n",
				i, _nek[i], _necnt[i], _nelen[i], _rho[i], _lambda[i]);
	}
}

void snpylm::save(const char *file) {
	FILE *fp = NULL;
	if ((fp = fopen(file, "wb")) == NULL)
		throw "failed to open save file in snpylm::save";
	try {
		int hdr[4] = {_n, _hn, _hl, _k};
		if (fwrite(hdr, sizeof(int), 4, fp) != 4)
			throw "failed to write header in snpylm::save";
		if (fwrite(&_v, sizeof(int), 1, fp) != 1)
			throw "failed to write _v in snpylm::save";
		double dhdr[3] = {_gamma, _alpha, _pi};
		if (fwrite(dhdr, sizeof(double), 3, fp) != 3)
			throw "failed to write double header in snpylm::save";
		int cnt[3] = {_pine, _piw, _pieos};
		if (fwrite(cnt, sizeof(int), 3, fp) != 3)
			throw "failed to write switching counters in snpylm::save";
		if (fwrite(_nek.data(), sizeof(int), _k+1, fp) != (size_t)(_k+1))
			throw "failed to write nek in snpylm::save";
		if (fwrite(_rho.data(), sizeof(int), _k+1, fp) != (size_t)(_k+1))
			throw "failed to write rho in snpylm::save";
		if (fwrite(_lambda.data(), sizeof(double), _k+1, fp) != (size_t)(_k+1))
			throw "failed to write lambda in snpylm::save";
		if (fwrite(_necnt.data(), sizeof(int), _k+1, fp) != (size_t)(_k+1))
			throw "failed to write necnt in snpylm::save";
		if (fwrite(_nelen.data(), sizeof(double), _k+1, fp) != (size_t)(_k+1))
			throw "failed to write nelen in snpylm::save";
		_bg->save(fp);
		_spell->save(fp);
		for (int i = 0; i <= _k; ++i) {
			(*_H)[i]->save(fp);
			(*_Hletter)[i]->save(fp);
		}
	} catch (const char *ex) {
		fclose(fp);
		throw ex;
	}
	fclose(fp);
}

void snpylm::load(const char *file) {
	FILE *fp = NULL;
	if ((fp = fopen(file, "rb")) == NULL)
		throw "failed to open save file in snpylm::load";
	try {
		int hdr[4] = {0};
		if (fread(hdr, sizeof(int), 4, fp) != 4)
			throw "failed to read header in snpylm::load";
		_n = hdr[0]; _hn = hdr[1]; _hl = hdr[2]; _k = hdr[3];
		if (fread(&_v, sizeof(int), 1, fp) != 1)
			throw "failed to read _v in snpylm::load";
		double dhdr[3] = {0};
		if (fread(dhdr, sizeof(double), 3, fp) != 3)
			throw "failed to read double header in snpylm::load";
		_gamma = dhdr[0]; _alpha = dhdr[1]; _pi = dhdr[2];
		int cnt[3] = {0};
		if (fread(cnt, sizeof(int), 3, fp) != 3)
			throw "failed to read switching counters in snpylm::load";
		_pine = cnt[0]; _piw = cnt[1]; _pieos = cnt[2];
		_nek.assign(_k+1, 0);
		_rho.assign(_k+1, 0);
		_lambda.assign(_k+1, LAMBDA_A);
		_necnt.assign(_k+1, 0);
		_nelen.assign(_k+1, 0);
		if (fread(_nek.data(), sizeof(int), _k+1, fp) != (size_t)(_k+1))
			throw "failed to read nek in snpylm::load";
		if (fread(_rho.data(), sizeof(int), _k+1, fp) != (size_t)(_k+1))
			throw "failed to read rho in snpylm::load";
		if (fread(_lambda.data(), sizeof(double), _k+1, fp) != (size_t)(_k+1))
			throw "failed to read lambda in snpylm::load";
		if (fread(_necnt.data(), sizeof(int), _k+1, fp) != (size_t)(_k+1))
			throw "failed to read necnt in snpylm::load";
		if (fread(_nelen.data(), sizeof(double), _k+1, fp) != (size_t)(_k+1))
			throw "failed to read nelen in snpylm::load";
		_id2k.clear();
		for (int k = 1; k <= _k; ++k)
			if (_nek[k] > 0)
				_id2k[_nek[k]] = k;
		_bg->load(fp);
		_spell->load(fp);
		while ((int)_H->size() < _k+1) {
			int idx = (int)_H->size();
			_Hletter->push_back(shared_ptr<vpyp>(new vpyp(_hl)));
			(*_Hletter)[idx]->set_v(_v);
			_H->push_back(shared_ptr<hpyp>(new hpyp(_hn)));
			(*_H)[idx]->set_base((*_Hletter)[idx].get());
		}
		for (int i = 0; i <= _k; ++i) {
			(*_H)[i]->load(fp);
			(*_Hletter)[i]->load(fp);
		}
		_install_cbase();
	} catch (const char *ex) {
		fclose(fp);
		throw ex;
	}
	fclose(fp);
}
