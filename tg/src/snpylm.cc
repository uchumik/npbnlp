#include"snpylm.h"
#include"rd.h"
#include"beta.h"
#include"generator.h"
#include"poisson.h"
#include"chunktype.h"
#include<cassert>
#include<cmath>
#include<cstdlib>
#include<cstdio>
#include<deque>
#include<random>
#include<algorithm>

#define SVOCAB 3000
#define SGAMMA 10.0
#define SALPHA 1.0
#define LAMBDA_A 2.0
#define LAMBDA_B 1.0
#define SLEN 8
#define SA 1.0
#define SB 5.0

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
	_n(n < 2 ? 2 : n), _hn(hn < 1 ? 1 : hn), _hl(hl), _l(SLEN), _k(k), _v(SVOCAB),
	_gamma(SGAMMA), _alpha(SALPHA), _pi(1.0/(1.0+SGAMMA)), _tau(1.0), _a(SA), _b(SB),
	_clength(chunktype2::n, SLEN),
	_bg(new hpyp(_n)), _spell(new hpyp(_hl)),
	_hk(new vector<shared_ptr<hpyp> >), _hkletter(new vector<shared_ptr<hpyp> >),
	_pine(0), _piw(0), _pieos(0) {
	_spell->set_v(_v);
	_nek.assign(1, 0);
	_rho.assign(1, 0);
	_lambda.assign(1, 0);
	_necnt.assign(1, 0);
	_nelen.assign(1, 0);
	for (int i = 0; i <= _k; ++i) {
		_hkletter->push_back(shared_ptr<hpyp>(new hpyp(_hl)));
		(*_hkletter)[i]->set_v(_v);
		_hk->push_back(shared_ptr<hpyp>(new hpyp(_hn)));
		(*_hk)[i]->set_base((*_hkletter)[i].get());
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
	// H_k^0 (per-class NE surface base) = character model over the whole span.
	// _hkletter is only a non-null guard for hpyp's chunk seating; the cbase
	// delegates seat/score characters directly (never touching word.m).
	for (int k = 0; k < (int)_hk->size(); ++k) {
		int kk = k;
		(*_hk)[k]->set_base((*_hkletter)[k].get());
		(*_hk)[k]->set_cbase([this, kk](chunk& c) { return _hk_surf_lp(kk, c); });
		(*_hk)[k]->set_cbase_add([this, kk](chunk& c) { _hk_surf_add(kk, c); });
		(*_hk)[k]->set_cbase_remove([this, kk](chunk& c) { _hk_surf_remove(kk, c); });
	}
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
	while ((int)_hk->size() < target) {
		int idx = (int)_hk->size();
		_hkletter->push_back(shared_ptr<hpyp>(new hpyp(_hl)));
		(*_hkletter)[idx]->set_v(_v);
		_hk->push_back(shared_ptr<hpyp>(new hpyp(_hn)));
		(*_hk)[idx]->set_base((*_hkletter)[idx].get());
	}
	while ((int)_rho.size() < target) _rho.push_back(0);
	while ((int)_lambda.size() < target) _lambda.push_back(LAMBDA_A);
	while ((int)_necnt.size() < target) _necnt.push_back(0);
	while ((int)_nelen.size() < target) _nelen.push_back(0);
	if (_k < k)
		_k = k;
	_install_cbase(); // wire cbase for the newly added H_k
	_register_symbols();
}

void snpylm::set(int v, int k) {
	_v = v;
	_spell->set_v(_v);
	for (auto it = _hkletter->begin(); it != _hkletter->end(); ++it)
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

void snpylm::set_temp(double tau) {
	_tau = (tau > 0) ? tau : 1.0;
}

void snpylm::slice(double a, double b) {
	if (a > 0 && b > 0) {
		_a = a;
		_b = b;
	}
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

// log G0^spell(w): fixed-order character model over the word's characters plus a
// trailing EOS. w[i] returns 0 for i out of [0,len) (BOS/EOS), matching _seat.
double snpylm::_spell_lp(word& w) {
	double lp = 0;
	int nn = _spell->n();
	for (int i = 0; i < w.len+1; ++i) {
		const context *h = _spell->h();
		for (int j = 1; j < nn; ++j) {
			const context *c = h->find((int)w[i-j]);
			if (!c)
				break;
			h = c;
		}
		lp += _spell->lp((int)w[i], h);
	}
	return lp;
}

// seat (add) / un-seat the word's characters into _spell at fixed-depth context.
// add builds the full-depth context (make); remove walks the identical key path
// (find), so the two are exactly symmetric with no per-word order scratch.
void snpylm::_spell_seat(word& w, bool add) {
	int nn = _spell->n();
	for (int i = 0; i < w.len+1; ++i) {
		int c = (int)w[i];
		if (add) {
			context *h = _spell->h();
			for (int j = 1; j < nn; ++j)
				h = h->make((int)w[i-j]);
			_spell->add(c, h);
		} else {
			context *h = _spell->h();
			for (int j = 1; j < nn; ++j)
				h = h->make((int)w[i-j]);
			_spell->remove(c, h);
		}
	}
}

// collect the characters of a chunk's whole span as one sequence (words
// concatenated) with a trailing EOS (0). Shared by the H_k^0 lp/add/remove so
// they walk an identical key sequence (CRP add/remove symmetry).
static void span_chars(chunk& ch, vector<int>& out) {
	out.clear();
	for (int w = 0; w < ch.len; ++w) {
		word& wd = ch.wd(w);
		for (int c = 0; c < wd.len; ++c)
			out.push_back((int)wd[c]);
	}
	out.push_back(0); // surface-final EOS
}

// H_k^0(x): fixed-order character model of the whole span (no word.m scratch) +
// the lambda_k Poisson length prior over the character count. Used as the base
// measure of the span chunk PYP via set_cbase.
double snpylm::_hk_surf_lp(int k, chunk& ch) {
	hpyp *lm = (*_hkletter)[k].get();
	int nn = lm->n();
	static thread_local vector<int> ch_s;
	span_chars(ch, ch_s);
	double lp = 0;
	for (int i = 0; i < (int)ch_s.size(); ++i) {
		const context *h = lm->h();
		for (int d = 1; d < nn && i-d >= 0; ++d) {
			const context *c = h->find(ch_s[i-d]);
			if (!c)
				break;
			h = c;
		}
		lp += lm->lp(ch_s[i], h);
	}
	poisson_distribution po;
	lp += po.lp(_lambda[k], (int)ch_s.size()-1);
	return lp;
}

void snpylm::_hk_surf_add(int k, chunk& ch) {
	hpyp *lm = (*_hkletter)[k].get();
	int nn = lm->n();
	static thread_local vector<int> ch_s;
	span_chars(ch, ch_s);
	for (int i = 0; i < (int)ch_s.size(); ++i) {
		context *h = lm->h();
		for (int d = 1; d < nn && i-d >= 0; ++d)
			h = h->make(ch_s[i-d]);
		lm->add(ch_s[i], h);
	}
}

void snpylm::_hk_surf_remove(int k, chunk& ch) {
	hpyp *lm = (*_hkletter)[k].get();
	int nn = lm->n();
	static thread_local vector<int> ch_s;
	span_chars(ch, ch_s);
	for (int i = 0; i < (int)ch_s.size(); ++i) {
		context *h = lm->h();
		for (int d = 1; d < nn && i-d >= 0; ++d) {
			context *c = h->find(ch_s[i-d]);
			if (!c)
				break;
			h = c;
		}
		lm->remove(ch_s[i], h);
	}
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
		_spell_seat(tv.wd(0), true);
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
		_spell_seat(tv.wd(0), false);
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
				(*_hk)[ch.k]->add(ch, (*_hk)[ch.k]->h());
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
				context *r = (*_hk)[ch.k]->h();
				(*_hk)[ch.k]->remove(ch, r);
				int clen = 0;
				for (int w = 0; w < ch.len; ++w)
					clen += ch.wd(w).len;
				--_necnt[ch.k];
				_nelen[ch.k] -= clen;
			}
		}
	}
}

// ---------------------------------------------------------------------------
// inference: semi-Markov FFBS over the template n-gram (math SKILL.md 4.5).
// A segment (t,len,k) has template token sigma; the transition into it is the
// G^bg predictive P(sigma|context), the emission is E_0=1 (O) or P(x|H_k) (NE).
// O (k=0) is length-1 only. The forward marginalizes over histories via the
// nested vt table (context window = previous n-1 template tokens); the backward
// samples the segmentation. BOS/EOS reuse the lattice out-of-range eos chunk
// (id 0), so the first/last transition context is sigma=0 exactly as trained.
// ---------------------------------------------------------------------------

// template token id of a segment: NE class -> NE symbol id; O -> the word id;
// the lattice eos pseudo-chunk (id 0) yields sigma=0 (BOS/EOS).
int snpylm::_sigma(chunk& ch, int k) {
	if (k >= 1)
		return (k < (int)_nek.size()) ? _nek[k] : 0;
	return ch.wd(0).id;
}

// E_k: O contributes no surface term (E_0=1 => 0 in log space); NE contributes
// the class chunk-PYP predictive of the span. The lambda_k Poisson length prior
// is folded into H_k^0 (the cbase base measure), so a cached surface keeps its
// original length draw and only novel surfaces pay it (math 4.2).
double snpylm::_emit_lp(int k, chunk& ch) {
	if (k <= 0)
		return 0.0;
	double lp = (*_hk)[k]->lp(ch, (*_hk)[k]->h());
	return (_tau == 1.0) ? lp : _tau * lp; // E_k^tau: tau>1 damps NE emission
}

// transition P(sigma(ch,k) | c): use the chunk overload so the base escape hits
// the G0 mixture (_cbase) rather than a uniform int base.
double snpylm::_bg_lp(chunk& ch, int k, const context *c) {
	chunk tv(ch);
	tv.id = _sigma(ch, k);
	return _bg->lp(tv, c ? c : _bg->h());
}

// condition the slice on the current assignment (WO-003 flavour). On-path cells
// (matching cur's segment ending at t with the same length) keep k_cur alive by
// drawing mu from k_cur; off-path (and cur==nullptr) draw from the per-cell
// posterior. Emission is cached into l.emit[t][len-1][k] for the forward.
void snpylm::_slice(clattice2& l, nsentence *cur) {
	static const bool noslice = (getenv("NPBNLP_NOSLICE") != NULL);
	int T = (int)l.c.size();
	vector<pair<int, int> > pathmap; // pathmap[t] = {len, k_cur}
	if (cur) {
		pathmap.assign(T, make_pair(0, 0));
		int e = -1;
		for (int j = 0; j < cur->size(); ++j) {
			chunk& ch = cur->ch(j);
			e += ch.len;
			if (e >= 0 && e < T && ch.k >= 0 && ch.k <= _k)
				pathmap[e] = make_pair(ch.len, ch.k);
		}
	}
	beta_distribution be;
	l.emit.resize(T);
	for (int t = 0; t < T; ++t) {
		l.emit[t].resize(l.c[t].size());
		for (int ci = 0; ci < (int)l.c[t].size(); ++ci) {
			chunk& c = l.c[t][ci];
			int len = c.len;
			int j = len-1;
			l.emit[t][j].assign(_k+1, 0);
			vector<double> table;
			vector<int> cls;
			// O (class 0) is length-1 only; NE classes 1.._k for any length.
			for (int k = (len == 1) ? 0 : 1; k <= _k; ++k) {
				double em = _emit_lp(k, c);
				l.emit[t][j][k] = em;
				table.push_back(em + _bg_lp(c, k, _bg->h()));
				cls.push_back(k);
			}
			if (noslice) {
				for (auto k : cls)
					l.k[t][j].push_back(k);
				continue;
			}
			double z = 0;
			for (int i = 0; i < (int)table.size(); ++i)
				z = math::lse(z, table[i], (i == 0));
			for (auto& x : table)
				x -= z;
			double mu;
			bool on_path = (cur && pathmap[t].first == len);
			int idx = -1;
			if (on_path) {
				int k_cur = pathmap[t].second;
				for (int i = 0; i < (int)cls.size(); ++i)
					if (cls[i] == k_cur) { idx = i; break; }
			}
			if (idx >= 0) {
				mu = log(be(_a, _b)) + table[idx];
			} else {
				int id = rd::ln_draw(table);
				mu = log(be(_a, _b)) + table[id];
			}
			for (int i = 0; i < (int)table.size(); ++i)
				if (table[i] >= mu)
					l.k[t][j].push_back(cls[i]);
#ifndef NDEBUG
			if (idx >= 0) {
				bool alive = false;
				for (auto v : l.k[t][j])
					if (v == cls[idx]) { alive = true; break; }
				assert(alive);
			}
#endif
		}
	}
}

void snpylm::_forward(clattice2& l, int i, const context *c, chunk& ch, int k, double emit, chunk& prev, int q, bool bos, vt& a, vt& b, int n) {
	if (n <= 1) {
		if (bos || b.is_init()) {
			double tr = _bg_lp(ch, k, c);
			a.v = math::lse(a.v, b.v + emit + tr, !a.is_init());
			a.set(true);
		}
	} else {
		for (int pp = 0; pp < l.size(i); ++pp) {
			chunk& y = l.ch(i, pp+1);
			for (auto r = l.begin(i, pp); r != l.end(i, pp); ++r) {
				int sig = _sigma(y, *r);
				const context *h = (sig != 1) ? c->find(sig) : NULL;
				_forward(l, i-y.len, (h ? h : c), ch, k, emit, y, *r, (i < 0),
						a[prev.len-1][q], b[pp][*r], n-1);
			}
		}
	}
}

void snpylm::_backward(clattice2& l, int i, const context *c, chunk& ch, int k, chunk& prev, int q, bool bos, double& lpr, vt& b, int n) {
	if (n <= 1) {
		if (bos || b.is_init()) {
			double tr = _bg_lp(ch, k, c);
			double emit = _emit_lp(k, ch);
			lpr = math::lse(lpr, b.v + emit + tr, (lpr == 1.));
		}
	} else {
		for (int pp = 0; pp < l.size(i); ++pp) {
			chunk& y = l.ch(i, pp+1);
			for (auto r = l.begin(i, pp); r != l.end(i, pp); ++r) {
				int sig = _sigma(y, *r);
				const context *h = (sig != 1) ? c->find(sig) : NULL;
				_backward(l, i-y.len, (h ? h : c), ch, k, y, *r, (i < 0),
						lpr, b[pp][*r], n-1);
			}
		}
	}
}

nsentence snpylm::sample(nio& f, int i) {
	return _infer(f, i, nullptr, false);
}

nsentence snpylm::sample(nio& f, int i, nsentence *cur) {
	return _infer(f, i, cur, false);
}

nsentence snpylm::parse(nio& f, int i) {
	return _infer(f, i, nullptr, true);
}

nsentence snpylm::_infer(nio& f, int i, nsentence *cur, bool best) {
	clattice2 l(f, i, _clength);
	vt dp;
	_slice(l, cur);
	int T = (int)l.c.size();
	// forward filtering: dp[t][len-1][k] (nested by context window for _n>2).
	for (int t = 0; t < T; ++t) {
		for (int j = 0; j < l.size(t); ++j) {
			chunk& ch = l.ch(t, j+1);
			int s = t - ch.len;
			for (auto k = l.begin(t, j); k != l.end(t, j); ++k) {
				double emit = l.emit[t][j][*k];
				for (int p = 0; p < l.size(s); ++p) {
					chunk& prev = l.ch(s, p+1);
					for (auto q = l.begin(s, p); q != l.end(s, p); ++q) {
						int sig = _sigma(prev, *q);
						const context *h = (_n > 1 && sig != 1) ? _bg->h()->find(sig) : NULL;
						_forward(l, s-prev.len, (h ? h : _bg->h()), ch, *k, emit,
								prev, *q, (s < 0), dp[t][j][*k], dp[s][p][*q], _n-1);
					}
				}
			}
		}
	}
	// backward sampling from EOS (the out-of-range eos chunk, id 0, class 0).
	nsentence out;
	chunk *ch = l.cp(T, 1);
	ch->k = 0;
	int t = T - ch->len;
	bool dbg_first = true;
	while (t >= 0) {
		vector<double> table;
		vector<int> len;
		vector<int> cls;
		for (int p = 0; p < l.size(t); ++p) {
			chunk& prev = l.ch(t, p+1);
			for (auto q = l.begin(t, p); q != l.end(t, p); ++q) {
				int jd = (int)table.size();
				table.push_back(1.);
				len.push_back(p+1);
				cls.push_back(*q);
				int sig = _sigma(prev, *q);
				const context *h = (_n > 1 && sig != 1) ? _bg->h()->find(sig) : NULL;
				_backward(l, t-prev.len, (h ? h : _bg->h()), *ch, ch->k,
						prev, *q, (t < 0), table[jd], dp[t][p][*q], _n-1);
			}
		}
		if (table.empty())
			throw "no viable segmentation in snpylm::_infer";
		if (dbg_first) {
			static const bool dbg = (getenv("NPBNLP_DEBUG_LK") != NULL);
			if (dbg) {
				double zz = 0;
				for (int m = 0; m < (int)table.size(); ++m)
					zz = math::lse(zz, table[m], (m == 0));
				fprintf(stderr, "lk %.10f\n", zz);
			}
			dbg_first = false;
		}
		int id = best ? rd::best(table) : rd::ln_draw(table);
		ch = l.cp(t, len[id]);
		ch->k = cls[id];
		out.c.push_back(*ch);
		t -= ch->len;
	}
	reverse(out.c.begin(), out.c.end());
	out.n.resize(out.c.size(), 0);
	return out;
}

void snpylm::estimate(int iter) {
	for (int i = 1; i <= _k; ++i) {
		(*_hk)[i]->gibbs(iter);
		(*_hk)[i]->estimate(iter);
		(*_hkletter)[i]->estimate(iter);
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

void snpylm::poisson_correction(int) {
	// The NE length prior is handled analytically by the per-class lambda_k
	// (Gamma-Poisson) folded into H_k^0; the character models carry no word
	// base corpus, so there is nothing to correct here.
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
		if (fwrite(_nek.data(), sizeof(int), _k+1, fp) != (size_t)_k + 1)
			throw "failed to write nek in snpylm::save";
		if (fwrite(_rho.data(), sizeof(int), _k+1, fp) != (size_t)_k + 1)
			throw "failed to write rho in snpylm::save";
		if (fwrite(_lambda.data(), sizeof(double), _k+1, fp) != (size_t)_k + 1)
			throw "failed to write lambda in snpylm::save";
		if (fwrite(_necnt.data(), sizeof(int), _k+1, fp) != (size_t)_k + 1)
			throw "failed to write necnt in snpylm::save";
		if (fwrite(_nelen.data(), sizeof(double), _k+1, fp) != (size_t)_k + 1)
			throw "failed to write nelen in snpylm::save";
		_bg->save(fp);
		_spell->save(fp);
		for (int i = 0; i <= _k; ++i) {
			(*_hk)[i]->save(fp);
			(*_hkletter)[i]->save(fp);
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
		if (fread(_nek.data(), sizeof(int), _k+1, fp) != (size_t)_k + 1)
			throw "failed to read nek in snpylm::load";
		if (fread(_rho.data(), sizeof(int), _k+1, fp) != (size_t)_k + 1)
			throw "failed to read rho in snpylm::load";
		if (fread(_lambda.data(), sizeof(double), _k+1, fp) != (size_t)_k + 1)
			throw "failed to read lambda in snpylm::load";
		if (fread(_necnt.data(), sizeof(int), _k+1, fp) != (size_t)_k + 1)
			throw "failed to read necnt in snpylm::load";
		if (fread(_nelen.data(), sizeof(double), _k+1, fp) != (size_t)_k + 1)
			throw "failed to read nelen in snpylm::load";
		_id2k.clear();
		for (int k = 1; k <= _k; ++k)
			if (_nek[k] > 0)
				_id2k[_nek[k]] = k;
		_bg->load(fp);
		_spell->load(fp);
		while ((int)_hk->size() < _k+1) {
			int idx = (int)_hk->size();
			_hkletter->push_back(shared_ptr<hpyp>(new hpyp(_hl)));
			(*_hkletter)[idx]->set_v(_v);
			_hk->push_back(shared_ptr<hpyp>(new hpyp(_hn)));
			(*_hk)[idx]->set_base((*_hkletter)[idx].get());
		}
		for (int i = 0; i <= _k; ++i) {
			(*_hk)[i]->load(fp);
			(*_hkletter)[i]->load(fp);
		}
		_install_cbase();
	} catch (const char *ex) {
		fclose(fp);
		throw ex;
	}
	fclose(fp);
}
