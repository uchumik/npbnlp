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
#define GEN_W 0.5   // default generic-backoff mixture weight (member _gen_w)
// serialize format marker for the generic-backoff block (WO-007). A negative
// sentinel distinguishes the new gvid-keyed targeted ledger from the legacy
// WO-006 tvid-keyed block, whose first int was the positive _ne_generic id.
#define SNPYLM_GEN_MAGIC (-770007)
// serialize marker for the WO-009 probabilistic chunktype NE prior block. A
// distinct negative sentinel, appended after the generic block; legacy models
// (pre-WO-009) hit EOF here and load with theta disabled + hard-gate fallback.
#define SNPYLM_THETA_MAGIC (-770009)
// theta hyperparameters (defaults). THETA_HI/LO are the asymmetric beta
// pseudo-counts transcribed from the NE_ADMISSIBLE bool table (admissible type
// -> HI, inadmissible -> LO); THETA_KAPPA is the class-layer concentration.
#define THETA_HI 30.0
#define THETA_LO 0.1
#define THETA_KAPPA 30.0
// serialize marker for the WO-012 per-chunktype soft NE gate block, appended
// after the theta block. Legacy models hit EOF here and load with the gate
// disabled + hard-gate fallback.
#define SNPYLM_GATE_MAGIC (-770012)
// soft NE gate defaults (WO-012). GATE_HI is the prior mean g_ct for an
// admissible chunk type and is 0.5 ON PURPOSE: the score is the log-odds
// log g - log(1-g), so g=0.5 contributes exactly 0 and an admissible type is
// scored exactly as in the type-neutral baseline (no regression by
// construction). GATE_LO=0.001 turns the bool table's hard "forbidden" into a
// finite log-odds penalty of about -6.9. GATE_STRENGTH is a+b, only consulted
// in the experimental self-estimation mode (--gate_learn).
#define GATE_HI 0.5
#define GATE_LO 0.001
#define GATE_STRENGTH 1000.0

using namespace std;
using namespace npbnlp;

using gamma_dist = gamma_distribution<double>;

// psi char-type Dirichlet prior (beta_type), indexed by the chartype enum
// (chartype.h). Asymmetric: entity-like scripts (kanji / katakana / latin /
// digit) get a high pseudo-count, hiragana a low one, everything else the base.
// This injects the language prior without any NE labels. Tuning is done in the
// commander's experiments; only the shape lives here.
#define PSI_HI 8.0
#define PSI_LO 1.0
#define PSI_MID 2.0
static const double PSI_BETA[chartype::n] = {
	PSI_LO,  // U_HIRAGANA
	PSI_HI,  // U_KATAKANA
	PSI_MID, // U_KATA_OR_HIRA
	PSI_HI,  // U_HANJI
	PSI_MID, // U_HIRA_KATA
	PSI_MID, // U_HIRA_HANJI
	PSI_MID, // U_KATA_HANJI
	PSI_MID, // U_HIRA_KATA_HANJI
	PSI_MID, // U_MISC
	PSI_MID, // U_ARABIC
	PSI_MID, // U_GREEK
	PSI_MID, // U_HANGUL
	PSI_MID, // U_HEBREW
	PSI_HI,  // U_LATIN
	PSI_MID, // U_MYANMAR
	PSI_MID, // U_THAI
	PSI_HI,  // U_DIGIT
	PSI_MID, // U_PUNC
	PSI_MID  // U_SYNBOL
};
static double psi_beta_sum() {
	double s = 0;
	for (int t = 0; t < chartype::n; ++t)
		s += PSI_BETA[t];
	return s;
}

// type-driven NE admission: per chunktype2 (n=32), whether a span of that type
// may be an NE (z>=1). Initial rule: admissible iff the type's composition
// contains an entity script (kanji / katakana / latin / digit). Hiragana-only,
// punctuation/symbol-only, hira+punct, and MISC are inadmissible. clattice2
// already tags each candidate with ch.type via chunktype2::start/transition, so
// this is an O(1) table lookup (no character rescan). The commander tunes the
// final values from lattice-coverage / per-type NE-density measurements.
// short chunktype2 names for the stats() dumps (theta and the WO-012 gate).
static const char *CTNAME[chunktype2::n] = {
	"hira", "kata", "kanji", "latin", "digit", "punc", "sym", "h+k",
	"h+kj", "h+d", "h+p", "k+kj", "k+l", "k+d", "k+p", "kj+l", "kj+d",
	"kj+p", "l+d", "l+p", "l+s", "d+p", "hkkj", "hkjd", "hkjp", "kkjd",
	"kkjp", "klp", "kdp", "kjdp", "ldp", "misc"
};

static const bool NE_ADMISSIBLE[chunktype2::n] = {
	false, //  0 CH_HIRAGANA        hira
	true,  //  1 CH_KATAKANA        kata *
	true,  //  2 CH_HANJI           kanji *
	true,  //  3 CH_LATIN           latin *
	true,  //  4 CH_DIGIT           digit *
	false, //  5 CH_PUNC            punct
	true,  //  6 CH_SYNBOL          symbol: density .0115 (>.004), 45 NE — real
	true,  //  7 CH_HIRA_KATA       hira+kata *
	false, //  8 CH_HIRA_HANJI      density .0014, 255k cand — biggest garbage pool
	true,  //  9 CH_HIRA_DIGIT      hira+digit *
	false, // 10 CH_HIRA_PUNC       hira+punct
	true,  // 11 CH_KATA_HANJI      kata+kanji *
	true,  // 12 CH_KATA_LATIN      kata+latin *
	true,  // 13 CH_KATA_DIGIT      kata+digit *
	true,  // 14 CH_KATA_PUNC       kata+punct *
	true,  // 15 CH_HANJI_LATIN     kanji+latin *
	true,  // 16 CH_HANJI_DIGIT     kanji+digit *
	false, // 17 CH_HANJI_PUNC      density .0034 (<.004) — common punctuated nouns
	true,  // 18 CH_LATIN_DIGIT     latin+digit *
	true,  // 19 CH_LATIN_PUNC      latin+punct *
	true,  // 20 CH_LATIN_SYNBOL    latin+symbol *
	true,  // 21 CH_DIGIT_PUNC      digit+punct *
	false, // 22 CH_HIRA_KATA_HANJI density .00105 — mostly ordinary verbs/phrases
	true,  // 23 CH_HIRA_HANJI_DIGIT hira+kanji+digit *
	false, // 24 CH_HIRA_HANJI_PUNC density .00014, 121k cand
	true,  // 25 CH_KATA_HANJI_DIGIT *
	true,  // 26 CH_KATA_HANJI_PUNC *
	true,  // 27 CH_KATA_LATIN_PUNC *
	true,  // 28 CH_KATA_DIGIT_PUNC *
	true,  // 29 CH_HANJI_DIGIT_PUNC *
	true,  // 30 CH_LATIN_DIGIT_PUNC *
	false  // 31 CH_MISC            mixed / unknown composition
};

// Persistent backing store for the synthetic NE-symbol spellings. The wid
// dictionary keeps word keys whose _doc points into these buffers, so the
// storage must outlive the dictionary; a deque never relocates its elements,
// keeping every _doc pointer valid for the program's lifetime.
static deque<vector<unsigned int> > ne_bufs;

snpylm::snpylm(): snpylm(2, 1, 8, 10) {
}

snpylm::snpylm(int n, int hn, int hl, int k):
	_n(n < 2 ? 2 : n), _hn(hn < 1 ? 1 : hn), _hl(hl), _l(SLEN), _k(k), _v(SVOCAB),
	_gamma(SGAMMA), _alpha(SALPHA), _pi(1.0/(1.0+SGAMMA)), _tau(1.0),
	_type_admission(true), _hard_type_admission(false), _theta_enabled(false),
	_theta_hi(THETA_HI), _theta_lo(THETA_LO), _theta_kappa(THETA_KAPPA),
	_gate_enabled(true), _gate_learn(false), _gate_hi(GATE_HI), _gate_lo(GATE_LO),
	_gate_strength(GATE_STRENGTH),
	_l1_cache(false), _freq_cap(0), _a(SA), _b(SB),
	_clength(chunktype2::n, SLEN),
	_bg(new hpyp(_n)), _bg_gen(new hpyp(_n)), _ne_generic(0), _generic_backoff(true),
	_gen_w(GEN_W), _spell(new hpyp(_hl)),
	_hk(new vector<shared_ptr<hpyp> >), _hkletter(new vector<shared_ptr<hpyp> >),
	_pine(0), _piw(0), _pieos(0) {
	_spell->set_v(_v);
	_nek.assign(1, 0);
	_rho.assign(1, 0);
	_lambda.assign(1, 0);
	_necnt.assign(1, 0);
	_nelen.assign(1, 0);
	_psi.assign((size_t)(_k+1)*chartype::n, 0);
	// theta ledger (WO-009): shared layer over chunktype2, class layer flat.
	// q_cand defaults to uniform so the centring is well defined even before
	// measure_qcand runs (e.g. the self-contained inference test).
	_theta_sh.assign(chunktype2::n, 0);
	_theta_k.assign((size_t)(_k+1)*chunktype2::n, 0);
	_qcand.assign(chunktype2::n, 1.0/(double)chunktype2::n);
	// soft NE gate ledger (WO-012): per-chunktype NE / O segment counts. Only
	// read in --gate_learn mode, but always seated (see _seat).
	_gate_ne.assign(chunktype2::n, 0);
	_gate_o.assign(chunktype2::n, 0);
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
	// generic NE slot symbol (class-agnostic), registered once.
	if (_ne_generic <= 0) {
		ne_bufs.emplace_back();
		vector<unsigned int>& gb = ne_bufs.back();
		gb.push_back(0x01u);
		gb.push_back((unsigned int)'N');
		gb.push_back((unsigned int)'E');
		gb.push_back(0x0E0000u); // class 0 slot reserved for the generic symbol
		word gw(gb, 0, (int)gb.size());
		_ne_generic = d->index(gw);
	}
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
	// flat (target x chartype::n); appending rows keeps existing k*n+t indices.
	if ((int)_psi.size() < target*chartype::n)
		_psi.resize((size_t)target*chartype::n, 0);
	// theta class layer grows the same way (shared layer _theta_sh is class-agnostic
	// and fixed at chunktype2::n).
	if ((int)_theta_k.size() < target*chunktype2::n)
		_theta_k.resize((size_t)target*chunktype2::n, 0);
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

void snpylm::set_type_admission(bool f) {
	_type_admission = f;
}

// theta (WO-009) is off by default since WO-012; an explicit set_theta_* call is
// what re-enables it (the CLI only calls these when --theta_* was given), so the
// A/B path stays available without changing the default model.
void snpylm::set_theta_hi(double f) {
	if (f > 0)
		_theta_hi = f;
	_theta_enabled = true;
}

void snpylm::set_theta_lo(double f) {
	if (f > 0)
		_theta_lo = f;
	_theta_enabled = true;
}

void snpylm::set_theta_kappa(double f) {
	if (f > 0)
		_theta_kappa = f;
	_theta_enabled = true;
}

void snpylm::set_hard_type_admission(bool f) {
	_hard_type_admission = f;
}

// soft NE gate hyperparameters (WO-012). g_ct is a probability, so hi/lo must
// lie strictly inside (0,1): 0 or 1 would make the log-odds infinite.
void snpylm::set_gate_hi(double f) {
	if (f <= 0 || f >= 1)
		throw "gate_hi must lie in (0,1) in snpylm::set_gate_hi";
	_gate_hi = f;
}

void snpylm::set_gate_lo(double f) {
	if (f <= 0 || f >= 1)
		throw "gate_lo must lie in (0,1) in snpylm::set_gate_lo";
	_gate_lo = f;
}

void snpylm::set_gate_strength(double f) {
	if (f <= 0)
		throw "gate_strength must be positive in snpylm::set_gate_strength";
	_gate_strength = f;
}

void snpylm::set_gate_learn(bool f) {
	_gate_learn = f;
}

// store q_cand: raw per-chunktype candidate counts -> add-one-smoothed,
// normalised probabilities. Robust to a wrong-length / empty input (falls back
// to uniform), so the centring in _theta_lp is always well defined.
void snpylm::set_qcand(const std::vector<double>& counts) {
	lock_guard<mutex> m(_mutex);
	_qcand.assign(chunktype2::n, 0);
	double z = 0;
	for (int t = 0; t < chunktype2::n; ++t) {
		double c = (t < (int)counts.size()) ? counts[t] : 0.0;
		double v = c + 1.0; // add-one smoothing
		_qcand[t] = v;
		z += v;
	}
	if (z <= 0) {
		_qcand.assign(chunktype2::n, 1.0/(double)chunktype2::n);
		return;
	}
	for (int t = 0; t < chunktype2::n; ++t)
		_qcand[t] /= z;
}

// measure q_cand from the corpus: build a clattice2 per sentence (with the
// model's own _clength cap) and tally every candidate span's chunk type. This is
// the fixed background candidate-type distribution used to centre the NE emission
// factor (log theta^_k - log q_cand); measured once before any sampling.
void snpylm::measure_qcand(nio& f) {
	std::vector<double> counts(chunktype2::n, 0);
	int nsent = (int)f.head.size()-1;
	for (int i = 0; i < nsent; ++i) {
		clattice2 l(f, i, _clength);
		for (int t = 0; t < (int)l.c.size(); ++t) {
			for (int ci = 0; ci < (int)l.c[t].size(); ++ci) {
				int ct = l.c[t][ci].type;
				if (ct >= 0 && ct < chunktype2::n)
					counts[ct] += 1.0;
			}
		}
	}
	set_qcand(counts);
}

void snpylm::set_l1_cache(bool f) {
	_l1_cache = f;
}

void snpylm::set_generic_backoff(bool f) {
	_generic_backoff = f;
}

void snpylm::set_gen_w(double w) {
	if (!(w > 0.0 && w < 1.0))
		throw "gen_w must be in (0,1) in snpylm::set_gen_w";
	_gen_w = w;
}

// vocab-scale uniform base for the generic ledger: hpyp::lp(int,...) bottoms
// out at -log(_v), and _bg_gen keeps the constructed default _v=1 unless this
// is called, so P_gen of a never-seen event would be ~1 and the backoff
// mixture would collapse into a flat bonus (see snpylm.h).
void snpylm::set_wv(int v) {
	if (v < 2)
		throw "wv must be >= 2 in snpylm::set_wv";
	_bg_gen->set_v(v);
}

// tally every corpus word type (id -> occurrence count). Called over each
// sentence once, before set_freq_cap, from the sne.cc collection loop (which
// runs independently of the all-O seed). This is a static observation of the
// training text, not a CRP ledger, so it is add/remove-neutral.
void snpylm::count_freq(nsentence& s) {
	lock_guard<mutex> m(_mutex);
	for (int j = 0; j < s.size(); ++j) {
		chunk& ch = s.ch(j);
		for (int w = 0; w < ch.len; ++w)
			++_wfreq[ch.wd(w).id];
	}
}

// set / derive the single-word NE frequency cap.
//   cap > 0  : use it verbatim.
//   cap == 0 : disable the gate.
//   cap == -1: auto. We want to bar the handful of very high-frequency word
//     types that together make up the top 1% of all token occurrences (in
//     Zipfian text these are exactly the function words の/を/年 ... that fuel
//     rich-get-richer). Sort real word types (id>1) by descending frequency and
//     accumulate their token mass; let f_cross be the frequency of the word at
//     which the running mass first exceeds 1% of the total. Every word in that
//     top block has freq >= f_cross, so a cap of f_cross-1 blocks the whole
//     block (allowed set is freq <= cap) and nothing else -- the maximal cap
//     that still bars the entire top-1% mass ("minimal over-blocking"). Note
//     this bars the crossing word itself, which is required: with the largest
//     single type often exceeding 1% on its own, using f_cross verbatim would
//     leave the most frequent word admissible. id<=1 (BOS/EOS/unk) are excluded
//     from the mass and the sort so unk never distorts the threshold.
void snpylm::set_freq_cap(int cap) {
	if (cap >= 0) {
		_freq_cap = cap;
		return;
	}
	// cap == -1 (auto)
	long long total = 0;
	std::vector<int> freqs;
	freqs.reserve(_wfreq.size());
	for (auto& kv : _wfreq) {
		if (kv.first <= 1)
			continue; // exclude reserved / unk
		total += kv.second;
		freqs.push_back(kv.second);
	}
	if (total <= 0 || freqs.empty()) {
		_freq_cap = 0; // no data -> gate disabled
		return;
	}
	std::sort(freqs.begin(), freqs.end(), std::greater<>());
	double mass_th = 0.01 * (double)total;
	long long cum = 0;
	int f_cross = freqs.front();
	for (int f : freqs) {
		cum += f;
		if ((double)cum > mass_th) {
			f_cross = f;
			break;
		}
	}
	_freq_cap = (f_cross > 1) ? (f_cross - 1) : 0;
}

bool snpylm::_freq_ok(int wid) const {
	if (_freq_cap <= 0)
		return true;      // gate disabled
	if (wid <= 1)
		return true;      // reserved / unk -> freq 0 -> always admissible
	auto it = _wfreq.find(wid);
	int f = (it != _wfreq.end()) ? it->second : 0; // unseen id -> freq 0
	return f <= _freq_cap;
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

// true iff id is a registered class NE symbol (some _nek[k], k>=1). O(1) hash
// lookup; the generic symbol _ne_generic is never in _id2k, so it returns false.
bool snpylm::_is_ne_sym(int id) const {
	return _id2k.find(id) != _id2k.end();
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

int snpylm::_psii(int k, int t) const {
	return k*chartype::n + t;
}

// log psi(x|k) = sum_c log( (n_k(type(c)) + beta_type) / (sum n_k + sum beta) ),
// the collapsed Dirichlet-multinomial predictive over the span's real characters
// (EOS excluded). Counts n_k are the current counts (before seating x).
double snpylm::_psi_lp(int k, chunk& ch) {
	double row = 0;
	for (int t = 0; t < chartype::n; ++t)
		row += _psi[_psii(k, t)];
	double denom = row + psi_beta_sum();
	double lp = 0;
	for (int w = 0; w < ch.len; ++w) {
		word& wd = ch.wd(w);
		for (int c = 0; c < wd.len; ++c) {
			int t = (int)chartype::get(wd[c]);
			lp += log(_psi[_psii(k, t)] + PSI_BETA[t]) - log(denom);
		}
	}
	return lp;
}

// ---------------------------------------------------------------------------
// theta: probabilistic chunktype NE prior (WO-009). 2-level Dirichlet over the
// span's single chunk type ct. See snpylm.h for the model.
// ---------------------------------------------------------------------------
int snpylm::_thetaki(int k, int ct) const {
	return k*chunktype2::n + ct;
}

// beta_ct: the asymmetric shared-layer prior transcribed from the hard bool
// table. Admissible type -> theta_hi, inadmissible -> theta_lo. The observed NE
// density is deliberately NOT folded in (empirical-Bayes cycle avoidance).
double snpylm::_theta_beta(int ct) const {
	return (ct >= 0 && ct < chunktype2::n && NE_ADMISSIBLE[ct]) ? _theta_hi : _theta_lo;
}

double snpylm::_theta_beta_sum() const {
	double s = 0;
	for (int t = 0; t < chunktype2::n; ++t)
		s += NE_ADMISSIBLE[t] ? _theta_hi : _theta_lo;
	return s;
}

// shared layer q^(ct) = (n(ct) + beta_ct) / (n(.) + sum beta).
double snpylm::_qhat(int ct) const {
	double row = 0;
	for (int t = 0; t < chunktype2::n; ++t)
		row += _theta_sh[t];
	double denom = row + _theta_beta_sum();
	return (_theta_sh[ct] + _theta_beta(ct)) / denom;
}

// plug-in class layer theta^_k(ct) = (n_k(ct) + kappa*q^(ct)) / (n_k(.) + kappa).
double snpylm::_theta_k_hat(int k, int ct) const {
	double nk = 0;
	for (int t = 0; t < chunktype2::n; ++t)
		nk += _theta_k[_thetaki(k, t)];
	double qh = _qhat(ct);
	return (_theta_k[_thetaki(k, ct)] + _theta_kappa*qh) / (nk + _theta_kappa);
}

// centred NE emission factor: log theta^_k(ct) - log q_cand(ct). Centring against
// the fixed background candidate-type distribution q_cand keeps the score from
// simply rewarding the most common lattice type; only a class' *relative* type
// preference survives.
double snpylm::_theta_lp(int k, chunk& ch) {
	int ct = ch.type;
	if (ct < 0 || ct >= chunktype2::n)
		return 0.0; // out-of-range type: no theta factor (see _emit_lp guard)
	double qc = (ct < (int)_qcand.size()) ? _qcand[ct] : 1.0/(double)chunktype2::n;
	if (qc <= 0)
		qc = 1e-12;
	return log(_theta_k_hat(k, ct)) - log(qc);
}

// theta emission factor active iff enabled, the master type switch is on, and
// the legacy hard gate is not in force (hard mode is a pure bool-gate A/B).
bool snpylm::_theta_score_on() const {
	return _theta_enabled && _type_admission && !_hard_type_admission;
}

// ---------------------------------------------------------------------------
// soft NE gate g_ct (WO-012). g_ct = P(z>=1 | type=ct), transcribed from the
// NE_ADMISSIBLE bool table. The emission contribution is the LOG-ODDS
// log g_ct - log(1-g_ct), applied to NE segments only (see _emit_lp).
// ---------------------------------------------------------------------------

// fixed prior mean, straight off the bool table.
double snpylm::_gate_prior(int ct) const {
	return (ct >= 0 && ct < chunktype2::n && NE_ADMISSIBLE[ct]) ? _gate_hi : _gate_lo;
}

// the g actually used. Default: the fixed prior (the type's NE-bearing tendency
// is language knowledge, not a corpus statistic to re-estimate -- a self-fitted
// prior is descriptive and self-reinforcing). --gate_learn switches to the Beta
// posterior mean g^_ct = (n_NE + a)/(n_NE + n_O + a + b) with a+b = _gate_strength;
// that mode exists to demonstrate the self-reinforcement, not to be used.
double snpylm::_gate_ghat(int ct) const {
	double g0 = _gate_prior(ct);
	if (!_gate_learn)
		return g0;
	if (ct < 0 || ct >= (int)_gate_ne.size())
		return g0;
	double a = _gate_strength * g0;
	double b = _gate_strength - a;
	double ne = _gate_ne[ct];
	double o = _gate_o[ct];
	double den = ne + o + a + b;
	if (den <= 0)
		return g0;
	return (ne + a) / den;
}

// log-odds contribution. Clamped away from 0/1 so an extreme hyperparameter or
// an all-NE type in learn mode can never produce +-inf in the lattice score.
double snpylm::_gate_lp(int ct) const {
	double g = _gate_ghat(ct);
	if (g < 1e-12)
		g = 1e-12;
	if (g > 1.0-1e-12)
		g = 1.0-1e-12;
	return log(g) - log(1.0-g);
}

// gate active iff enabled, the master type switch is on, and the legacy hard
// gate is not in force (hard mode is a pure bool-gate A/B, gate contributes 0).
bool snpylm::_gate_score_on() const {
	return _gate_enabled && _type_admission && !_hard_type_admission;
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
	lp += _psi_lp(k, ch); // char-type prior factor (same site as the Poisson)
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
	// psi char-type counts (real characters only, EOS excluded)
	for (int w = 0; w < ch.len; ++w) {
		word& wd = ch.wd(w);
		for (int c = 0; c < wd.len; ++c)
			++_psi[_psii(k, (int)chartype::get(wd[c]))];
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
	for (int w = 0; w < ch.len; ++w) {
		word& wd = ch.wd(w);
		for (int c = 0; c < wd.len; ++c)
			--_psi[_psii(k, (int)chartype::get(wd[c]))];
	}
}

// GEM predictive rho_k = (m_k + alpha/K)/(m. + alpha), summing to 1 over the K
// active classes. Shared by the G0 mixture and the generic-backoff interpolation.
double snpylm::_rho_k(int k) const {
	double denom = (double)_pine + _alpha;
	double num = (double)_rho[k] + _alpha/(double)(_k > 0 ? _k : 1);
	return num/denom;
}

// log G0(v). NE weight uses the DP predictive rho_k (see _rho_k) so G0 stays a
// proper mixture.
double snpylm::_g0_lp(chunk& tv) {
	int kind = _kind(tv.id);
	if (kind > 0) {
		return log(_pi) + log(_rho_k(kind));
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
	// tvid[j] = template token id; gvid[j] = generic-replacement id (NE positions
	// collapse to _ne_generic so the generic-slot context pools frames across
	// classes, WO-007). _bg (the body ledger) keeps its tvid keys unchanged.
	vector<int> tvid(M, 0);
	vector<int> gvid(M, 0);
	for (int j = 0; j < M; ++j) {
		tvid[j] = _tvid(s.ch(j));
		gvid[j] = (s.ch(j).k >= 1) ? _ne_generic : tvid[j];
	}
	// does the n-1 history window ending just before position `pos` contain an
	// NE? (pos in [0,M]; pos==M is the EOS slot.) Depends only on s, so add and
	// remove agree on which generic-slot customers are seated.
	auto window_has_ne = [&](int pos) -> bool {
		for (int d = 1; d < _n; ++d) {
			int p = pos - d;
			if (p >= 0 && p < M && s.ch(p).k >= 1)
				return true;
		}
		return false;
	};
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
			// soft NE gate ledger (WO-012): every segment, NE and O alike, is
			// tallied against its chunk type. Seated unconditionally (not under
			// _gate_learn) so flipping the flag mid-run cannot leak; the score
			// side reads these counts only when _gate_learn is on.
			if (ch.type >= 0 && ch.type < chunktype2::n) {
				if (ch.k >= 1)
					++_gate_ne[ch.type];
				else if (ch.k == 0)
					++_gate_o[ch.type];
			}
			if (ch.k >= 1) {
				ch.id = cdic->index(ch);
				(*_hk)[ch.k]->add(ch, (*_hk)[ch.k]->h());
				int clen = 0;
				for (int w = 0; w < ch.len; ++w)
					clen += ch.wd(w).len;
				++_necnt[ch.k];
				_nelen[ch.k] += clen;
				// theta ledger (WO-009): +1 on both layers for this span's chunk
				// type. Seated unconditionally (like _psi) so add/remove stay
				// symmetric regardless of the theta/hard/neutral score mode; the
				// score side reads the counts only when _theta_score_on().
				int ct = ch.type;
				if (ct >= 0 && ct < chunktype2::n) {
					++_theta_sh[ct];
					++_theta_k[_thetaki(ch.k, ct)];
				}
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
		// targeted generic-slot ledger (gvid-keyed contexts). (a) the NE slot at
		// every NE position, (b) the exit-frame word after an NE, (c) EOS after an
		// NE; NE-free positions are not seated.
		if (_generic_backoff) {
			for (int j = 0; j < M; ++j) {
				int tok;
				if (s.ch(j).k >= 1)
					tok = _ne_generic;   // (a) NE slot
				else if (window_has_ne(j))
					tok = tvid[j];       // (b) exit-frame word
				else
					continue;
				context *hg = _bg_gen->h();
				for (int d = 1; d < _n; ++d) {
					int pid = (j-d >= 0) ? gvid[j-d] : 0;
					hg = hg->make(pid);
				}
				_bg_gen->add(tok, hg);
			}
			if (window_has_ne(M)) {          // (c) EOS after an NE
				context *hg = _bg_gen->h();
				for (int d = 1; d < _n; ++d) {
					int pid = (M-d >= 0) ? gvid[M-d] : 0;
					hg = hg->make(pid);
				}
				_bg_gen->add(0, hg);
			}
		}
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
			// symmetric un-seating of the gate ledger (identical predicate).
			if (ch.type >= 0 && ch.type < chunktype2::n) {
				if (ch.k >= 1)
					--_gate_ne[ch.type];
				else if (ch.k == 0)
					--_gate_o[ch.type];
			}
			if (ch.k >= 1) {
				context *r = (*_hk)[ch.k]->h();
				(*_hk)[ch.k]->remove(ch, r);
				int clen = 0;
				for (int w = 0; w < ch.len; ++w)
					clen += ch.wd(w).len;
				--_necnt[ch.k];
				_nelen[ch.k] -= clen;
				int ct = ch.type;
				if (ct >= 0 && ct < chunktype2::n) {
					--_theta_sh[ct];
					--_theta_k[_thetaki(ch.k, ct)];
				}
			}
		}
		// symmetric removal of the targeted generic-slot ledger (same predicate).
		if (_generic_backoff) {
			for (int j = 0; j < M; ++j) {
				int tok;
				if (s.ch(j).k >= 1)
					tok = _ne_generic;
				else if (window_has_ne(j))
					tok = tvid[j];
				else
					continue;
				context *hg = _bg_gen->h();
				for (int d = 1; d < _n && hg; ++d) {
					int pid = (j-d >= 0) ? gvid[j-d] : 0;
					hg = hg->find(pid);
				}
				if (!hg)
					throw "bg_gen context not found in snpylm::remove";
				_bg_gen->remove(tok, hg);
			}
			if (window_has_ne(M)) {
				context *hg = _bg_gen->h();
				for (int d = 1; d < _n && hg; ++d) {
					int pid = (M-d >= 0) ? gvid[M-d] : 0;
					hg = hg->find(pid);
				}
				if (!hg)
					throw "bg_gen eos context not found in snpylm::remove";
				_bg_gen->remove(0, hg);
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
	double lp;
	// single-word cache bypass: a len==1 NE pays the base measure H_k^0 directly
	// instead of the chunk-PYP predictive, so a frequent word cannot accumulate a
	// cheap cached table and monopolise a class (rich-get-richer). len>=2 spans
	// keep the cache. This makes the SCORE deficient as a generative model (the
	// len==1 emission no longer normalises as the H_k predictive), but the seat
	// ledger (rho/lambda/psi via add/remove) is untouched, so it is a consistent
	// sampler score. set_l1_cache(true) restores the exact cached predictive.
	if (_l1_cache || ch.len >= 2)
		lp = (*_hk)[k]->lp(ch, (*_hk)[k]->h());
	else
		lp = _hk_surf_lp(k, ch);
	lp = (_tau == 1.0) ? lp : _tau * lp; // E_k^tau: tau>1 damps NE emission
	// probabilistic chunktype NE prior theta (WO-009): add the centred factor
	// log theta^_k(ct) - log q_cand(ct) OUTSIDE the tau annealing (it is a prior
	// over the span's type, not part of the surface likelihood that anneals). The
	// O side (k<=0, returned above) never pays it -- a deliberate generative
	// deficiency, but the seat ledger (_theta_sh/_theta_k) is add/remove symmetric,
	// so the sampler score stays consistent (same argument as the l1-cache bypass).
	if (_theta_score_on() && ch.type >= 0 && ch.type < chunktype2::n)
		lp += _theta_lp(k, ch);
	// soft NE gate (WO-012): the log-odds log g_ct - log(1-g_ct), also OUTSIDE the
	// tau annealing (a prior over the span's type, not surface likelihood). Added
	// on the NE side ONLY -- the O branch returned 0 above and must keep returning
	// 0. Paying log(1-g) per O segment would look symmetric but would make the
	// total score depend on the NUMBER of segments, biasing the sampler towards
	// coarser segmentations; the log-odds form is chunk-count neutral. An
	// out-of-range ch.type contributes nothing.
	if (_gate_score_on() && ch.type >= 0 && ch.type < chunktype2::n)
		lp += _gate_lp(ch.type);
	return lp;
}

// transition P(sigma(ch,k) | c): use the chunk overload so the base escape hits
// the G0 mixture (_cbase) rather than a uniform int base.
double snpylm::_bg_lp(chunk& ch, int k, const context *c) {
	chunk tv(ch);
	tv.id = _sigma(ch, k);
	return _bg->lp(tv, c ? c : _bg->h());
}

// transition score with generic NE-slot backoff (WO-007, both frame sides).
//   k >= 1 (an NE slot):  the LEFT side. Interpolate the class-specific slot
//     predictive with the pooled generic slot (rho_k re-personalises it):
//       log( w * P_bg(NE_k|ctx) + (1-w) * P_gen(NE_generic|gctx) * rho_k ).
//   k == 0 (a word / EOS): the RIGHT side (exit frame "<NE> 氏 が"). Only when
//     the n-1 history window contains an NE (ctx_has_ne) is the word predictive
//     mixed with the pooled generic exit-frame count (no rho_k):
//       log( w * P_bg(v|ctx) + (1-w) * P_gen(v|gctx) ).
//     ctx_has_ne == false leaves the score exactly P_bg(v|ctx), so NE-free
//     regions are unchanged (regression guard).
// gctx (cg) is the parallel _bg_gen context, keyed on gvid: NE symbols in the
// history are folded to _ne_generic (see _forward/_backward's gsig). w is the
// member _gen_w. Computed in log space (manual logsumexp) to avoid underflow.
//
// Deficiency: the mixture is applied only to transitions whose window contains
// an NE, so it does not normalise as a generative model over all next tokens
// (a proper G^bg would). But the seating ledger (_bg / _bg_gen add/remove) is
// exactly symmetric, so this is a consistent sampler score -- the same argument
// as the len==1 l1-cache bypass in _emit_lp above.
double snpylm::_trans_lp(chunk& ch, int k, const context *c, const context *cg,
		bool ctx_has_ne) {
	double base = _bg_lp(ch, k, c);
	if (!_generic_backoff)
		return base;
	if (k >= 1) {
		double gen = _bg_gen->lp(_ne_generic, cg ? cg : _bg_gen->h());
		double la = log(_gen_w) + base;
		double lb = log(1.0 - _gen_w) + gen + log(_rho_k(k));
		double m = (la > lb) ? la : lb;
		return m + log(exp(la-m) + exp(lb-m));
	}
	// k == 0: right-context mixing only when the window holds an NE.
	if (!ctx_has_ne)
		return base;
	int sig = _sigma(ch, 0);
	double gen = _bg_gen->lp(sig, cg ? cg : _bg_gen->h());
	double la = log(_gen_w) + base;
	double lb = log(1.0 - _gen_w) + gen;
	double m = (la > lb) ? la : lb;
	return m + log(exp(la-m) + exp(lb-m));
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
			// type-driven admission (WO-009 / WO-012). Three modes:
			//  - neutral (--no_type_admission, _type_admission==false): every type is
			//    an NE candidate, no soft gate, no theta, no hard gate.
			//  - hard (--hard_type_admission): revive the legacy bool gate; a span
			//    whose chunk type is not entity-bearing cannot be NE. The soft gate
			//    and theta are both off.
			//  - default: every type is an NE candidate; the soft gate g_ct (in
			//    _emit_lp) suppresses inadmissible types by a finite log-odds
			//    penalty instead of a hard cut, so the current assignment can never
			//    be eliminated from the lattice by the type prior (slice safety).
			int ct = c.type;
			bool ne_ok;
			if (!_type_admission)
				ne_ok = true;                                 // neutral
			else if (_hard_type_admission)
				ne_ok = (ct >= 0 && ct < chunktype2::n && NE_ADMISSIBLE[ct]); // hard gate
			else
				ne_ok = true;                                 // theta handles it
			// rarity gate: only a low-frequency single word (len==1) may be an
			// NE. len>=2 spans are exempt (multi-word NE are not frequency
			// gated). ANDs with the type admission above.
			bool freq_ok = (len != 1) || _freq_ok(c.wd(0).id);
			// O (class 0) is length-1 only; NE classes 1.._k for any length.
			for (int k = (len == 1) ? 0 : 1; k <= _k; ++k) {
				if (k >= 1 && !ne_ok)
					continue;
				if (k >= 1 && len == 1 && !freq_ok)
					continue;
				double em = _emit_lp(k, c);
				l.emit[t][j][k] = em;
				// root context (no history) -> ctx_has_ne=false: identical to the
				// pre-WO-007 slice score (NE path unaffected by ctx_has_ne).
				table.push_back(em + _trans_lp(c, k, _bg->h(), _bg_gen->h(), false));
				cls.push_back(k);
			}
			if (table.empty()) // no admissible class here (e.g. len>1 non-entity)
				continue;
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

void snpylm::_forward(clattice2& l, int i, const context *c, const context *cg, chunk& ch, int k, double emit, chunk& prev, int q, bool bos, vt& a, vt& b, int n, bool ctx_has_ne) {
	if (n <= 1) {
		if (bos || b.is_init()) {
			double tr = _trans_lp(ch, k, c, cg, ctx_has_ne);
			a.v = math::lse(a.v, b.v + emit + tr, !a.is_init());
			a.set(true);
		}
	} else {
		for (int pp = 0; pp < l.size(i); ++pp) {
			chunk& y = l.ch(i, pp+1);
			for (auto r = l.begin(i, pp); r != l.end(i, pp); ++r) {
				int sig = _sigma(y, *r);
				bool is_ne = _is_ne_sym(sig);
				// generic-slot context folds NE symbols to _ne_generic (gsig).
				int gsig = is_ne ? _ne_generic : sig;
				const context *h = (sig != 1) ? c->find(sig) : NULL;
				const context *hg = (gsig != 1) ? cg->find(gsig) : NULL;
				_forward(l, i-y.len, (h ? h : c), (hg ? hg : cg), ch, k, emit,
						y, *r, (i < 0), a[prev.len-1][q], b[pp][*r], n-1,
						ctx_has_ne || is_ne);
			}
		}
	}
}

void snpylm::_backward(clattice2& l, int i, const context *c, const context *cg, chunk& ch, int k, chunk& prev, int q, bool bos, double& lpr, vt& b, int n, bool ctx_has_ne) {
	if (n <= 1) {
		if (bos || b.is_init()) {
			double tr = _trans_lp(ch, k, c, cg, ctx_has_ne);
			double emit = _emit_lp(k, ch);
			lpr = math::lse(lpr, b.v + emit + tr, (lpr == 1.));
		}
	} else {
		for (int pp = 0; pp < l.size(i); ++pp) {
			chunk& y = l.ch(i, pp+1);
			for (auto r = l.begin(i, pp); r != l.end(i, pp); ++r) {
				int sig = _sigma(y, *r);
				bool is_ne = _is_ne_sym(sig);
				int gsig = is_ne ? _ne_generic : sig;
				const context *h = (sig != 1) ? c->find(sig) : NULL;
				const context *hg = (gsig != 1) ? cg->find(gsig) : NULL;
				_backward(l, i-y.len, (h ? h : c), (hg ? hg : cg), ch, k, y, *r,
						(i < 0), lpr, b[pp][*r], n-1, ctx_has_ne || is_ne);
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

// Phase-A diagnostic. For sentence i, dump a background score for every lattice
// span candidate (t,len) to stderr:
//   surp   = -sum_{w in span} log P_bg(w | left O-context)  -- the cost of
//            generating the span word-by-word on the background (O) path.
//   fgain  = log P_bg(w_{e+1} | ctx WITHOUT span's last word w_e)
//          - log P_bg(w_{e+1} | ctx WITH w_e)  -- how much abstracting w_e (as
//            if the span were one NE token) improves the right-context
//            prediction of the next word. For n=2 "without w_e" is the root.
// Char offsets [char_s, char_e) use the same cumulative word-length coordinate
// as NPBNLP_LATTICE_COVER. Read-only; run right after the all-O seed.
void snpylm::span_score_dump(nio& f, int i) {
	clattice2 l(f, i, _clength);
	int T = (int)l.c.size();
	if (T == 0)
		return;
	vector<int> cum(T+1, 0);
	for (int p = 0; p < T; ++p)
		cum[p+1] = cum[p] + l.c[p][0].wd(0).len; // word char length
	for (int t = 0; t < T; ++t) {
		for (int ci = 0; ci < (int)l.c[t].size(); ++ci) {
			chunk& c = l.c[t][ci];
			int len = c.len;
			int s = t - len + 1;
			// surp: sum of O-path word surprisals over [s..t]
			double surp = 0;
			for (int wpos = s; wpos <= t; ++wpos) {
				const context *h = _bg->h();
				for (int d = 1; d < _n; ++d) {
					int wi = wpos - d;
					int pid = (wi >= 0) ? l.c[wi][0].wd(0).id : 0;
					const context *g = h->find(pid);
					if (!g)
						break;
					h = g;
				}
				surp += -_bg_lp(l.c[wpos][0], 0, h);
			}
			// fgain: next-word prediction with vs without the span's last word w_t
			double fgain = 0;
			if (t+1 < T) {
				chunk& nxt = l.c[t+1][0];
				const context *hw = _bg->h();  // with w_t (and preceding, depth n-1)
				for (int d = 0; d < _n-1; ++d) {
					int wi = t - d;
					int pid = (wi >= 0) ? l.c[wi][0].wd(0).id : 0;
					const context *g = hw->find(pid);
					if (!g)
						break;
					hw = g;
				}
				const context *hwo = _bg->h(); // without w_t: start one earlier
				for (int d = 0; d < _n-1; ++d) {
					int wi = t - 1 - d;
					int pid = (wi >= 0) ? l.c[wi][0].wd(0).id : 0;
					const context *g = hwo->find(pid);
					if (!g)
						break;
					hwo = g;
				}
				fgain = _bg_lp(nxt, 0, hwo) - _bg_lp(nxt, 0, hw);
			}
			fprintf(stderr, "spanscore %d %d-%d %.4f %.4f\n",
					i, cum[s], cum[t+1], surp, fgain);
		}
	}
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
						bool is_ne = _is_ne_sym(sig);
						int gsig = is_ne ? _ne_generic : sig;
						const context *h = (_n > 1 && sig != 1) ? _bg->h()->find(sig) : NULL;
						const context *hg = (_n > 1 && gsig != 1) ? _bg_gen->h()->find(gsig) : NULL;
						_forward(l, s-prev.len, (h ? h : _bg->h()), (hg ? hg : _bg_gen->h()),
								ch, *k, emit, prev, *q, (s < 0), dp[t][j][*k], dp[s][p][*q], _n-1, is_ne);
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
				bool is_ne = _is_ne_sym(sig);
				int gsig = is_ne ? _ne_generic : sig;
				const context *h = (_n > 1 && sig != 1) ? _bg->h()->find(sig) : NULL;
				const context *hg = (_n > 1 && gsig != 1) ? _bg_gen->h()->find(gsig) : NULL;
				_backward(l, t-prev.len, (h ? h : _bg->h()), (hg ? hg : _bg_gen->h()),
						*ch, ch->k, prev, *q, (t < 0), table[jd], dp[t][p][*q], _n-1, is_ne);
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
	_bg_gen->estimate(iter); // d,theta only; no base corpus so gibbs is a no-op
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
	// frequency gate: effective cap + how many real word types it bars.
	if (_freq_cap > 0) {
		int blocked = 0;
		for (auto& kv : _wfreq)
			if (kv.first > 1 && kv.second > _freq_cap)
				++blocked;
		fprintf(stderr, "[snpylm] freq_cap=%d blocked_types=%d wfreq_types=%zu\n",
				_freq_cap, blocked, _wfreq.size());
	} else {
		fprintf(stderr, "[snpylm] freq_cap=0(disabled) wfreq_types=%zu\n",
				_wfreq.size());
	}
	// soft NE gate (WO-012): the per-type g^_ct actually in force plus its NE / O
	// segment counts. In --gate_learn mode this is the drift monitor: watch a type
	// that starts at gate_lo (e.g. hira) climb epoch by epoch, and note that once
	// g^ passes 0.5 the log-odds turns positive and the prior self-reinforces.
	if (_gate_score_on()) {
		fprintf(stderr, "[snpylm] gate hi=%.4f lo=%.4f strength=%.1f learn=%s\n",
				_gate_hi, _gate_lo, _gate_strength, _gate_learn ? "on" : "off");
		int ord[chunktype2::n];
		for (int t = 0; t < chunktype2::n; ++t)
			ord[t] = t;
		// insertion sort by g^ descending (chunktype2::n is 32; keep it simple).
		for (int i = 1; i < chunktype2::n; ++i) {
			int key = ord[i];
			double kv = _gate_ghat(key);
			int j = i-1;
			while (j >= 0 && _gate_ghat(ord[j]) < kv) {
				ord[j+1] = ord[j];
				--j;
			}
			ord[j+1] = key;
		}
		const int SHOW = 5;
		for (int r = 0; r < SHOW && r < chunktype2::n; ++r) {
			int t = ord[r];
			fprintf(stderr, "  [gate top%d] %s g=%.4f dlt=%+.3f ne=%d o=%d\n",
					r+1, CTNAME[t], _gate_ghat(t), _gate_lp(t),
					_gate_ne[t], _gate_o[t]);
		}
		for (int r = 0; r < SHOW && chunktype2::n-1-r >= SHOW; ++r) {
			int t = ord[chunktype2::n-1-r];
			fprintf(stderr, "  [gate bot%d] %s g=%.4f dlt=%+.3f ne=%d o=%d\n",
					r+1, CTNAME[t], _gate_ghat(t), _gate_lp(t),
					_gate_ne[t], _gate_o[t]);
		}
	} else {
		fprintf(stderr, "[snpylm] gate disabled (%s)\n",
				!_type_admission ? "neutral" :
				(_hard_type_admission ? "hard gate" : "legacy load"));
	}
	static const char *tname[chartype::n] = {
		"hira", "kata", "k/h", "kanji", "h+k", "h+kj", "k+kj", "hkk", "misc",
		"arab", "grk", "hang", "hebr", "latn", "myan", "thai", "digit", "punc", "sym"
	};
	for (int i = 1; i <= _k; ++i) {
		if (_necnt[i] == 0 && _rho[i] == 0)
			continue;
		fprintf(stderr, "  [class %d] ne_id=%d spans=%d chars=%.0f rho=%d lambda=%.3f",
				i, _nek[i], _necnt[i], _nelen[i], _rho[i], _lambda[i]);
		// top char-type shares of this class' surfaces
		double tot = 0;
		for (int t = 0; t < chartype::n; ++t)
			tot += _psi[_psii(i, t)];
		if (tot > 0) {
			int best = -1, second = -1;
			for (int t = 0; t < chartype::n; ++t) {
				if (best < 0 || _psi[_psii(i, t)] > _psi[_psii(i, best)]) {
					second = best; best = t;
				} else if (second < 0 || _psi[_psii(i, t)] > _psi[_psii(i, second)]) {
					second = t;
				}
			}
			fprintf(stderr, " top:%s=%.2f,%s=%.2f", tname[best],
					_psi[_psii(i, best)]/tot, tname[second], _psi[_psii(i, second)]/tot);
		}
		// theta (WO-009): per-class top-2 chunk types and their mass share.
		const char **ctname = CTNAME;
		double ttot = 0;
		for (int t = 0; t < chunktype2::n; ++t)
			ttot += _theta_k[_thetaki(i, t)];
		if (ttot > 0) {
			int tb = -1, ts = -1;
			for (int t = 0; t < chunktype2::n; ++t) {
				if (tb < 0 || _theta_k[_thetaki(i, t)] > _theta_k[_thetaki(i, tb)]) {
					ts = tb; tb = t;
				} else if (ts < 0 || _theta_k[_thetaki(i, t)] > _theta_k[_thetaki(i, ts)]) {
					ts = t;
				}
			}
			fprintf(stderr, " ct:%s=%.2f,%s=%.2f", ctname[tb],
					_theta_k[_thetaki(i, tb)]/ttot, ctname[ts], _theta_k[_thetaki(i, ts)]/ttot);
		}
		fprintf(stderr, "\n");
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
		// psi char-type counts, appended last so legacy models (which lack them)
		// load as all-zero (EOF).
		int pn = (int)_psi.size();
		if (fwrite(&pn, sizeof(int), 1, fp) != 1)
			throw "failed to write psi size in snpylm::save";
		if (pn > 0 && fwrite(_psi.data(), sizeof(int), pn, fp) != (size_t)pn)
			throw "failed to write psi counts in snpylm::save";
		// frequency gate (cap + word-type table), appended last so legacy
		// models load as EOF -> gate disabled (see load()).
		if (fwrite(&_freq_cap, sizeof(int), 1, fp) != 1)
			throw "failed to write freq_cap in snpylm::save";
		int wn = (int)_wfreq.size();
		if (fwrite(&wn, sizeof(int), 1, fp) != 1)
			throw "failed to write wfreq size in snpylm::save";
		if (wn > 0) {
			std::vector<int> buf;
			buf.reserve((size_t)wn*2);
			for (auto& kv : _wfreq) {
				buf.push_back(kv.first);
				buf.push_back(kv.second);
			}
			if (fwrite(buf.data(), sizeof(int), buf.size(), fp) != buf.size())
				throw "failed to write wfreq pairs in snpylm::save";
		}
		// generic NE-slot backoff, appended last (WO-007 gvid-keyed targeted
		// ledger). A negative magic marks the new format; a legacy WO-006 block
		// began with the positive _ne_generic id, so load() can tell them apart
		// and disable the incompatible legacy ledger. Layout after the magic:
		// {ne_generic, backoff flag} (int), gen_w (double), then _bg_gen.
		int magic = SNPYLM_GEN_MAGIC;
		if (fwrite(&magic, sizeof(int), 1, fp) != 1)
			throw "failed to write generic magic in snpylm::save";
		int gg[2] = {_ne_generic, _generic_backoff ? 1 : 0};
		if (fwrite(gg, sizeof(int), 2, fp) != 2)
			throw "failed to write generic header in snpylm::save";
		if (fwrite(&_gen_w, sizeof(double), 1, fp) != 1)
			throw "failed to write gen_w in snpylm::save";
		_bg_gen->save(fp);
		// probabilistic chunktype NE prior theta (WO-009), appended last. Layout
		// after the magic: {ct_n, K} (int), {theta_hi, theta_lo, theta_kappa}
		// (double), q_cand[ct_n] (double), _theta_sh[ct_n] (int), then _theta_k
		// size + data (int), then {theta_enabled, hard_type_admission} flags (int).
		int tmagic = SNPYLM_THETA_MAGIC;
		if (fwrite(&tmagic, sizeof(int), 1, fp) != 1)
			throw "failed to write theta magic in snpylm::save";
		int tdim[2] = {chunktype2::n, _k};
		if (fwrite(tdim, sizeof(int), 2, fp) != 2)
			throw "failed to write theta dims in snpylm::save";
		double th[3] = {_theta_hi, _theta_lo, _theta_kappa};
		if (fwrite(th, sizeof(double), 3, fp) != 3)
			throw "failed to write theta hyper in snpylm::save";
		if (fwrite(_qcand.data(), sizeof(double), chunktype2::n, fp) != (size_t)chunktype2::n)
			throw "failed to write qcand in snpylm::save";
		if (fwrite(_theta_sh.data(), sizeof(int), chunktype2::n, fp) != (size_t)chunktype2::n)
			throw "failed to write theta_sh in snpylm::save";
		int tkn = (int)_theta_k.size();
		if (fwrite(&tkn, sizeof(int), 1, fp) != 1)
			throw "failed to write theta_k size in snpylm::save";
		if (tkn > 0 && fwrite(_theta_k.data(), sizeof(int), tkn, fp) != (size_t)tkn)
			throw "failed to write theta_k in snpylm::save";
		int tf[2] = {_theta_enabled ? 1 : 0, _hard_type_admission ? 1 : 0};
		if (fwrite(tf, sizeof(int), 2, fp) != 2)
			throw "failed to write theta flags in snpylm::save";
		// per-chunktype soft NE gate (WO-012), appended last. Layout after the
		// magic: {ct_n, gate_enabled, gate_learn} (int), {gate_hi, gate_lo,
		// gate_strength} (double), _gate_ne[ct_n] (int), _gate_o[ct_n] (int).
		int gmagic = SNPYLM_GATE_MAGIC;
		if (fwrite(&gmagic, sizeof(int), 1, fp) != 1)
			throw "failed to write gate magic in snpylm::save";
		int gdim[3] = {chunktype2::n, _gate_enabled ? 1 : 0, _gate_learn ? 1 : 0};
		if (fwrite(gdim, sizeof(int), 3, fp) != 3)
			throw "failed to write gate dims in snpylm::save";
		double gh[3] = {_gate_hi, _gate_lo, _gate_strength};
		if (fwrite(gh, sizeof(double), 3, fp) != 3)
			throw "failed to write gate hyper in snpylm::save";
		if (fwrite(_gate_ne.data(), sizeof(int), chunktype2::n, fp) != (size_t)chunktype2::n)
			throw "failed to write gate_ne in snpylm::save";
		if (fwrite(_gate_o.data(), sizeof(int), chunktype2::n, fp) != (size_t)chunktype2::n)
			throw "failed to write gate_o in snpylm::save";
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
		// psi char-type counts; absent in legacy models -> keep the zero-init
		// table (fread failure is not an error here).
		_psi.assign((size_t)(_k+1)*chartype::n, 0);
		// theta ledger defaults (WO-009): sized from _k, overwritten below if the
		// model carries a theta block. Legacy models keep these zero/uniform values.
		_theta_sh.assign(chunktype2::n, 0);
		_theta_k.assign((size_t)(_k+1)*chunktype2::n, 0);
		_qcand.assign(chunktype2::n, 1.0/(double)chunktype2::n);
		_theta_hi = THETA_HI;
		_theta_lo = THETA_LO;
		_theta_kappa = THETA_KAPPA;
		int pn = 0;
		if (fread(&pn, sizeof(int), 1, fp) == 1 && pn > 0) {
			_psi.resize(pn, 0);
			if (fread(_psi.data(), sizeof(int), pn, fp) != (size_t)pn)
				throw "failed to read psi counts in snpylm::load";
		}
		// frequency gate; absent in legacy models -> disabled + empty table.
		_freq_cap = 0;
		_wfreq.clear();
		int fc = 0;
		if (fread(&fc, sizeof(int), 1, fp) == 1) {
			_freq_cap = fc;
			int wn = 0;
			if (fread(&wn, sizeof(int), 1, fp) == 1 && wn > 0) {
				std::vector<int> buf((size_t)wn*2, 0);
				if (fread(buf.data(), sizeof(int), buf.size(), fp) != buf.size())
					throw "failed to read wfreq pairs in snpylm::load";
				_wfreq.reserve(wn);
				for (int i = 0; i < wn; ++i)
					_wfreq[buf[2*i]] = buf[2*i+1];
			}
		}
		// generic NE-slot backoff. Three cases distinguished by the first int:
		//   == SNPYLM_GEN_MAGIC : new WO-007 gvid-keyed block -> load fully.
		//   >  0 (a positive _ne_generic id) : legacy WO-006 tvid-keyed block.
		//     Its ledger semantics are incompatible, so disable the backoff and
		//     keep _bg_gen empty (a one-line warning); the rest of the legacy
		//     block is intentionally not read (it is the last block in the file).
		//   fread fails (pre-generic legacy) : disabled, empty _bg_gen.
		// _gen_w keeps its constructed default (0.5) unless the new block sets it.
		_generic_backoff = false;
		bool generic_consumed = false; // true iff the file position is known past here
		int first = 0;
		if (fread(&first, sizeof(int), 1, fp) == 1) {
			if (first == SNPYLM_GEN_MAGIC) {
				int gg[2] = {0, 0};
				if (fread(gg, sizeof(int), 2, fp) != 2)
					throw "failed to read generic header in snpylm::load";
				if (gg[0] > 0)
					_ne_generic = gg[0];
				_generic_backoff = (gg[1] != 0);
				if (fread(&_gen_w, sizeof(double), 1, fp) != 1)
					throw "failed to read gen_w in snpylm::load";
				_bg_gen->load(fp);
				generic_consumed = true;
			} else if (first > 0) {
				fprintf(stderr, "snpylm::load: legacy WO-006 generic block "
					"detected; generic backoff disabled (incompatible ledger)\n");
				// the rest of that legacy block is not read; do not attempt theta.
			}
		}
		// probabilistic chunktype NE prior theta (WO-009). Only reachable when the
		// generic block was the new format (position known). Absent (EOF) or a
		// legacy model -> theta disabled + hard-gate fallback (one-line warning).
		_theta_enabled = false;
		_hard_type_admission = false;
		bool theta_loaded = false;
		if (generic_consumed) {
			int tmagic = 0;
			if (fread(&tmagic, sizeof(int), 1, fp) == 1 && tmagic == SNPYLM_THETA_MAGIC) {
				int tdim[2] = {0, 0};
				if (fread(tdim, sizeof(int), 2, fp) != 2)
					throw "failed to read theta dims in snpylm::load";
				if (tdim[0] != chunktype2::n)
					throw "theta chunktype count mismatch in snpylm::load";
				double th[3] = {0, 0, 0};
				if (fread(th, sizeof(double), 3, fp) != 3)
					throw "failed to read theta hyper in snpylm::load";
				_theta_hi = th[0]; _theta_lo = th[1]; _theta_kappa = th[2];
				_qcand.assign(chunktype2::n, 0);
				if (fread(_qcand.data(), sizeof(double), chunktype2::n, fp) != (size_t)chunktype2::n)
					throw "failed to read qcand in snpylm::load";
				_theta_sh.assign(chunktype2::n, 0);
				if (fread(_theta_sh.data(), sizeof(int), chunktype2::n, fp) != (size_t)chunktype2::n)
					throw "failed to read theta_sh in snpylm::load";
				int tkn = 0;
				if (fread(&tkn, sizeof(int), 1, fp) != 1)
					throw "failed to read theta_k size in snpylm::load";
				_theta_k.assign((size_t)(tkn > 0 ? tkn : (_k+1)*chunktype2::n), 0);
				if (tkn > 0 && fread(_theta_k.data(), sizeof(int), tkn, fp) != (size_t)tkn)
					throw "failed to read theta_k in snpylm::load";
				int tf[2] = {0, 0};
				if (fread(tf, sizeof(int), 2, fp) != 2)
					throw "failed to read theta flags in snpylm::load";
				_theta_enabled = (tf[0] != 0);
				_hard_type_admission = (tf[1] != 0);
				theta_loaded = true;
			}
		}
		if (!theta_loaded) {
			fprintf(stderr, "snpylm::load: no theta block (legacy model); NE theta "
				"prior disabled, hard type-admission gate fallback\n");
			_theta_enabled = false;
			_hard_type_admission = true;
		}
		// per-chunktype soft NE gate (WO-012). Same positional rule as theta: only
		// reachable when the theta block was present (file position known). Absent
		// -> gate disabled + hard-gate fallback (one-line warning), matching how a
		// pre-WO-012 model was actually trained.
		_gate_enabled = false;
		_gate_ne.assign(chunktype2::n, 0);
		_gate_o.assign(chunktype2::n, 0);
		bool gate_loaded = false;
		if (theta_loaded) {
			int gmagic = 0;
			if (fread(&gmagic, sizeof(int), 1, fp) == 1 && gmagic == SNPYLM_GATE_MAGIC) {
				int gdim[3] = {0, 0, 0};
				if (fread(gdim, sizeof(int), 3, fp) != 3)
					throw "failed to read gate dims in snpylm::load";
				if (gdim[0] != chunktype2::n)
					throw "gate chunktype count mismatch in snpylm::load";
				_gate_enabled = (gdim[1] != 0);
				_gate_learn = (gdim[2] != 0);
				double gh[3] = {0, 0, 0};
				if (fread(gh, sizeof(double), 3, fp) != 3)
					throw "failed to read gate hyper in snpylm::load";
				_gate_hi = gh[0]; _gate_lo = gh[1]; _gate_strength = gh[2];
				if (fread(_gate_ne.data(), sizeof(int), chunktype2::n, fp) != (size_t)chunktype2::n)
					throw "failed to read gate_ne in snpylm::load";
				if (fread(_gate_o.data(), sizeof(int), chunktype2::n, fp) != (size_t)chunktype2::n)
					throw "failed to read gate_o in snpylm::load";
				gate_loaded = true;
			}
		}
		if (!gate_loaded) {
			fprintf(stderr, "snpylm::load: no gate block (legacy model); soft NE "
				"gate disabled, hard type-admission gate fallback\n");
			_gate_enabled = false;
			_gate_learn = false;
			_hard_type_admission = true;
		}
		_install_cbase();
	} catch (const char *ex) {
		fclose(fp);
		throw ex;
	}
	fclose(fp);
}
