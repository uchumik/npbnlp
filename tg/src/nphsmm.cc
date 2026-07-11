#include"nphsmm.h"
#include"convinience.h"
#include"rd.h"
#include"beta.h"
#include"wordtype.h"
#include"chunktype.h"
#include"negative_binomial.h"
#include<cstdlib>
#include<atomic>
#include<chrono>
#ifdef _OPENMP
#include<omp.h>
#endif

//#define C 50000
//#define K 1000
#define C 1
#define K 20
#define A 1.
#define B 2.
#define L 50
#define ZERO 1e-16
#define CHUNK_CDF_TH 0.999
using namespace std;
using namespace npbnlp;

static unordered_map<int, int> cfreq;
static unordered_map<int, int> kfreq;
static negative_binomial nb;

// diagnostics (env gated): phase wall time and slice survivor stats
static std::atomic<long long> ph_lat(0), ph_slice(0), ph_prior(0), ph_fwd(0), ph_bwd(0), ph_sent(0);
static std::atomic<long long> sl_seg(0), sl_srv(0), sl_max(0);
struct diag_report {
	~diag_report() {
		if (getenv("NPBNLP_PHASE_TIME") && ph_sent > 0)
			fprintf(stderr, "[phase ms] lattice=%lld slice=%lld prior=%lld forward=%lld backward=%lld sentences=%lld\n",
					ph_lat.load()/1000, ph_slice.load()/1000, ph_prior.load()/1000, ph_fwd.load()/1000, ph_bwd.load()/1000, ph_sent.load());
		if (getenv("NPBNLP_SLICE_STATS") && sl_seg > 0)
			fprintf(stderr, "[slice] segments=%lld survivors_avg=%.3f survivors_max=%lld\n",
					sl_seg.load(), (double)sl_srv.load()/sl_seg.load(), sl_max.load());
	}
};
static diag_report _diag;
static inline long long diag_us(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
	return std::chrono::duration_cast<std::chrono::microseconds>(b-a).count();
}

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

nphsmm::nphsmm(): _n(1), _m(2), _l(10), _k(20), _v(C), _K(K), _a(1), _b(1), _original(false), _class(new hpyp(_n)), _chunk(new vector<shared_ptr<hpyp> >), _word(new vector<shared_ptr<hpyp> >), _letter(new vector<shared_ptr<vpyp> >), _prior(new vector<double>(chunktype2::n, 0)), _length(new vector<int>(chunktype2::n, 0)), _num(new vector<int>(chunktype2::n, 0)), _change(new vector<int>(chunktype2::n, 0)), _clength(new vector<int>(chunktype2::n, 0)), _cprior(new vector<double>(chunktype2::n, 1)), _bp(new vector<double>(chunktype2::n*chartype::n, 0)), _bcount(new vector<int>(chunktype2::n*chartype::n, 0)), _ccount(new vector<int>(chunktype2::n*chartype::n, 0)), _posbase(false), _posv(0), _posseq(new vector<shared_ptr<hpyp> >), _ctxj(0), _lctx(new vector<shared_ptr<hpyp> >), _rctx(new vector<shared_ptr<hpyp> >), _lbg(new hpyp(2)), _rbg(new hpyp(2)), _ctxgate(false), _wclass(false), _wc(new std::vector<int>()), _wbeta(1.0) {
	//_class->set_v(K);
	for (auto i = 0; i < _k+1; ++i) {
		_chunk->push_back(shared_ptr<hpyp>(new hpyp(_n)));
		_word->push_back(shared_ptr<hpyp>(new hpyp(_m)));
		_letter->push_back(shared_ptr<vpyp>(new vpyp(_l)));
		//(*_letter)[i]->set_v(_v);
		(*_word)[i]->set_base((*_letter)[i].get());
		(*_chunk)[i]->set_base((*_word)[i].get());
		_posseq->push_back(shared_ptr<hpyp>(new hpyp(_m)));
		_lctx->push_back(shared_ptr<hpyp>(new hpyp(2)));
		_rctx->push_back(shared_ptr<hpyp>(new hpyp(2)));
	}
	beta_distribution be;
	for (auto& p : *_prior) {
		p = 1.-be(A, B);
	}
	for (auto& p : *_cprior) {
		p = 1.-be(A, B);
	}
	for (auto& p : *_bp) {
		p = 1.-be(A, B);
	}
	for (auto k = 0; k < chunktype2::n; ++k) {
		double cdf = 0;
		int l = 1;
		for (; cdf < CHUNK_CDF_TH && l < L; ++l) {
			cdf = nb.cdf((*_prior)[k], 1, l-1);
		}
		(*_clength)[k] = l;
	}
	//_cprior = 1.-be(_change, _clength);
}

nphsmm::nphsmm(int n, int m, int l, int k): _n(n), _m(m), _l(l), _k(k), _v(C), _K(K), _a(1), _b(1), _original(false), _class(new hpyp(_n)), _chunk(new vector<shared_ptr<hpyp> >), _word(new vector<shared_ptr<hpyp> >), _letter(new vector<shared_ptr<vpyp> >), _prior(new vector<double>(chunktype2::n, 0)), _length(new vector<int>(chunktype2::n, 0)), _num(new vector<int>(chunktype2::n, 0)), _change(new vector<int>(chunktype2::n, 0)), _clength(new vector<int>(chunktype2::n, 0)), _cprior(new vector<double>(chunktype2::n, 1)), _bp(new vector<double>(chunktype2::n*chartype::n, 0)), _bcount(new vector<int>(chunktype2::n*chartype::n, 0)), _ccount(new vector<int>(chunktype2::n*chartype::n, 0)), _posbase(false), _posv(0), _posseq(new vector<shared_ptr<hpyp> >), _ctxj(0), _lctx(new vector<shared_ptr<hpyp> >), _rctx(new vector<shared_ptr<hpyp> >), _lbg(new hpyp(2)), _rbg(new hpyp(2)), _ctxgate(false), _wclass(false), _wc(new std::vector<int>()), _wbeta(1.0) {
	//_class->set_v(K);
	for (auto i = 0; i < _k+1; ++i) {
		_chunk->push_back(shared_ptr<hpyp>(new hpyp(_n)));
		_word->push_back(shared_ptr<hpyp>(new hpyp(_m)));
		_letter->push_back(shared_ptr<vpyp>(new vpyp(_l)));
		//(*_letter)[i]->set_v(_v);
		(*_word)[i]->set_base((*_letter)[i].get());
		(*_chunk)[i]->set_base((*_word)[i].get());
		_posseq->push_back(shared_ptr<hpyp>(new hpyp(_m)));
		_lctx->push_back(shared_ptr<hpyp>(new hpyp(2)));
		_rctx->push_back(shared_ptr<hpyp>(new hpyp(2)));
	}
	beta_distribution be;
	for (auto& p : *_prior) {
		p = 1.-be(A, B);
	}
	for (auto& p : *_cprior) {
		p = 1.-be(A, B);
	}
	for (auto& p : *_bp) {
		p = 1.-be(A, B);
	}
	for (auto k = 0; k < chunktype2::n; ++k) {
		double cdf = 0;
		int l = 1;
		for (; cdf < CHUNK_CDF_TH && l < L; ++l) {
			cdf = nb.cdf((*_prior)[k], 1, l-1);
		}
		(*_clength)[k] = l;
	}
	//_cprior = 1.-be(_change, _clength);
}

nphsmm::~nphsmm() {
}

void nphsmm::set_k(int k) {
	if (k > 0)
		_K = k;
}

void nphsmm::set(int v, int k) {
	_v = v;
	_K = k;
	_k = min(_k, _K);
	for (auto it = _letter->begin(); it != _letter->end(); ++it) {
		(*it)->set_v(_v);
	}
	_class->set_v(_K);
}

int nphsmm::n() {
	return _n;
}

int nphsmm::m() {
	return _m;
}

int nphsmm::l() {
	return _l;
}

void nphsmm::slice(double a, double b) {
	if (a <= 0 || b <= 0) {
		return;
	}
	_a = a;
	_b = b;
}

void nphsmm::set_original(bool f) {
	_original = f;
}

void nphsmm::set_posbase(bool f) {
	_posbase = f;
	_install_cbase();
}

void nphsmm::set_ctx(int j) {
	// vocabulary of the context HPYPs is acquired from data (hpyp::add ++_v),
	// so no external V / set_v is needed.
	_ctxj = j;
	while ((int)_lctx->size() < _k+1)
		_lctx->push_back(shared_ptr<hpyp>(new hpyp(2)));
	while ((int)_rctx->size() < _k+1)
		_rctx->push_back(shared_ptr<hpyp>(new hpyp(2)));
}

void nphsmm::set_ctxgate(bool f) {
	_ctxgate = f;
}

// logsumexp of the raw context factor (lctx+rctx) over the surviving classes at
// segment (t,j). Used as the softmax-gate normalizer: gate_k = ctx_k - _ctx_norm,
// so sum_k exp(gate_k) = 1 and the factor cannot inflate chunk-existence mass.
double nphsmm::_ctx_norm(clattice2& l, int t, int j, int start) {
	double z = 0;
	bool init = true;
	for (auto k = l.begin(t, j); k != l.end(t, j); ++k) {
		z = math::lse(z, l.lctx[start][*k]+l.rctx[t][*k], init);
		init = false;
	}
	return z;
}

void nphsmm::set_wclass(bool f) {
	_wclass = f;
}

void nphsmm::set_wbeta(double b) {
	_wbeta = b;
}

// lazy-size the theta count table once _posv (tokenizer class count) is known.
void nphsmm::_wc_ensure() {
	if ((int)_wc->size() != (_K+1)*(_posv+1))
		_wc->assign((_K+1)*(_posv+1), 0);
}

// per-word tokenizer-class channel: log P(theta) for a chunk under class z.
// P(p|z) = (n_{z,p}+alpha) / (sum_p n_{z,p} + alpha*K_tok); words with pos out of
// [1,_posv] (BOS/EOS/unassigned) contribute 0 (neutral). Added on top of the word LM.
double nphsmm::_wclass_lp(int z, chunk& ch) {
	if (!_wclass || _posv <= 0)
		return 0;
	_wc_ensure();
	const double alpha = 1.0;
	double row = 0;
	for (int p = 1; p <= _posv; ++p)
		row += (*_wc)[_wci(z, p)];
	double denom = row + alpha*_posv;
	double v = 0;
	for (int i = 0; i < ch.len; ++i) {
		int p = ch.wd(i).pos;
		if (p < 1 || p > _posv)
			continue;
		v += log((*_wc)[_wci(z, p)] + alpha) - log(denom);
	}
	return v;
}

// seat (d=+1) / unseat (d=-1) the theta counts for a chunk of latent class ch.k.
void nphsmm::_wclass_count(chunk& ch, int d) {
	if (!_wclass || _posv <= 0)
		return;
	_wc_ensure();
	for (int i = 0; i < ch.len; ++i) {
		int p = ch.wd(i).pos;
		if (p < 1 || p > _posv)
			continue;
		(*_wc)[_wci(ch.k, p)] += d;
	}
}

// B-obs: score each candidate chunk by its surrounding words. lctx[startpos][k]
// = sum_{e=1..j} log psi^L_{k,e}(word[startpos-e]); rctx[endpos][k] symmetric on
// the right. psi = HPYP(2) with context symbol e -> distance/class/uniform backoff.
// Out-of-sentence positions are skipped (matched by _ctx_seat's counting).
void nphsmm::_context_factor(clattice2& l) {
	if (_ctxj <= 0)
		return;
	int T = (int)l.c.size();
	l.lctx.assign(T, vector<double>(_k+1, 0));
	l.rctx.assign(T, vector<double>(_k+1, 0));
	for (int p = 0; p < T; ++p) {
		for (int k = 1; k < _k+1; ++k) {
			double lv = 0, rv = 0;
			for (int e = 1; e <= _ctxj; ++e) {
				if (p-e >= 0) {
					word& w = l.c[p-e][0].wd(0);
					const context *c = (*_lctx)[k]->h()->find(e);
					const context *bc = _lbg->h()->find(e);
					lv += (*_lctx)[k]->lp(w, c ? c : (*_lctx)[k]->h())
						- _lbg->lp(w, bc ? bc : _lbg->h());
				}
				if (p+e < T) {
					word& w = l.c[p+e][0].wd(0);
					const context *c = (*_rctx)[k]->h()->find(e);
					const context *bc = _rbg->h()->find(e);
					rv += (*_rctx)[k]->lp(w, c ? c : (*_rctx)[k]->h())
						- _rbg->lp(w, bc ? bc : _rbg->h());
				}
			}
			l.lctx[p][k] = lv;
			l.rctx[p][k] = rv;
		}
	}
}

// seat (add=true) or unseat the context-word events of the whole sentence's
// current segmentation. Flattens real chunks to a word list, then for each
// chunk [ws..we] (class k) seats the e-distance outside words into _lctx/_rctx.
void nphsmm::_ctx_seat(nsentence& s, bool add) {
	if (_ctxj <= 0)
		return;
	vector<word*> flat;
	vector<int> cstart, cend, cls;
	for (auto i = 0; i < s.size(); ++i) {
		chunk& ch = s.ch(i);
		if (!ch.id || ch.type < 0)
			continue;
		cstart.push_back((int)flat.size());
		for (auto jj = 0; jj < ch.len; ++jj)
			flat.push_back(&ch.wd(jj));
		cend.push_back((int)flat.size()-1);
		cls.push_back(ch.k);
	}
	int F = (int)flat.size();
	for (auto m = 0; m < (int)cls.size(); ++m) {
		int k = cls[m];
		for (int e = 1; e <= _ctxj; ++e) {
			int li = cstart[m]-e, ri = cend[m]+e;
			if (li >= 0) {
				if (add) {
					(*_lctx)[k]->add(*flat[li], (*_lctx)[k]->h()->make(e));
					_lbg->add(*flat[li], _lbg->h()->make(e));
				} else {
					context *f = (*_lctx)[k]->h()->find(e); if (f) (*_lctx)[k]->remove(*flat[li], f);
					context *bf = _lbg->h()->find(e); if (bf) _lbg->remove(*flat[li], bf);
				}
			}
			if (ri < F) {
				if (add) {
					(*_rctx)[k]->add(*flat[ri], (*_rctx)[k]->h()->make(e));
					_rbg->add(*flat[ri], _rbg->h()->make(e));
				} else {
					context *f = (*_rctx)[k]->h()->find(e); if (f) (*_rctx)[k]->remove(*flat[ri], f);
					context *bf = _rbg->h()->find(e); if (bf) _rbg->remove(*flat[ri], bf);
				}
			}
		}
	}
}

void nphsmm::set_lex(std::function<double(word&, int)> f, int posv) {
	_lex = f;
	if (posv > 0) {
		_posv = posv;
		for (auto it = _posseq->begin(); it != _posseq->end(); ++it)
			(*it)->set_v(_posv);
	}
}

// pos-pattern base measure G0(c|z) = prod_j P_posseq_z(p_j|p_{j-1}..) * P_lex(w_j|p_j)
// with within-chunk BOS reset (0 padding) and EOS pos 0, mirroring _lpb(chunk)'s
// BOS/EOS conventions so the slice memoization can reproduce it exactly.
double nphsmm::_posseq_lp(int k, chunk& c) {
	int M = _m;
	static thread_local vector<int> prev;
	prev.assign(M > 1 ? M-1 : 1, 0);
	double v = 0;
	for (int i = 0; i < c.len+1; ++i) {
		int pj = (i < c.len) ? c.wd(i).pos : 0;
		for (int d = 0; d < M-1; ++d)
			prev[d] = (i-1-d >= 0 && i-1-d < c.len) ? c.wd(i-1-d).pos : 0;
		v += (*_posseq)[k]->wlp(pj, prev.data(), M-1);
		if (i < c.len)
			v += (_lex) ? _lex(c.wd(i), c.wd(i).pos) : -log((double)_v);
	}
	return v;
}

// seat/unseat the chunk's pos n-gram events; contexts are made to full depth
// with 0 padding, mirroring hpyp::make(chunk&, i)'s convention.
void nphsmm::_posseq_add(int k, chunk& c) {
	for (int i = 0; i < c.len+1; ++i) {
		int pj = (i < c.len) ? c.wd(i).pos : 0;
		context *h = (*_posseq)[k]->h();
		for (int d = 1; d < _m; ++d) {
			int pd = (i-d >= 0 && i-d < c.len) ? c.wd(i-d).pos : 0;
			h = h->make(pd);
		}
		(*_posseq)[k]->add(pj, h);
	}
}

void nphsmm::_posseq_remove(int k, chunk& c) {
	for (int i = 0; i < c.len+1; ++i) {
		int pj = (i < c.len) ? c.wd(i).pos : 0;
		context *h = (*_posseq)[k]->h();
		for (int d = 1; d < _m; ++d) {
			int pd = (i-d >= 0 && i-d < c.len) ? c.wd(i-d).pos : 0;
			context *f = h->find(pd);
			if (!f)
				break;
			h = f;
		}
		(*_posseq)[k]->remove(pj, h);
	}
}

void nphsmm::_install_cbase() {
	while ((int)_posseq->size() < (int)_chunk->size())
		_posseq->push_back(shared_ptr<hpyp>(new hpyp(_m)));
	if (_posv > 0) {
		for (auto it = _posseq->begin(); it != _posseq->end(); ++it)
			(*it)->set_v(_posv);
	}
	for (auto i = 0; i < (int)_chunk->size(); ++i) {
		if (_posbase) {
			int k = i;
			(*_chunk)[i]->set_cbase([this, k](chunk& c) { return _posseq_lp(k, c); });
			(*_chunk)[i]->set_cbase_add([this, k](chunk& c) { _posseq_add(k, c); });
			(*_chunk)[i]->set_cbase_remove([this, k](chunk& c) { _posseq_remove(k, c); });
		} else {
			(*_chunk)[i]->set_cbase(nullptr);
			(*_chunk)[i]->set_cbase_add(nullptr);
			(*_chunk)[i]->set_cbase_remove(nullptr);
		}
	}
}

void nphsmm::save(const char *file) {
	FILE *fp = NULL;
	if ((fp = fopen(file,"wb")) == NULL)
		throw "failed to open save file in nphsmm::save";
	try {
		if (fwrite(&_n, sizeof(int), 1, fp) != 1)
			throw "failed to write _n in nphsmm::save";
		if (fwrite(&_m, sizeof(int), 1, fp) != 1)
			throw "failed to write _m in nphsmm::save";
		if (fwrite(&_l, sizeof(int), 1, fp) != 1)
			throw "failed to write _l in nphsmm::save";
		if (fwrite(&_k, sizeof(int), 1, fp) != 1)
			throw "failed to write _k in nphsmm::save";
		if (fwrite(&_v, sizeof(int), 1, fp) != 1)
			throw "failed to write _v in nphsmm::save";
		if (fwrite(&_K, sizeof(int), 1, fp) != 1)
			throw "failed to write _K in nphsmm::save";
		if (fwrite(&_a, sizeof(double), 1, fp) != 1)
			throw "failed to write _a in nphsmm::save";
		if (fwrite(&_b, sizeof(double), 1, fp) != 1)
			throw "failed to write _b in nphsmm::save";
		int n = _prior->size();
		if (fwrite(&n, sizeof(int), 1, fp) != 1)
			throw "failed to write prior class num in nphsmm::save";
		if (fwrite(_prior->data(), sizeof(double), n, fp) != n)
			throw "failed to write parameters of duration prior in nphsmm::save";
		if (fwrite(_cprior->data(), sizeof(double), n, fp) != n)
			throw "failed to write parameters of type change prior in nphsmm::save";
		// per-(chunk_type, word_type) boundary probabilities (size = 32*19)
		int bn = _bp->size();
		if (fwrite(&bn, sizeof(int), 1, fp) != 1)
			throw "failed to write bp size in nphsmm::save";
		if (fwrite(_bp->data(), sizeof(double), bn, fp) != bn)
			throw "failed to write bp in nphsmm::save";
		/*
		if (fwrite(&_cprior, sizeof(double), 1, fp) != 1)
			throw "failed to write type change prior in nphsmm::save";
			*/
		_class->save(fp);
		for (auto i = 0; i < _k+1; ++i) {
			(*_chunk)[i]->save(fp);
			(*_word)[i]->save(fp);
			(*_letter)[i]->save(fp);
		}
		// pos-pattern base measure (A-2): flag + per-class pos-seq LMs (appended
		// at the end so models without posbase keep the legacy layout)
		int pb = _posbase ? 1 : 0;
		if (fwrite(&pb, sizeof(int), 1, fp) != 1)
			throw "failed to write posbase flag in nphsmm::save";
		if (pb) {
			for (auto i = 0; i < _k+1; ++i)
				(*_posseq)[i]->save(fp);
		}
		// context-distribution factor (B-obs): radius + per-class L/R LMs + background
		// (vocabulary is stored inside each hpyp's own save)
		int cj = _ctxj;
		if (fwrite(&cj, sizeof(int), 1, fp) != 1)
			throw "failed to write ctx radius in nphsmm::save";
		if (cj > 0) {
			for (auto i = 0; i < _k+1; ++i) {
				(*_lctx)[i]->save(fp);
				(*_rctx)[i]->save(fp);
			}
			_lbg->save(fp);
			_rbg->save(fp);
		}
		// gate flag written last so legacy ctx models (which lack it) load as off.
		int cg = _ctxgate ? 1 : 0;
		if (fwrite(&cg, sizeof(int), 1, fp) != 1)
			throw "failed to write ctx gate flag in nphsmm::save";
		// per-word tokenizer-class channel: flag + _posv (stride) + theta counts,
		// appended last so legacy models (which lack them) load as off.
		int wcf = _wclass ? 1 : 0;
		int pv = _posv;
		int wn = (int)_wc->size();
		if (fwrite(&wcf, sizeof(int), 1, fp) != 1 ||
		    fwrite(&pv, sizeof(int), 1, fp) != 1 ||
		    fwrite(&wn, sizeof(int), 1, fp) != 1)
			throw "failed to write wclass header in nphsmm::save";
		if (wn > 0 && fwrite(_wc->data(), sizeof(int), wn, fp) != (size_t)wn)
			throw "failed to write wclass counts in nphsmm::save";
		// per-word tokenizer-class channel temperature; appended last so legacy
		// models (which lack it) load as 1.0 (see load()).
		if (fwrite(&_wbeta, sizeof(double), 1, fp) != 1)
			throw "failed to write _wbeta in nphsmm::save";
	} catch (const char *ex) {
		throw ex;
	}
	fclose(fp);
}

void nphsmm::load(const char *file) {
	FILE *fp = NULL;
	if ((fp = fopen(file,"rb")) == NULL)
		throw "failed to open save file in nphsmm::load";
	try {
		if (fread(&_n, sizeof(int), 1, fp) != 1)
			throw "failed to read _n in nphsmm::load";
		if (fread(&_m, sizeof(int), 1, fp) != 1)
			throw "failed to read _m in nphsmm::load";
		if (fread(&_l, sizeof(int), 1, fp) != 1)
			throw "failed to read _l in nphsmm::load";
		if (fread(&_k, sizeof(int), 1, fp) != 1)
			throw "failed to read _k in nphsmm::load";
		if (fread(&_v, sizeof(int), 1, fp) != 1)
			throw "failed to read _v in nphsmm::load";
		if (fread(&_K, sizeof(int), 1, fp) != 1)
			throw "failed to read _K in nphsmm::load";
		if (fread(&_a, sizeof(double), 1, fp) != 1)
			throw "failed to read _a in nphsmm::load";
		if (fread(&_b, sizeof(double), 1, fp) != 1)
			throw "failed to read _b in nphsmm::load";
		int n = 0;
		if (fread(&n, sizeof(int), 1, fp) != 1)
			throw "failed to read prior class num in nphsmm::load";
		_prior->resize(n);
		if (fread(_prior->data(), sizeof(double), n, fp) != n)
			throw "failed to read parameters of duration prior in nphsmm::load";
		_cprior->resize(n);
		if (fread(_cprior->data(), sizeof(double), n, fp) != n)
			throw "failed to read parameters of type change prior in nphsmm::load";
		// per-(chunk_type, word_type) boundary probabilities
		int bn = 0;
		if (fread(&bn, sizeof(int), 1, fp) != 1)
			throw "failed to read bp size in nphsmm::load";
		_bp->resize(bn);
		if (fread(_bp->data(), sizeof(double), bn, fp) != bn)
			throw "failed to read bp in nphsmm::load";
		/*
		if (fread(&_cprior, sizeof(double), 1, fp) != 1)
			throw "failed to read type change prior in nphsmm::load";
			*/
		_class->load(fp);
		while ((int)_chunk->size() < _k+1) {
			int k = _chunk->size();
			_chunk->push_back(shared_ptr<hpyp>(new hpyp(_n)));
			_word->push_back(shared_ptr<hpyp>(new hpyp(_m)));
			_letter->push_back(shared_ptr<vpyp>(new vpyp(_l)));
			(*_chunk)[k]->set_base((*_word)[k].get());
			(*_word)[k]->set_base((*_letter)[k].get());
			//(*_letter)[k]->set_v(_v);
		}
		for (auto i = 0; i < _k+1; ++i) {
			(*_chunk)[i]->load(fp);
			(*_word)[i]->load(fp);
			(*_letter)[i]->load(fp);
		}
		// pos-pattern base measure (A-2): flag absent in legacy models -> off.
		// the file decides; a CLI-set _posbase is overridden here.
		int pb = 0;
		if (fread(&pb, sizeof(int), 1, fp) != 1)
			pb = 0;
		_posbase = (pb == 1);
		if (_posbase) {
			while ((int)_posseq->size() < _k+1)
				_posseq->push_back(shared_ptr<hpyp>(new hpyp(_m)));
			for (auto i = 0; i < _k+1; ++i)
				(*_posseq)[i]->load(fp);
		}
		_install_cbase();
		// context-distribution factor (B-obs): flag absent in legacy models -> off.
		int cj = 0;
		if (fread(&cj, sizeof(int), 1, fp) != 1)
			cj = 0;
		_ctxj = cj;
		if (_ctxj > 0) {
			while ((int)_lctx->size() < _k+1)
				_lctx->push_back(shared_ptr<hpyp>(new hpyp(2)));
			while ((int)_rctx->size() < _k+1)
				_rctx->push_back(shared_ptr<hpyp>(new hpyp(2)));
			for (auto i = 0; i < _k+1; ++i) {
				(*_lctx)[i]->load(fp);
				(*_rctx)[i]->load(fp);
			}
			_lbg->load(fp);
			_rbg->load(fp);
		}
		// gate flag written last; absent in legacy models -> additive (off).
		int cg = 0;
		if (fread(&cg, sizeof(int), 1, fp) != 1)
			cg = 0;
		_ctxgate = (cg != 0);
		// per-word tokenizer-class channel: header + counts; absent in legacy
		// models -> off (fread fails, _wclass stays false, _wc stays empty).
		int wcf = 0, pv = 0, wn = 0;
		if (fread(&wcf, sizeof(int), 1, fp) == 1 &&
		    fread(&pv, sizeof(int), 1, fp) == 1 &&
		    fread(&wn, sizeof(int), 1, fp) == 1) {
			_wclass = (wcf != 0);
			if (pv > 0)
				_posv = pv;
			_wc->assign(wn, 0);
			if (wn > 0 && fread(_wc->data(), sizeof(int), wn, fp) != (size_t)wn)
				throw "failed to read wclass counts in nphsmm::load";
		}
		// per-word tokenizer-class channel temperature; absent in legacy models
		// -> keep default 1.0 (fread failure is not an error here).
		double wb = 1.0;
		if (fread(&wb, sizeof(double), 1, fp) == 1)
			_wbeta = wb;
		if (getenv("NPBNLP_LOAD_STATS")) {
			bool cbase_on = (bool)(*_chunk)[1]->has_cbase();
			fprintf(stderr, "[load_stats] posbase=%d ctxj=%d ctxgate=%d wclass=%d posv=%d wbeta=%.3f chunk_base=%s\n",
					_posbase ? 1 : 0, _ctxj, _ctxgate ? 1 : 0, _wclass ? 1 : 0, _posv, _wbeta,
					cbase_on ? "POS(cbase)" : "WORD(_base)");
		}
		// estimate chunk size
		for (auto k = 0; k < chunktype2::n; ++k) {
			double cdf = 0;
			int l = 1;
			for (; cdf < CHUNK_CDF_TH && l < L; ++l) {
				cdf = nb.cdf((*_prior)[k], 1, l-1);
			}
			(*_clength)[k] = l;
		}
		if (getenv("NPBNLP_MODEL_STATS")) {
			fprintf(stderr, "[model] n=%d m=%d l=%d k=%d prior_n=%d\n", _n, _m, _l, _k, (int)_prior->size());
			for (auto k = 0; k < chunktype2::n; ++k) {
				double p = (k < (int)_prior->size()) ? (*_prior)[k] : -1;
				double cp = (k < (int)_cprior->size()) ? (*_cprior)[k] : -1;
				fprintf(stderr, "[prior] type=%d prior=%.6f cprior=%.6f clength=%d", k, p, cp, (*_clength)[k]);
				if (p >= 0) {
					fprintf(stderr, " cdf:");
					int q[6] = {1, 2, 3, 5, 10, 20};
					for (auto i = 0; i < 6; ++i)
						fprintf(stderr, " %d:%.4f", q[i], nb.cdf(p, 1, q[i]-1));
				}
				fprintf(stderr, "\n");
			}
		}
	} catch (const char *ex) {
		throw ex;
	}
	fclose(fp);
}

void nphsmm::init(nsentence& s) {
	lock_guard<mutex> m(_mutex);
	shared_ptr<cid> dic = cid::create();
	for (int i = 0; i < s.size(); ++i) {
		chunk& x = s.ch(i);
		if ((*dic)[x] == 1) { // unk
			x.id = dic->index(x);
		} else {
			x.id = (*dic)[x];
		}
		cfreq[x.id]++;
		context *h = _class->h();
		for (int j = 1; j < _n; ++j) {
			chunk& ch = s.ch(i-j);
			h = h->make(ch.k);
		}
		vector<double> table;
		for (int k = 1; k < _k+1; ++k) {
			const context *c = (*_chunk)[k]->h();
			for (int j = 1; j < _n; ++j) {
				chunk& ch = s.ch(i-j);
				context *d = c->find(ch.id);
				if (!d)
					break;
				c = d;
			}
			double lp = _class->lp(k, h)+(*_chunk)[k]->lp(x, c);
			table.push_back(lp);
		}
		x.k = 1+rd::ln_draw(table);
		if (x.k == _k) {
			_resize();
		}
		int clen = 0;
		int change = 0;
		type t = wordtype::get(x.wd(0));
		for (auto j = 0; j < x.len; ++j) {
			type u = wordtype::get(x.wd(j));
			if (t != u)
				++change;
			if (x.type >= 0 && x.type < chunktype2::n) {
				int idx = x.type*chartype::n + (int)u;
				if (j < x.len-1) (*_ccount)[idx]++;
				else (*_bcount)[idx]++;
			}
			t = u;
		}
		(*_change)[x.type] += change;
		(*_clength)[x.type] += x.len;
		context *c = (*_chunk)[x.k]->make(s, i);
		(*_chunk)[x.k]->add(x, c);
		_class->add(x.k, h);
		_wclass_count(x, 1);
		kfreq[x.k]++;
		(*_length)[x.type] += x.len;
		(*_num)[x.type] += x.len-1;
	}
	// eos
	context *h = (*_chunk)[0]->make(s, s.size());
	(*_chunk)[0]->add(s.ch(s.size()), h);
	context *c = _class->h();
	int eos = s.size();
	for (int j = 1; j < _n; ++j) {
		chunk& ch = s.ch(eos-j);
		c = c->make(ch.k);
	}
	_class->add(s.ch(s.size()).k, c);
	_ctx_seat(s, true);
}

void nphsmm::add(nsentence& s) {
	lock_guard<mutex> m(_mutex);
	shared_ptr<cid> dic = cid::create();
	for (auto i = 0; i < s.size(); ++i) {
		chunk& ch = s.ch(i);
		if ((*dic)[ch] == 1) {
			ch.id = dic->index(ch);
		} else {
			ch.id = (*dic)[ch];
		}
		cfreq[ch.id]++;
		kfreq[ch.k]++;
	}
	int rd[s.size()+1] = {0};
	rd::shuffle(rd, s.size()+1);
	for (int i = 0; i < s.size()+1; ++i) {
		chunk& ch = s.ch(rd[i]);
		while (_k <= ch.k && _k < _K)
			_resize();
		context *h = (*_chunk)[ch.k]->make(s, rd[i]);
		(*_chunk)[ch.k]->add(ch, h);
		context *c = _class->h();
		for (int j = 1; j < _n; ++j) {
			chunk& x = s.ch(rd[i]-j);
			c = c->make(x.k);
		}
		_class->add(ch.k, c);
		if (!ch.id || ch.type < 0) // skip eos
			continue;
		_wclass_count(ch, 1);
		int clen = 0;
		int change = 0;
		type t = wordtype::get(ch.wd(0));
		for (auto j = 0; j < ch.len; ++j) {
			type u = wordtype::get(ch.wd(j));
			if (t != u)
				++change;
			int idx = ch.type*chartype::n + (int)u;
			if (j < ch.len-1) (*_ccount)[idx]++;
			else (*_bcount)[idx]++;
			t = u;
		}
		(*_num)[ch.type] += ch.len-1;
		(*_length)[ch.type] += ch.len;
		(*_change)[ch.type] += change;
		//(*_clength)[ch.type] += ch.len;
	}
	_ctx_seat(s, true);
}

void nphsmm::remove(nsentence& s) {
	lock_guard<mutex> m(_mutex);
	_ctx_seat(s, false);
	shared_ptr<cid> dic = cid::create();
	for (auto i = 0; i < s.size(); ++i) {
		chunk& ch = s.ch(i);
		cfreq[ch.id]--;
		kfreq[ch.k]--;
		if (cfreq[ch.id] == 0) {
			dic->remove(ch);
			cfreq.erase(ch.id);
		}
	}
	for (int i = 0; i < s.size()+1; ++i) {
		chunk& ch = s.ch(i);
		context *h = (*_chunk)[ch.k]->find(s, i);
		(*_chunk)[ch.k]->remove(ch, h);
		context *c = _class->h();
		for (int j = 1; j < _n; ++j) {
			chunk& x = s.ch(i-j);
			c = c->find(x.k);
		}
		_class->remove(ch.k, c);
		if (!ch.id || ch.type < 0) // skip eos
			continue;
		_wclass_count(ch, -1);
		int clen = 0;
		int change = 0;
		type t = wordtype::get(ch.wd(0));
		for (auto j = 0; j < ch.len; ++j) {
			type u = wordtype::get(ch.wd(j));
			if (t != u)
				++change;
			int idx = ch.type*chartype::n + (int)u;
			if (j < ch.len-1) (*_ccount)[idx]--;
			else (*_bcount)[idx]--;
			t = u;
		}
		(*_num)[ch.type] -= ch.len-1;
		(*_length)[ch.type] -= ch.len;
		(*_change)[ch.type] -= change;
		//(*_clength)[ch.type] -= ch.len;
	}
	for (int k = _k-1; kfreq[k] == 0; --k) {
		_shrink();
	}
}

void nphsmm::estimate(int iter) {
	for (int i = 1; i < _k+1; ++i) {
		(*_chunk)[i]->gibbs(iter);
		(*_word)[i]->gibbs(iter);
		(*_chunk)[i]->estimate(iter);
		(*_word)[i]->estimate(iter);
		(*_letter)[i]->estimate(iter);
		if (_posbase)
			(*_posseq)[i]->estimate(iter);
	}
	_class->estimate(iter);
	// chunk length prior
	beta_distribution be;
	for (auto t = 0; t < chunktype2::n; ++t) {
		(*_cprior)[t] = 1.-be(A+(*_change)[t], B+(*_length)[t]);
		// per-(chunk_type, word_type) boundary probability: Beta-Bernoulli
		// posterior draw (successes = boundary events, failures = continue events).
		int bsum = 0, csum = 0;
		for (auto u = 0; u < chartype::n; ++u) {
			int idx = t*chartype::n + u;
			(*_bp)[idx] = be(A+(*_bcount)[idx], B+(*_ccount)[idx]);
			bsum += (*_bcount)[idx];
			csum += (*_ccount)[idx];
		}
		// aggregate boundary rate (correct geometric MLE = boundaries/positions,
		// Beta-smoothed) drives the _clength cap for lattice construction.
		(*_prior)[t] = (double)(A+bsum)/(A+B+bsum+csum);
	}
	for (auto k = 0; k < chunktype2::n; ++k) {
		double cdf = 0;
		int l = 1;
		for (; cdf < CHUNK_CDF_TH && l < L; ++l) {
			cdf = nb.cdf((*_prior)[k], 1, l-1);
		}
		(*_clength)[k] = l;
	}
}

void nphsmm::poisson_correction(int n) {
	for (int i = 1; i < _k+1; ++i) {
		(*_word)[i]->poisson_correction(n);
	}
}

nsentence nphsmm::parse(nio& f, int i) {
	if (!_original)
		return _minfer(f, i, true);
	clattice2 l(f, i, *_clength);
	vt dp;
	_slice(l);
	_length_prior(l);
	_context_factor(l);
	for (auto t = 0; t < (int)l.c.size(); ++t) {
		for (auto j = 0; j < l.size(t); ++j) {
			chunk& ch = l.ch(t, j+1);
			double prior = l.prior[t][j];
			double ctxz = (_ctxj > 0 && _ctxgate) ? _ctx_norm(l, t, j, t-ch.len+1) : 0;
			for (auto k = l.begin(t, j); k != l.end(t, j); ++k) {
				double pc = prior + ((_ctxj > 0) ? l.lctx[t-ch.len+1][*k]+l.rctx[t][*k]-ctxz : 0)
					+ (_wclass ? _wbeta*_wclass_lp(*k, ch) : 0);
				const context *c = (*_chunk)[*k]->h();
				const context *z = _class->h();
				for (auto p = 0; p < l.size(t-ch.len); ++p) {
					const context *h = NULL;
					chunk& prev = l.ch(t-ch.len, p+1);
					if (_n > 1)
						h = c->find(prev.id);
					for (auto q = l.begin(t-ch.len, p); q != l.end(t-ch.len, p); ++q) {
						const context *u = NULL;
						if (_n > 1)
							u = z->find(*q);
						if (h && u)
							_forward(l, t-ch.len-prev.len, h, u, pc, ch, *k, prev, *q, dp[t][j][*k], dp[t-ch.len][p][*q], _n-1, false, false);
						else if (h)
							_forward(l, t-ch.len-prev.len, h, z, pc, ch, *k, prev, *q, dp[t][j][*k], dp[t-ch.len][p][*q], _n-1, false, true);
						else if (u)
							_forward(l, t-ch.len-prev.len, c, u, pc, ch, *k, prev, *q, dp[t][j][*k], dp[t-ch.len][p][*q], _n-1, true, false);
						else
							_forward(l, t-ch.len-prev.len, c, z, pc, ch, *k, prev, *q, dp[t][j][*k], dp[t-ch.len][p][*q], _n-1, true, true);
					}
				}
			}
		}
	}
	nsentence s;
	chunk *ch = l.cp(l.c.size(), 1);
	int t = (int)l.c.size()-ch->len;
	bool dbg_first = true;
	while (t >= 0) {
		const context *c = (*_chunk)[ch->k]->h();
		const context *z = _class->h();
		vector<double> table;
		vector<int> len;
		vector<int> k;
		for (int p = 0; p < l.size(t); ++p) {
			const context *h = NULL;
			chunk& prev = l.ch(t, p+1);
			if (_n > 1)
				h = c->find(prev.id);
			for (auto q = l.begin(t, p); q != l.end(t, p); ++q) {
				const context *u = NULL;
				int j = table.size();
				table.push_back(1.);
				len.push_back(p+1);
				k.push_back(*q);
				if (_n > 1)
					u = z->find(*q);
				if (h && u)
					_backward(l, t-prev.len, h, u, *ch, ch->k, prev, *q, table[j], dp[t][p][*q], _n-1, false, false);
				else if (h)
					_backward(l, t-prev.len, h, z, *ch, ch->k, prev, *q, table[j], dp[t][p][*q], _n-1, false, true);
				else if (u)
					_backward(l, t-prev.len, c, u, *ch, ch->k, prev, *q, table[j], dp[t][p][*q], _n-1, true, false);
				else
					_backward(l, t-prev.len, c, z, *ch, ch->k, prev, *q, table[j], dp[t][p][*q], _n-1, true, true);
			}
		}
		if (dbg_first) {
			dbg_lk(table);
			dbg_first = false;
		}
		int id = rd::best(table);
		ch = l.cp(t, len[id]);
		ch->k = k[id];
		s.c.push_back(*ch);
		t -= ch->len;
	}
	reverse(s.c.begin(), s.c.end());
	s.n.resize(s.c.size(), 0);
	return s;
}

nsentence nphsmm::sample(nio& f, int i) {
	if (!_original)
		return _minfer(f, i, false);
	clattice2 l(f, i, *_clength);
	vt dp;
	_slice(l);
	_length_prior(l);
	_context_factor(l);
	for (auto t = 0; t < (int)l.c.size(); ++t) {
		for (auto j = 0; j < l.size(t); ++j) {
			chunk& ch = l.ch(t, j+1);
			double prior = l.prior[t][j];
			double ctxz = (_ctxj > 0 && _ctxgate) ? _ctx_norm(l, t, j, t-ch.len+1) : 0;
			for (auto k = l.begin(t, j); k != l.end(t, j); ++k) {
				double pc = prior + ((_ctxj > 0) ? l.lctx[t-ch.len+1][*k]+l.rctx[t][*k]-ctxz : 0)
					+ (_wclass ? _wbeta*_wclass_lp(*k, ch) : 0);
				const context *c = (*_chunk)[*k]->h();
				const context *z = _class->h();
				for (auto p = 0; p < l.size(t-ch.len); ++p) {
					const context *h = NULL;
					chunk& prev = l.ch(t-ch.len, p+1);
					if (_n > 1)
						h = c->find(prev.id);
					for (auto q = l.begin(t-ch.len, p); q != l.end(t-ch.len, p); ++q) {
						const context *u = NULL;
						if (_n > 1)
							u = z->find(*q);
						if (h && u)
							_forward(l, t-ch.len-prev.len, h, u, pc, ch, *k, prev, *q, dp[t][j][*k], dp[t-ch.len][p][*q], _n-1, false, false);
						else if (h)
							_forward(l, t-ch.len-prev.len, h, z, pc, ch, *k, prev, *q, dp[t][j][*k], dp[t-ch.len][p][*q], _n-1, false, true);
						else if (u)
							_forward(l, t-ch.len-prev.len, c, u, pc, ch, *k, prev, *q, dp[t][j][*k], dp[t-ch.len][p][*q], _n-1, true, false);
						else
							_forward(l, t-ch.len-prev.len, c, z, pc, ch, *k, prev, *q, dp[t][j][*k], dp[t-ch.len][p][*q], _n-1, true, true);
					}
				}
			}
		}
	}
	nsentence s;
	chunk *ch = l.cp(l.c.size(), 1);
	int t = (int)l.c.size()-ch->len;
	while (t >= 0) {
		const context *c = (*_chunk)[ch->k]->h();
		const context *z = _class->h();
		vector<double> table;
		vector<int> len;
		vector<int> k;
		for (int p = 0; p < l.size(t); ++p) {
			const context *h = NULL;
			chunk& prev = l.ch(t, p+1);
			if (_n > 1)
				h = c->find(prev.id);
			for (auto q = l.begin(t, p); q != l.end(t, p); ++q) {
				const context *u = NULL;
				int j = table.size();
				table.push_back(1.);	
				len.push_back(p+1);
				k.push_back(*q);
				if (_n > 1)
					u = z->find(*q);
				if (h && u)
					_backward(l, t-prev.len, h, u, *ch, ch->k, prev, *q, table[j], dp[t][p][*q], _n-1, false, false);
				else if (h)
					_backward(l, t-prev.len, h, z, *ch, ch->k, prev, *q, table[j], dp[t][p][*q], _n-1, false, true);
				else if (u)
					_backward(l, t-prev.len, c, u, *ch, ch->k, prev, *q, table[j], dp[t][p][*q], _n-1, true, false);
				else
					_backward(l, t-prev.len, c, z, *ch, ch->k, prev, *q, table[j], dp[t][p][*q], _n-1, true, true);
			}
		}
		int id = rd::ln_draw(table);
		ch = l.cp(t, len[id]);
		ch->k = k[id];
		s.c.push_back(*ch);
		t -= ch->len;
	}
	reverse(s.c.begin(), s.c.end());
	s.n.resize(s.c.size(), 0);
	return s;
}

void nphsmm::_forward(clattice2& l, int i, const context *c, const context *z, double& prior, chunk& ch, int k, chunk& prev, int q, vt& a, vt& b, int n, bool unk, bool not_exist) {
	if (n <= 1) {
		a.v = math::lse(a.v, b.v+(*_chunk)[k]->lp(ch, c)+_class->lp(k, z)+prior, !a.is_init());
		if (!a.is_init())
			a.set(true);
	} else {
		for (auto j = 0; j < l.size(i); ++j) {
			chunk& y = l.ch(i, j+1);
			const context *h = NULL;
			if (!unk && n > 1)
				h = c->find(y.id);
			for (auto r = l.begin(i, j); r != l.end(i, j); ++r) {
				const context *u = NULL;
				if (!not_exist && n > 1)
					u = z->find(*r);
				if (h && u)
					_forward(l, i-y.len, h, u, prior, ch, k, y, *r, a[prev.len-1][q], b[j][*r], n-1, false, false);
				else if (h)
					_forward(l, i-y.len, h, z, prior, ch, k, y, *r, a[prev.len-1][q], b[j][*r], n-1, false, true);
				else if (u)
					_forward(l, i-y.len, c, u, prior, ch, k, y, *r, a[prev.len-1][q], b[j][*r], n-1, true, false);
				else
					_forward(l, i-y.len, c, z, prior, ch, k, y, *r, a[prev.len-1][q], b[j][*r], n-1, true, true);
			}
		}
	}
}

void nphsmm::_backward(clattice2& l, int i, const context *c, const context *z, chunk& ch, int k, chunk& prev, int q, double& lpr, vt& b, int n, bool unk, bool not_exist) {
	if (n <= 1) {
		lpr = math::lse(lpr, b.v+(*_chunk)[k]->lp(ch, c)+_class->lp(k, z), (lpr == 1.));
	} else {
		for (auto j = 0; j < l.size(i); ++j) {
			chunk& y = l.ch(i, j+1);
			const context *h = NULL;
			if (!unk && n > 1)
				h = c->find(y.id);
			for (auto r = l.begin(i, j); r != l.end(i, j); ++r) {
				const context *u = NULL;
				if (!not_exist && n > 1)
					u = z->find(*r);
				if (h && u)
					_backward(l, i-y.len, h, u, ch, k, y, *r, lpr, b[j][*r], n-1, false, false);
				else if (h)
					_backward(l, i-y.len, h, z, ch, k, y, *r, lpr, b[j][*r], n-1, false, true);
				else if (u)
					_backward(l, i-y.len, c, u, ch, k, y, *r, lpr, b[j][*r], n-1, true, false);
				else
					_backward(l, i-y.len, c, z, ch, k, y, *r, lpr, b[j][*r], n-1, true, true);
			}
		}
	}
}

void nphsmm::_slice(clattice2& l) {
	static const bool noslice = (getenv("NPBNLP_NOSLICE") != NULL);
	static const bool slcheck = (getenv("NPBNLP_SLICE_CHECK") != NULL);
	beta_distribution be;
	shared_ptr<generator> g = generator::create();
	int T = (int)l.c.size();
	int M = _m; // word n-gram order; emission context depths 0..M-1
	// Memoize the chunk-base (=_lpb) word log-probs per (class, absolute
	// position, context depth). Overlapping chunk candidates share these, so
	// the per-(k,pos,depth) table replaces O(segments*K*len) word-LM calls
	// with O(T*K*M) calls; each chunk base is then a sum of memoized values.
	static thread_local vector<word*> wobj;
	static thread_local vector<int> wid;
	static thread_local vector<double> WL, WLE, lexv;
	static thread_local vector<int> prev;
	wobj.resize(T);
	wid.resize(T);
	for (int p = 0; p < T; ++p) {
		wobj[p] = &l.c[p][0].wd(0);
		// posbase: memoize over tokenizer pos ids instead of word ids
		wid[p] = (_posbase) ? wobj[p]->pos : wobj[p]->id;
	}
	word& eosw = l.ch(-1, 1).wd(0); // static eos word (matches _lpb's b.wd(len))
	size_t stride = (size_t)T*M;
	WL.assign((size_t)(_k+1)*stride, 0);
	WLE.assign((size_t)(_k+1)*stride, 0);
	prev.assign(M > 1 ? M-1 : 1, 0);
	if (_posbase) {
		// class-independent lexical fill-in P_lex(w_p|pos_p), one value per position
		lexv.resize(T);
		for (int p = 0; p < T; ++p)
			lexv[p] = (_lex) ? _lex(*wobj[p], wobj[p]->pos) : -log((double)_v);
	}
	for (int k = 1; k < _k+1; ++k) {
		for (int p = 0; p < T; ++p) {
			for (int d = 0; d < M; ++d) {
				// real word at p with d preceding within-chunk words (then BOS)
				for (int j = 0; j < M-1; ++j)
					prev[j] = (j < d && p-1-j >= 0) ? wid[p-1-j] : 0;
				WL[(size_t)k*stride + (size_t)p*M + d] = (_posbase)
					? (*_posseq)[k]->wlp(wid[p], prev.data(), M-1)+lexv[p]
					: (*_word)[k]->wlp(*wobj[p], prev.data(), M-1);
				// eos term for a chunk ending at p with depth d last words
				for (int j = 0; j < M-1; ++j)
					prev[j] = (j < d && p-j >= 0) ? wid[p-j] : 0;
				WLE[(size_t)k*stride + (size_t)p*M + d] = (_posbase)
					? (*_posseq)[k]->wlp(0, prev.data(), M-1)
					: (*_word)[k]->wlp(eosw, prev.data(), M-1);
			}
		}
	}
	double maxdiff = 0;
	l.emit.resize(l.c.size());
	for (auto t = 0; t < (int)l.c.size(); ++t) {
		l.emit[t].resize(l.c[t].size());
		for (auto c = l.c[t].begin(); c != l.c[t].end(); ++c) {
			int len = c->len;
			int s = t - len + 1;
			int de = min(len, M-1);
			double z = 0;
			vector<double> table;
			l.emit[t][c->len-1].resize(_k+1, 0);
			for (auto k = 1; k < _k+1; ++k) {
				double base = 0;
				for (int i = 0; i < len; ++i)
					base += WL[(size_t)k*stride + (size_t)(s+i)*M + min(i, M-1)];
				base += WLE[(size_t)k*stride + (size_t)t*M + de];
				double clp = (*_chunk)[k]->lp_root_base(*c, base);
				if (slcheck) {
					double ref = (*_chunk)[k]->lp(*c, (*_chunk)[k]->h());
					double diff = fabs(clp-ref);
					if (diff > maxdiff)
						maxdiff = diff;
				}
				l.emit[t][c->len-1][k] = clp;
				double lp = clp+_class->lp(k, _class->h());
				z = math::lse(z, lp, (z==0));
				table.push_back(lp);
			}
			if (noslice) {
				for (auto k = 1; k < _k+1; ++k)
					l.k[t][c->len-1].push_back(k);
				continue;
			}
			// (slice survivor stats are collected at the end of this function)
			for (auto i = table.begin(); i != table.end(); ++i) {
				*i -= z;
			}
			int id = rd::ln_draw(table);
			double mu = log(be(_a, _b))+table[id];
			for (auto i = 0; i < (int)table.size(); ++i) {
				if (table[i] >= mu)
					l.k[t][c->len-1].push_back(i+1);
			}
		}
	}
	if (slcheck)
		fprintf(stderr, "[slice_check] max|clp-ref|=%.3e\n", maxdiff);
	static const bool sst = (getenv("NPBNLP_SLICE_STATS") != NULL);
	if (sst) {
		long long segs = 0, srv = 0, mx = 0;
		for (auto t = 0; t < (int)l.k.size(); ++t) {
			for (auto j = 0; j < (int)l.k[t].size(); ++j) {
				++segs;
				srv += l.k[t][j].size();
				if ((long long)l.k[t][j].size() > mx)
					mx = l.k[t][j].size();
			}
		}
		sl_seg += segs;
		sl_srv += srv;
		long long cur = sl_max.load();
		while (mx > cur && !sl_max.compare_exchange_weak(cur, mx));
	}
}

void nphsmm::_length_prior(clattice2& l) {
	l.prior.resize(l.c.size());
	for (auto t = 0; t < (int)l.c.size(); ++t) {
		l.prior[t].resize(l.c[t].size(), 0);
		for (auto i = 0; i < (int)l.c[t].size(); ++i) {
			chunk& c = l.c[t][i];
			int ct = (c.type >= 0 && c.type < chunktype2::n) ? c.type : chunktype2::n-1;
			int change = 0;
			// per-(chunk_type, word_type) Bernoulli duration: continue after each
			// interior word (log(1-p)), boundary after the last word (log(p)),
			// p = _bp[ct][wordtype]. Folded into the existing type-change loop.
			double ln_dur = 0;
			type tp = wordtype::get(c.wd(0));
			for (auto j = 0; j < c.len; ++j) {
				type u = wordtype::get(c.wd(j));
				if (tp != u)
					++change;
				double p = (*_bp)[ct*chartype::n + (int)u];
				ln_dur += (j < c.len-1) ? log(max(ZERO, 1.0-p)) : log(max(ZERO, p));
				tp = u;
			}
			double pr_chg = max(ZERO, nb.density((*_cprior)[ct], c.len-change, change));
			l.prior[t][i] = ln_dur+log(pr_chg);
		}
	}
}

/*
 * efficient forward filtering with marginalized forward prob
 * dp keys: [t][len][class][λ1..λ_{n-1}][κ1..κ_{n-2}]
 *   λ_i: lengths of the previous i-th chunks (0 means BOS padding)
 *   κ_i: classes of the previous (i+1)-th chunks
 * am keys: [t][len][λ1..λ_{n-2}][class][κ1..κ_{n-2}]
 *   = lse over the deepest length λ_{n-1} of dp (marginalized connection target)
 */
nsentence nphsmm::_minfer(nio& f, int i, bool best) {
	static const bool pht = (getenv("NPBNLP_PHASE_TIME") != NULL);
	auto c0 = std::chrono::steady_clock::now();
	clattice2 l(f, i, *_clength);
	auto c1 = std::chrono::steady_clock::now();
	vt dp;
	vt am;
	vt trm;
	vt bos;
	_slice(l);
	auto c2 = std::chrono::steady_clock::now();
	_length_prior(l);
	_context_factor(l);
	auto c3 = std::chrono::steady_clock::now();
	int nw = _n-1;
	int nc = max(_n-1, 1);
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
	auto c4 = std::chrono::steady_clock::now();
	if (pht) {
		ph_lat += diag_us(c0, c1);
		ph_slice += diag_us(c1, c2);
		ph_prior += diag_us(c2, c3);
		ph_fwd += diag_us(c3, c4);
		++ph_sent;
	}
	// backward sampling
	nsentence s;
	chunk *ch = l.cp(l.c.size(), 1);
	int t = (int)l.c.size()-ch->len;
	if (ch->k < 0 || ch->k > _k)
		ch->k = 0;
	vector<int> lam; // window of context chunk lengths (λ1..λ_{nw})
	vector<int> rcs; // window of context classes (r1..r_{nc})
	{
		// joint draw of the last chunk chain from eos
		vector<double> tbl;
		vector<vector<int> > lpath;
		vector<vector<int> > rpath;
		vector<int> cl;
		vector<int> cr;
		const context *c = (*_chunk)[ch->k]->h();
		_mtable(l, t, nw, nc, c, false, *ch, am[t], trm, cl, cr, tbl, lpath, rpath);
		if (tbl.empty())
			throw "failed to construct backward table in nphsmm::_minfer";
		dbg_lk(tbl);
		int id = (best) ? rd::best(tbl) : rd::ln_draw(tbl);
		lam = lpath[id];
		rcs = rpath[id];
	}
	while (t >= 0) {
		int P = rcs[0];
		int J = 0;
		if (nw == 0) {
			// draw the length of the chunk ending at t
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
				throw "failed to draw a chunk length in nphsmm::_minfer";
			int id = (best) ? rd::best(tb) : rd::ln_draw(tb);
			J = cand[id];
		} else {
			J = lam[0];
		}
		chunk *cur = l.cp(t, J);
		cur->k = P;
		s.c.push_back(*cur);
		int tn = t-cur->len;
		if (tn < 0)
			break;
		int lnew = 0;
		if (nw >= 1) {
			// draw the deepest context chunk length of the dp entry at t
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
				throw "failed to draw a context length in nphsmm::_minfer";
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
			throw "failed to draw a class in nphsmm::_minfer";
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
	if (pht)
		ph_bwd += diag_us(c4, std::chrono::steady_clock::now());
	reverse(s.c.begin(), s.c.end());
	s.n.resize(s.c.size(), 0);
	return s;
}

void nphsmm::_mfill(clattice2& l, vt& dp, vt& am, vt& bos, vt& trm) {
	int nw = _n-1;
	for (auto t = 0; t < (int)l.c.size(); ++t) {
		for (auto j = 0; j < l.size(t); ++j) {
			chunk& ch = l.ch(t, j+1);
			double pi = l.prior[t][j];
			int s = t-ch.len;
			double ctxz = (_ctxj > 0 && _ctxgate) ? _ctx_norm(l, t, j, t-ch.len+1) : 0;
			vt& as = (s < 0) ? bos : am[s];
			if (!as.is_init())
				continue;
			for (auto pt = l.begin(t, j); pt != l.end(t, j); ++pt) {
				int p = *pt;
				const context *c = (*_chunk)[p]->h();
				double lnp = pi+((_n == 1) ? l.emit[t][j][p] : 0)
					+((_ctxj > 0) ? l.lctx[t-ch.len+1][p]+l.rctx[t][p]-ctxz : 0)
					+(_wclass ? _wbeta*_wclass_lp(p, ch) : 0);
				// own length is a kept key of am only when the emission context is non-empty;
				// for nw == 0 it is the deepest length and is marginalized out
				_mchain(l, s, nw, c, false, ch, p, lnp, as, dp[t][ch.len][p], (nw >= 1) ? am[t][ch.len] : am[t], trm);
			}
		}
	}
}

void nphsmm::_mchain(clattice2& l, int pos, int d, const context *c, bool unk, chunk& ch, int p, double lnp, vt& as, vt& dpn, vt& an, vt& trm) {
	if (d <= 0) {
		double base = lnp+((_n > 1) ? (*_chunk)[p]->lp(ch, c) : 0);
		vector<int> rc;
		_mcls(max(_n-1, 1), rc, as, dpn, an[p], trm, p, base);
	} else {
		for (auto it = as.begin(); it != as.end(); ++it) {
			int lam = it->first;
			vt& child = *(it->second);
			if (!child.is_init())
				continue;
			chunk& y = (lam > 0 && pos >= 0) ? l.ch(pos, lam) : l.ch(-1, 1);
			const context *h = (!unk) ? c->find(y.id) : NULL;
			_mchain(l, pos-y.len, d-1, (h) ? h : c, (unk || !h), ch, p, lnp, child, dpn[lam], (d > 1) ? an[lam] : an, trm);
		}
	}
}

void nphsmm::_mcls(int e, vector<int>& rc, vt& as, vt& dpn, vt& an, vt& trm, int p, double base) {
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

double nphsmm::_mtr(int p, vector<int>& rc, vt& trm) {
	vt *n = &trm;
	for (auto r = rc.begin(); r != rc.end(); ++r)
		n = &(*n)[*r];
	vt& leaf = (*n)[p];
	if (!leaf.is_init()) {
		const context *u = _class->h();
		int d = 0;
		for (auto r = rc.begin(); r != rc.end() && d < _n-1; ++r, ++d) {
			const context *f = u->find(*r);
			if (!f)
				break;
			u = f;
		}
		leaf.v = _class->lp(p, u);
		leaf.set(true);
	}
	return leaf.v;
}

void nphsmm::_mtable(clattice2& l, int pos, int d, int e, const context *c, bool unk, chunk& ch, vt& as, vt& trm, vector<int>& cl, vector<int>& cr, vector<double>& tbl, vector<vector<int> >& lpath, vector<vector<int> >& rpath) {
	if (d > 0) {
		for (auto it = as.begin(); it != as.end(); ++it) {
			int lam = it->first;
			vt& child = *(it->second);
			if (!child.is_init())
				continue;
			chunk& y = (lam > 0 && pos >= 0) ? l.ch(pos, lam) : l.ch(-1, 1);
			const context *h = (!unk && _n > 1) ? c->find(y.id) : NULL;
			cl.push_back(lam);
			_mtable(l, pos-y.len, d-1, e, (h) ? h : c, (unk || !h), ch, child, trm, cl, cr, tbl, lpath, rpath);
			cl.pop_back();
		}
	} else if (e > 1) {
		for (auto it = as.begin(); it != as.end(); ++it) {
			vt& child = *(it->second);
			if (!child.is_init())
				continue;
			cr.push_back(it->first);
			_mtable(l, pos, 0, e-1, c, unk, ch, child, trm, cl, cr, tbl, lpath, rpath);
			cr.pop_back();
		}
	} else {
		double em = (*_chunk)[ch.k]->lp(ch, (_n > 1) ? c : (*_chunk)[ch.k]->h());
		for (auto it = as.begin(); it != as.end(); ++it) {
			vt& child = *(it->second);
			if (!child.is_init())
				continue;
			cr.push_back(it->first);
			double tr = _mtr(ch.k, cr, trm);
			tbl.push_back(em+tr+child.v);
			lpath.push_back(cl);
			rpath.push_back(cr);
			cr.pop_back();
		}
	}
}

void nphsmm::_resize() {
	if (_k+1 > _K)
		return;
	++_k;
	_chunk->resize(_k+1, shared_ptr<hpyp>(new hpyp(_n)));
	_word->resize(_k+1, shared_ptr<hpyp>(new hpyp(_m)));
	_letter->resize(_k+1, shared_ptr<vpyp>(new vpyp(_l)));
	_posseq->resize(_k+1, shared_ptr<hpyp>(new hpyp(_m)));
	if (_posv > 0)
		(*_posseq)[_k]->set_v(_posv);
	while ((int)_lctx->size() < _k+1)
		_lctx->push_back(shared_ptr<hpyp>(new hpyp(2)));
	while ((int)_rctx->size() < _k+1)
		_rctx->push_back(shared_ptr<hpyp>(new hpyp(2)));
	(*_chunk)[_k]->set_base((*_word)[_k].get());
	(*_word)[_k]->set_base((*_letter)[_k].get());
	//(*_letter)[_k]->set_v(_v);
	if (_posbase) {
		int k = _k;
		(*_chunk)[_k]->set_cbase([this, k](chunk& c) { return _posseq_lp(k, c); });
		(*_chunk)[_k]->set_cbase_add([this, k](chunk& c) { _posseq_add(k, c); });
		(*_chunk)[_k]->set_cbase_remove([this, k](chunk& c) { _posseq_remove(k, c); });
	}
}

void nphsmm::_shrink() {
	--_k;
	_chunk->pop_back();
	_word->pop_back();
	_letter->pop_back();
	_posseq->pop_back();
	if ((int)_lctx->size() > _k+1)
		_lctx->pop_back();
	if ((int)_rctx->size() > _k+1)
		_rctx->pop_back();
}
