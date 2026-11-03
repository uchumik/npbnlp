#include"nphsmm.h"
#include"convinience.h"
#include"rd.h"
#include"beta.h"
#include"wordtype.h"
#include"chunktype.h"
#include"negative_binomial.h"
#include"lpb_cache.h"
#ifdef _OPENMP
#include<omp.h>
#endif

//#define C 50000
//#define K 1000
#define C 1
#define K 20
// _v is learned from the data, so no vocabulary seed is applied.
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

nphsmm::nphsmm(): _n(1), _m(2), _l(10), _k(20), _v(C), _K(K), _a(1), _b(1), _class(new hpyp(_n)), _chunk(new vector<shared_ptr<hpyp> >), _word(new vector<shared_ptr<hpyp> >), _letter(new vector<shared_ptr<vpyp> >), _prior(new vector<double>(chunktype2::n, 0)), _length(new vector<int>(chunktype2::n, 0)), _num(new vector<int>(chunktype2::n, 0)), _change(new vector<int>(chunktype2::n, 0)), _clength(new vector<int>(chunktype2::n, 0)), _cprior(new vector<double>(chunktype2::n, 1)) {
	for (auto i = 0; i < _k+1; ++i) {
		_chunk->push_back(shared_ptr<hpyp>(new hpyp(_n)));
		_word->push_back(shared_ptr<hpyp>(new hpyp(_m)));
		_letter->push_back(shared_ptr<vpyp>(new vpyp(_l)));
		(*_word)[i]->set_base((*_letter)[i].get());
		(*_chunk)[i]->set_base((*_word)[i].get());
	}
	beta_distribution be;
	for (auto& p : *_prior) {
		p = 1.-be(A, B);
	}
	for (auto& p : *_cprior) {
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

nphsmm::nphsmm(int n, int m, int l, int k): _n(n), _m(m), _l(l), _k(k), _v(C), _K(K), _a(1), _b(1), _class(new hpyp(_n)), _chunk(new vector<shared_ptr<hpyp> >), _word(new vector<shared_ptr<hpyp> >), _letter(new vector<shared_ptr<vpyp> >), _prior(new vector<double>(chunktype2::n, 0)), _length(new vector<int>(chunktype2::n, 0)), _num(new vector<int>(chunktype2::n, 0)), _change(new vector<int>(chunktype2::n, 0)), _clength(new vector<int>(chunktype2::n, 0)), _cprior(new vector<double>(chunktype2::n, 1)) {
	for (auto i = 0; i < _k+1; ++i) {
		_chunk->push_back(shared_ptr<hpyp>(new hpyp(_n)));
		_word->push_back(shared_ptr<hpyp>(new hpyp(_m)));
		_letter->push_back(shared_ptr<vpyp>(new vpyp(_l)));
		(*_word)[i]->set_base((*_letter)[i].get());
		(*_chunk)[i]->set_base((*_word)[i].get());
	}
	beta_distribution be;
	for (auto& p : *_prior) {
		p = 1.-be(A, B);
	}
	for (auto& p : *_cprior) {
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

/*
void nphsmm::set(int v, int k) {
	_v = v;
	_K = k;
	_k = min(_k, _K);
	for (auto it = _letter->begin(); it != _letter->end(); ++it) {
		(*it)->set_v(_v);
	}
	_class->set_v(_K);
}
*/

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
		}
		for (auto i = 0; i < _k+1; ++i) {
			(*_chunk)[i]->load(fp);
			(*_word)[i]->load(fp);
			(*_letter)[i]->load(fp);
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
		lpb_clear();
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
			t = u;
		}
		(*_change)[x.type] += change;
		(*_clength)[x.type] += x.len;
		context *c = (*_chunk)[x.k]->make(s, i);
		(*_chunk)[x.k]->add(x, c);
		_class->add(x.k, h);
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
		int clen = 0;
		int change = 0;
		type t = wordtype::get(ch.wd(0));
		for (auto j = 0; j < ch.len; ++j) {
			type u = wordtype::get(ch.wd(j));
			if (t != u)
				++change;
			t = u;
		}
		(*_num)[ch.type] += ch.len-1;
		(*_length)[ch.type] += ch.len;
		(*_change)[ch.type] += change;
		//(*_clength)[ch.type] += ch.len;
	}
}

void nphsmm::remove(nsentence& s) {
	lock_guard<mutex> m(_mutex);
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
		int clen = 0;
		int change = 0;
		type t = wordtype::get(ch.wd(0));
		for (auto j = 0; j < ch.len; ++j) {
			type u = wordtype::get(ch.wd(j));
			if (t != u)
				++change;
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
	}
	_class->estimate(iter);
	// chunk length prior
	beta_distribution be;
	for (auto t = 0; t < chunktype2::n; ++t) {
		(*_prior)[t] = 1.-be(A+(*_num)[t], B+(*_length)[t]);
		(*_cprior)[t] = 1.-be(A+(*_change)[t], B+(*_length)[t]);
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
	return _minfer(f, i, true, NULL);
}

nsentence nphsmm::sample(nio& f, int i) {
	return _minfer(f, i, false, NULL);
}

nsentence nphsmm::sample(nio& f, int i, nsentence *cur) {
	return _minfer(f, i, false, cur);
}

void nphsmm::_slice(clattice2& l, nsentence *cur) {
	beta_distribution be;
	vector<vector<int> > anchor(l.c.size());
	for (auto t = 0; t < (int)l.c.size(); ++t)
		anchor[t].resize(l.size(t), 0);
	if (cur) {
		int t = -1;
		for (auto c = cur->c.begin(); c != cur->c.end(); ++c) {
			t += c->len;
			if (t >= 0 && t < (int)l.c.size() && c->len <= l.size(t))
				anchor[t][c->len-1] = c->k;
		}
	}
	for (auto t = 0; t < (int)l.c.size(); ++t) {
		for (auto j = 0; j < l.size(t); ++j) {
			chunk& c = l.ch(t, j+1);
			double z = 0;
			vector<double> table;
			for (auto k = 1; k < _k+1; ++k) {
				double lp = (*_chunk)[k]->lp(c, (*_chunk)[k]->h())+_class->lp(k, _class->h());
				z = math::lse(z, lp, (z==0));
				table.push_back(lp);
			}
			for (auto i = table.begin(); i != table.end(); ++i) {
				*i -= z;
			}
			int id = anchor[t][j]-1;
			if (id < 0 || id >= _k)
				id = rd::ln_draw(table);
			double mu = log(be(_a, _b))+table[id];
			for (auto i = 0; i < (int)table.size(); ++i) {
				if (table[i] >= mu)
					l.k[t][j].push_back(i+1);
			}
			if (anchor[t][j] && find(l.k[t][j].begin(), l.k[t][j].end(), anchor[t][j]) == l.k[t][j].end())
				throw "slice removed the current nphsmm state";
		}
	}
}

nsentence nphsmm::_minfer(nio& f, int i, bool best, nsentence *cur) {
	lpb_clear();
	clattice2 l(f, i, *_clength);
	vt dp, am, trm, bos;
	_slice(l, cur);
	_length_prior(l);
	int nw = _n-1;
	int nc = max(_n-1, 1);
	vt *node = &bos;
	for (int d = 0; d < nw+nc; ++d)
		node = &(*node)[0];
	node->v = 0;
	node->set(true);
	_mfill(l, dp, am, bos, trm);

	nsentence s;
	chunk *eos = l.cp(l.c.size(), 1);
	int t = (int)l.c.size()-eos->len;
	vector<int> lam, rcs, cl, cr;
	vector<double> tbl;
	vector<vector<int> > lpath, rpath;
	_mtable(l, t, nw, nc, (*_chunk)[eos->k]->h(), false, *eos,
	        am[t], trm, cl, cr, tbl, lpath, rpath);
	if (tbl.empty())
		throw "failed to construct backward table in nphsmm::_minfer";
	int id = best ? rd::best(tbl) : rd::ln_draw(tbl);
	lam = lpath[id];
	rcs = rpath[id];
	while (t >= 0) {
		int P = rcs[0];
		int J = 0;
		if (nw == 0) {
			vector<double> tb;
			vector<int> cand;
			for (auto it = dp[t].begin(); it != dp[t].end(); ++it) {
				vt *leaf = &(*(it->second))[P];
				for (int d = 1; d < nc; ++d)
					leaf = &(*leaf)[rcs[d]];
				if (leaf->is_init()) {
					cand.push_back(it->first);
					tb.push_back(leaf->v);
				}
			}
			if (tb.empty())
				throw "failed to draw a chunk length in nphsmm::_minfer";
			J = cand[best ? rd::best(tb) : rd::ln_draw(tb)];
		} else {
			J = lam[0];
		}
		chunk *ch = l.cp(t, J);
		ch->k = P;
		s.c.push_back(*ch);
		int tn = t-ch->len;
		if (tn < 0)
			break;
		int lnew = 0;
		if (nw >= 1) {
			vt *lnode = &dp[t];
			lnode = &(*lnode)[J];
			lnode = &(*lnode)[P];
			for (int d = 1; d < nw; ++d)
				lnode = &(*lnode)[lam[d]];
			vector<double> tb;
			vector<int> cand;
			for (auto it = lnode->begin(); it != lnode->end(); ++it) {
				vt *leaf = it->second.get();
				for (int d = 1; d < nc; ++d)
					leaf = &(*leaf)[rcs[d]];
				if (leaf->is_init()) {
					cand.push_back(it->first);
					tb.push_back(leaf->v);
				}
			}
			if (tb.empty())
				throw "failed to draw a context length in nphsmm::_minfer";
			lnew = cand[best ? rd::best(tb) : rd::ln_draw(tb)];
		}
		vt *anode = &am[tn];
		for (int d = 1; d < nw; ++d)
			anode = &(*anode)[lam[d]];
		if (nw >= 1)
			anode = &(*anode)[lnew];
		for (int d = 1; d < nc; ++d)
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
		int rnew = cand[best ? rd::best(tb) : rd::ln_draw(tb)];
		for (int d = 0; d+1 < nw; ++d)
			lam[d] = lam[d+1];
		if (nw >= 1)
			lam[nw-1] = lnew;
		for (int d = 0; d+1 < nc; ++d)
			rcs[d] = rcs[d+1];
		rcs[nc-1] = rnew;
		t = tn;
	}
	reverse(s.c.begin(), s.c.end());
	s.n.resize(s.c.size(), 0);
	return s;
}

void nphsmm::_mfill(clattice2& l, vt& dp, vt& am, vt& bos, vt& trm) {
	int nw = _n-1;
	for (int t = 0; t < (int)l.c.size(); ++t) {
		for (int j = 0; j < l.size(t); ++j) {
			chunk& ch = l.ch(t, j+1);
			int s = t-ch.len;
			vt& as = (s < 0) ? bos : am[s];
			if (!as.is_init())
				continue;
			for (auto pt = l.begin(t, j); pt != l.end(t, j); ++pt) {
				int p = *pt;
				_mchain(l, s, nw, (*_chunk)[p]->h(), false, ch, p, l.prior[t][j], as,
				        dp[t][ch.len][p], (nw >= 1) ? am[t][ch.len] : am[t], trm);
			}
		}
	}
}

void nphsmm::_mchain(clattice2& l, int pos, int d, const context *c, bool unk, chunk& ch, int p, double ln_prior, vt& as, vt& dpn, vt& an, vt& trm) {
	if (d <= 0) {
		vector<int> rc;
		_mcls(max(_n-1, 1), rc, as, dpn, an[p], trm, p, ln_prior+(*_chunk)[p]->lp(ch, c));
		return;
	}
	for (auto it = as.begin(); it != as.end(); ++it) {
		int lam = it->first;
		vt& child = *(it->second);
		if (!child.is_init())
			continue;
		chunk& y = (lam > 0 && pos >= 0) ? l.ch(pos, lam) : l.ch(-1, 1);
		const context *h = !unk ? c->find(y.id) : NULL;
		_mchain(l, pos-y.len, d-1, h ? h : c, unk || !h, ch, p, ln_prior, child,
		        dpn[lam], (d > 1) ? an[lam] : an, trm);
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
			x = math::lse(x, _mtr(p, rc, trm)+child.v, !init);
			rc.pop_back();
			init = true;
		}
		if (!init)
			return;
		dpn.v = base+x;
		dpn.set(true);
		an.v = math::lse(an.v, dpn.v, !an.is_init());
		an.set(true);
		return;
	}
	for (auto it = as.begin(); it != as.end(); ++it) {
		vt& child = *(it->second);
		if (!child.is_init())
			continue;
		rc.push_back(it->first);
		_mcls(e-1, rc, child, dpn[it->first], an[it->first], trm, p, base);
		rc.pop_back();
	}
}

double nphsmm::_mtr(int p, vector<int>& rc, vt& trm) {
	vt *node = &trm;
	for (auto r = rc.begin(); r != rc.end(); ++r)
		node = &(*node)[*r];
	vt& leaf = (*node)[p];
	if (!leaf.is_init()) {
		const context *u = _class->h();
		int d = 0;
		for (auto r = rc.begin(); r != rc.end() && d < _n-1; ++r, ++d) {
			const context *next = u->find(*r);
			if (!next)
				break;
			u = next;
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
			const context *h = !unk ? c->find(y.id) : NULL;
			cl.push_back(lam);
			_mtable(l, pos-y.len, d-1, e, h ? h : c, unk || !h, ch, child, trm, cl, cr, tbl, lpath, rpath);
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
		double em = (*_chunk)[ch.k]->lp(ch, c);
		for (auto it = as.begin(); it != as.end(); ++it) {
			vt& child = *(it->second);
			if (!child.is_init())
				continue;
			cr.push_back(it->first);
			tbl.push_back(em+_mtr(ch.k, cr, trm)+child.v);
			lpath.push_back(cl);
			rpath.push_back(cr);
			cr.pop_back();
		}
	}
}

void nphsmm::_length_prior(clattice2& l) {
	l.prior.resize(l.c.size());
	for (auto t = 0; t < (int)l.c.size(); ++t) {
		l.prior[t].resize(l.c[t].size(), 0);
		for (auto i = 0; i < (int)l.c[t].size(); ++i) {
			chunk& c = l.c[t][i];
			/*
			   double ln_prior = 0;
			   */
			int change = 0;
			int len = 0;
			type tp = wordtype::get(c.wd(0));
			for (auto j = 0; j < c.len; ++j) {
				type u = wordtype::get(c.wd(j));
				if (tp != u)
					++change;
				tp = u;
			}
			double pr_dur = max(ZERO, nb.density((*_prior)[c.type], 1, c.len-1));
			double pr_chg = max(ZERO, nb.density((*_cprior)[c.type], c.len-change, change));
			l.prior[t][i] = log(pr_dur)+log(pr_chg);
			//l.prior[t][i] = log(nb.density((*_prior)[c.type], 1, c.len-1)) + log(nb.density((*_cprior)[c.type], c.len-change, change));
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
	(*_chunk)[_k]->set_base((*_word)[_k].get());
	(*_word)[_k]->set_base((*_letter)[_k].get());
}

void nphsmm::_shrink() {
	--_k;
	_chunk->pop_back();
	_word->pop_back();
	_letter->pop_back();
}
