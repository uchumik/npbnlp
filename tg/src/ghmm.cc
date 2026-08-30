#include "ghmm.h"
#include "rd.h"
#include <cmath>
#include <iostream>

using namespace std;
using namespace npbnlp;

#define GHMM_STATES 50

ghmm::ghmm(): vhmm(), _d(0), _obs(nullptr) {
}

ghmm::ghmm(int n, int min_n, int k, int d): vhmm(n, min_n, 1, k), _d(d), _obs(new niw(d)) {
	if (d < 1)
		throw "invalid ghmm dimension";
	_obs->resize(_k+1);
}

ghmm::~ghmm() {
}

int ghmm::dim() const { return _d; }
int ghmm::seq_count() const { return (int)_shadow.size(); }

void ghmm::set_sample_mode(bool f) { if (_obs) _obs->set_sample_mode(f); }
void ghmm::seed_sample() { if (_obs) _obs->estimate(); }
void ghmm::init_prior(gio& f, double kappa0) { if (_obs) _obs->init_prior_from_data(f, kappa0); }
void ghmm::set_prior(const vector<double>& mu0, double kappa0, double nu0, double lambda) {
	if (_obs) _obs->set_prior(mu0, kappa0, nu0, lambda);
}

void ghmm::_build_shadow(gio& f) {
	_x.clear();
	_shadow.clear();
	for (auto& g : f.seq) {
		sentence s;
		for (int i = 0; i < g.size(); ++i) {
			word w;
			w.id = (int)_x.size();
			w.len = 1;
			w.pos = 0;
			w.n = 1;
			s.w.push_back(w);
			_x.push_back(g.v[i]);
		}
		s.n.resize(s.size()+1, 1);
		_shadow.push_back(s);
	}
}

void ghmm::set_corpus(gio& f) {
	if (f.dim() != _d)
		throw "ghmm::set_corpus dimension does not match the model";
	_build_shadow(f);
}

double ghmm::_emission_lp(vlattice& l, int i, int k) {
	int id = l.wd(i).id;
	if (id < 0 || id >= (int)_x.size())
		throw "ghmm::_emission_lp observation index out of range";
	return _obs->lp(k, _x[id]);
}

void ghmm::_resize_locked() {
	int before = _k;
	vhmm::_resize_locked();
	if (_obs && _k != before)
		_obs->resize(_k+1);
}

void ghmm::init(int seq) {
	if (seq < 0 || seq >= (int)_shadow.size())
		throw "ghmm::init sequence index out of range";
	sentence& s = _shadow[seq];
	lock_guard<mutex> l(_mutex);
	s.n[s.size()] = _mn;
	s.wd(s.size()).pos = 0;
	s.wd(s.size()).n = s.n[s.size()];
	for (int i = 0; i < s.size(); ++i) {
		word& w = s.wd(i);
		vector<double> table;
		vector<int> table_n, table_k;
		for (int k = 1; k <= _k; ++k) {
			double ln_pr_em = _obs->lp(k, _x[w.id]);
			context *c = _pos->h();
			double ln_pass = 0., ln_pr = _pos->lp(k, _pos->h());
			for (int j = 1; j <= _n && i-j >= -2; ++j) {
				table.push_back(ln_pr_em + _pos->lp_order(k, c, ln_pass, ln_pr));
				table_n.push_back(j);
				table_k.push_back(k);
				if (c) c = static_cast<vhdp_context*>(c)->find(s.wd(i-j).pos);
			}
		}
		int id = rd::ln_draw(table);
		w.pos = table_k[id];
		s.n[i] = max(_mn, min(min(i+2, _n), table_n[id]));
		w.n = s.n[i];
		context *p = _pos->make(s, i, s.n[i]);
		_obs->add(w.pos, _x[w.id]);
		_pos->add(w.pos, p);
		if (w.pos == _k) _resize();
	}
	context *p = _pos->make(s, s.size(), s.n[s.size()]);
	_pos->add(0, p);
}

void ghmm::inference_init(int seq) {
	if (seq < 0 || seq >= (int)_shadow.size())
		throw "ghmm::inference_init sequence index out of range";
	sentence& s = _shadow[seq];
	for (int i = 0; i < s.size(); ++i) {
		word& w = s.wd(i);
		s.n[i] = min(_n, max(_mn, min(i+2, _n)));
		w.n = s.n[i];
		context *p = _pos->find_exist(s, i, s.n[i]);
		vector<double> table;
		for (int k = 1; k <= _k; ++k)
			table.push_back(_pos->lp(k, p) + _obs->lp(k, _x[w.id]));
		w.pos = 1 + rd::ln_draw(table);
	}
	s.n[s.size()] = _mn;
	s.wd(s.size()).pos = 0;
	s.wd(s.size()).n = s.n[s.size()];
}

void ghmm::add(int seq) {
	sentence& s = _shadow[seq];
	lock_guard<mutex> l(_mutex);
	for (int i = 0; i < s.size(); ++i) {
		word& w = s.wd(i);
		_obs->add(w.pos, _x[w.id]);
		context *p = _pos->make(s, i, s.n[i]);
		_pos->add(w.pos, p);
	}
	context *p = _pos->make(s, s.size(), s.n[s.size()]);
	_pos->add(0, p);
}

void ghmm::remove(int seq) {
	sentence& s = _shadow[seq];
	lock_guard<mutex> l(_mutex);
	for (int i = 0; i < s.size(); ++i) {
		word& w = s.wd(i);
		_obs->remove(w.pos, _x[w.id]);
		context *p = _pos->find_exist(s, i, s.n[i]);
		_pos->remove(w.pos, p);
	}
	context *p = _pos->find_exist(s, s.size(), s.n[s.size()]);
	_pos->remove(0, p);
}

void ghmm::sample(int seq) {
	_shadow[seq] = vhmm::sample(_shadow[seq]);
}

void ghmm::parse(int seq) {
	_shadow[seq] = vhmm::parse(_shadow[seq]);
}

void ghmm::store(int seq, gsentence& g) const {
	const sentence& s = _shadow[seq];
	for (int i = 0; i < (int)s.w.size() && i < g.size(); ++i) {
		g.v[i].pos = s.w[i].pos;
		g.v[i].n = s.w[i].n;
	}
}

void ghmm::estimate(int iter) {
#ifdef _OPENMP
	int keep = omp_get_max_threads();
	omp_set_num_threads(1);
#endif
	if (_obs)
		_obs->estimate();
	_pos->estimate(iter);
#ifdef _OPENMP
	omp_set_num_threads(keep);
#endif
}

void ghmm::save(const char *file) {
	FILE *fp = fopen(file, "wb");
	if (!fp) throw "failed to open save file in ghmm::save";
	if (fwrite(&_n, sizeof(int), 1, fp) != 1 || fwrite(&_mn, sizeof(int), 1, fp) != 1 ||
	    fwrite(&_d, sizeof(int), 1, fp) != 1 || fwrite(&_k, sizeof(int), 1, fp) != 1 ||
	    fwrite(&_K, sizeof(int), 1, fp) != 1)
		throw "failed to write ghmm parameters";
	_pos->save(fp);
	_obs->save(fp);
	fclose(fp);
}

void ghmm::load(const char *file) {
	FILE *fp = fopen(file, "rb");
	if (!fp) throw "failed to open model file in ghmm::load";
	if (fread(&_n, sizeof(int), 1, fp) != 1 || fread(&_mn, sizeof(int), 1, fp) != 1 ||
	    fread(&_d, sizeof(int), 1, fp) != 1 || fread(&_k, sizeof(int), 1, fp) != 1 ||
	    fread(&_K, sizeof(int), 1, fp) != 1)
		throw "failed to read ghmm parameters";
	_pos = shared_ptr<vhdp>(new vhdp(_n));
	_pos->load(fp);
	_obs = shared_ptr<niw>(new niw(_d));
	_obs->load(fp);
	fclose(fp);
}

void ghmm::dump(FILE *fp) const {
	if (_obs) _obs->dump(fp);
}

void ghmm::dump_posterior(FILE *fp) const {
	if (!_obs)
		return;
	vector<double> s, ss, sigma;
	for (int k = 0; k <= _k; ++k) {
		int n = _obs->count(k);
		if (n <= 0)
			continue;
		_obs->sufficient_statistics(k, s, ss);
		_obs->posterior_sigma_mean(k, sigma);
		fprintf(fp, "state %d n %d s", k, n);
		for (int i = 0; i < _d; ++i)
			fprintf(fp, " %.17g", s[i]);
		fprintf(fp, "\n  ss");
		for (int i = 0; i < _d*_d; ++i)
			fprintf(fp, " %.17g", ss[i]);
		fprintf(fp, "\n  sigma");
		for (int i = 0; i < _d*_d; ++i)
			fprintf(fp, " %.17g", sigma[i]);
		fprintf(fp, "\n");
	}
	context *r = _pos->h();
	for (int k = 0; k <= _k; ++k)
		fprintf(fp, "trans %d %.17g\n", k, _pos->lp(k, r));
	fprintf(fp, "params n %d mn %d d %d k %d K %d\n", _n, _mn, _d, _k, _K);
}
