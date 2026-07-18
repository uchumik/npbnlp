#include"ipcfg.h"
#include"cyk.h"
#include"rd.h"
#include"convinience.h"
#include"generator.h"
#include<queue>
#include<cassert>
#include<cstdint>
#include<limits>
#include<cstdlib>

#ifdef _OPENMP
#include<omp.h>
#endif

//#define C 50000
#define C 1
#define K 1000

using namespace std;
using namespace npbnlp;

ipcfg::ipcfg():_m(20), _k(20),_K(K), _v(C), _a(1), _b(1), _span(false), _bottom_up(false), _shared_letter(true), _span_a(1), _span_b(1), _span_p(.5), _span_stop(0), _span_continue(0), _slice_terminal_cells(0), _slice_terminal_labels(0), _slice_internal_cells(0), _slice_internal_labels(0), _nonterm(new hpyp(3)),_word(new vector<shared_ptr<hpyp> >),_letter(new vector<shared_ptr<vpyp> >) {
	shared_ptr<vpyp> letter(new vpyp(_m));
	for (auto i = 0; i < _k+1; ++i) {
		_word->push_back(shared_ptr<hpyp>(new hpyp(1)));
		_letter->push_back(letter);
		(*_word)[i]->set_base(letter.get());
	}
}

ipcfg::ipcfg(int m):_m(m), _k(20), _K(K), _v(C), _a(1), _b(1), _span(false), _bottom_up(false), _shared_letter(true), _span_a(1), _span_b(1), _span_p(.5), _span_stop(0), _span_continue(0), _slice_terminal_cells(0), _slice_terminal_labels(0), _slice_internal_cells(0), _slice_internal_labels(0), _nonterm(new hpyp(3)), _word(new vector<shared_ptr<hpyp> >), _letter(new vector<shared_ptr<vpyp> >) {
	shared_ptr<vpyp> letter(new vpyp(_m));
	for (auto i = 0; i < _k+1; ++i) {
		_word->push_back(shared_ptr<hpyp>(new hpyp(1)));
		_letter->push_back(letter);
		(*_word)[i]->set_base(letter.get());
	}
}

ipcfg::~ipcfg() {
}

void ipcfg::save(const char *f) {
	FILE *fp = NULL;
	if ((fp = fopen(f, "wb")) == NULL)
		throw "failed to open save file in ipcfg::save";
	try {
		_save(fp);
	} catch (const char *ex) {
		fclose(fp);
		throw ex;
	}
	fclose(fp);
}

void ipcfg::_save(FILE *fp) const {
	try {
		if (fwrite(&_m, sizeof(int), 1, fp) != 1)
			throw "failed to write _m in ipcfg::save";
		if (fwrite(&_k, sizeof(int), 1, fp) != 1)
			throw "failed to write _k in ipcfg::save";
		if (fwrite(&_K, sizeof(int), 1, fp) != 1)
			throw "failed to write _K in ipcfg::save";
		if (fwrite(&_v, sizeof(int), 1, fp) != 1)
			throw "failed to write _v in ipcfg::save";
		_nonterm->save(fp);
		for (auto i = 0; i < _k+1; ++i) {
			(*_word)[i]->save(fp);
			(*_letter)[i]->save(fp);
		}
		// Optional tail block: old models end after the language models and
		// therefore load with the geometric span prior disabled.
		uint32_t magic = 0x50414750; // "PAGP"
		uint32_t version = 4;
		int enabled = _span ? 1 : 0;
		int free_parent = 0; // version-3 compatibility slot; ordered is fixed.
		int shared_letter = _shared_letter ? 1 : 0;
		if (fwrite(&magic, sizeof(uint32_t), 1, fp) != 1 ||
		    fwrite(&version, sizeof(uint32_t), 1, fp) != 1 ||
		    fwrite(&enabled, sizeof(int), 1, fp) != 1 ||
		    fwrite(&_span_a, sizeof(double), 1, fp) != 1 ||
		    fwrite(&_span_b, sizeof(double), 1, fp) != 1 ||
		    fwrite(&_span_p, sizeof(double), 1, fp) != 1 ||
		    fwrite(&_span_stop, sizeof(long long), 1, fp) != 1 ||
		    fwrite(&_span_continue, sizeof(long long), 1, fp) != 1 ||
		    fwrite(&free_parent, sizeof(int), 1, fp) != 1 ||
		    fwrite(&shared_letter, sizeof(int), 1, fp) != 1 ||
		    fwrite(&_bottom_up, sizeof(bool), 1, fp) != 1)
			throw "failed to write span prior tail in ipcfg::save";
	} catch (const char *ex) {
		throw ex;
	}
}

void ipcfg::load(const char *f) {
	FILE *fp = NULL;
	if ((fp = fopen(f, "rb")) == NULL)
		throw "failed to open save file in ipcfg::load";
	try {
		_load(fp);
	} catch (const char *ex) {
		fclose(fp);
		throw ex;
	}
	fclose(fp);
}

void ipcfg::_load(FILE *fp) {
	try {
		if (fread(&_m, sizeof(int), 1, fp) != 1)
			throw "failed to read _m in ipcfg::load";
		if (fread(&_k, sizeof(int), 1, fp) != 1)
			throw "failed to read _k in ipcfg::load";
		if (fread(&_K, sizeof(int), 1, fp) != 1)
			throw "failed to read _K in ipcfg::load";
		if (fread(&_v, sizeof(int), 1, fp) != 1)
			throw "failed to read _v in ipcfg::load";
		_nonterm->load(fp);
		// Serialized records always contain one VPYP payload per class slot.
		// Detach the default shared pointers before consuming those payloads;
		// version-3 models are collapsed back to one shared object below.
		if (_shared_letter)
			_unshare_letters();
		while ((int)_word->size() < _k+1) {
			_word->push_back(shared_ptr<hpyp>(new hpyp(1)));
			_letter->push_back(shared_ptr<vpyp>(new vpyp(_m)));
			(*_word)[_word->size()-1]->set_base((*_letter)[_word->size()-1].get());
		}
		for (auto i = 0; i < _k+1; ++i) {
			(*_word)[i]->load(fp);
			(*_letter)[i]->load(fp);
		}
		uint32_t magic = 0;
		size_t nr = fread(&magic, sizeof(uint32_t), 1, fp);
		if (nr == 0 && feof(fp)) {
			clearerr(fp);
			_span = false;
			_span_a = _span_b = 1.;
			_span_p = .5;
			_span_stop = _span_continue = 0;
			_shared_letter = false;
		} else {
			uint32_t version = 0;
			int enabled = 0;
			if (nr != 1 || magic != 0x50414750 ||
			    fread(&version, sizeof(uint32_t), 1, fp) != 1 || (version != 1 && version != 2 && version != 3 && version != 4) ||
			    fread(&enabled, sizeof(int), 1, fp) != 1 ||
			    fread(&_span_a, sizeof(double), 1, fp) != 1 ||
			    fread(&_span_b, sizeof(double), 1, fp) != 1 ||
			    fread(&_span_p, sizeof(double), 1, fp) != 1 ||
			    fread(&_span_stop, sizeof(long long), 1, fp) != 1 ||
			    fread(&_span_continue, sizeof(long long), 1, fp) != 1)
				throw "failed to read span prior tail in ipcfg::load";
			_span = enabled != 0;
			_bottom_up = false;
			_shared_letter = false;
			if (version == 2) {
				int free_parent = 0;
				if (fread(&free_parent, sizeof(int), 1, fp) != 1)
					throw "failed to read parent-label mode in ipcfg::load";
			} else if (version == 3 || version == 4) {
				int free_parent = 0;
				int shared_letter = 0;
				if (fread(&free_parent, sizeof(int), 1, fp) != 1 ||
				    fread(&shared_letter, sizeof(int), 1, fp) != 1)
					throw "failed to read iPCFG shared-letter mode in ipcfg::load";
				_shared_letter = shared_letter != 0;
				if (_shared_letter)
					_share_letters();
				if (version == 4 && fread(&_bottom_up, sizeof(bool), 1, fp) != 1)
					throw "failed to read iPCFG rule-factor mode in ipcfg::load";
			}
		}
	} catch (const char *ex) {
		throw ex;
	}
}

unique_ptr<ipcfg> ipcfg::snapshot() const {
	char *buf = nullptr;
	size_t size = 0;
	FILE *out = open_memstream(&buf, &size);
	if (!out)
		throw "failed to create in-memory iPCFG snapshot";
	try {
		_save(out);
		if (fclose(out) != 0)
			throw "failed to finalize in-memory iPCFG snapshot";
		out = nullptr;
		FILE *in = fmemopen(buf, size, "rb");
		if (!in)
			throw "failed to reopen in-memory iPCFG snapshot";
		unique_ptr<ipcfg> copy(new ipcfg(_m));
		copy->_load(in);
		fclose(in);
		copy->_a = _a;
		copy->_b = _b;
		copy->_tfreq = _tfreq;
		free(buf);
		return copy;
	} catch (const char *ex) {
		if (out) fclose(out);
		free(buf);
		throw ex;
	}
}

tree ipcfg::sample(io& f, int i) {
	return sample(f, i, nullptr);
}

tree ipcfg::init(io& f, int i) {
	lock_guard<mutex> m(_mutex);
	sentence s(*f.raw, f.head[i], f.head[i+1]);
	tree t(s);
	if (s.size() == 1) {
		t[0].k = 0;
		t[0].i = 0;
		t[0].j = 0;
		return t;
	}
	int root = s.size()-1;
	t[root].i = 0;
	t[root].j = s.size()-1;
	_init_node(t, root, 0);
	// Match add(): a category used as the current upper boundary opens the
	// next one only after this whole tree has been generated.
	if (_tfreq[_k] > 0)
		_resize();
	return t;
}

void ipcfg::_init_node(tree& t, int idx, int label) {
	node& z = t[idx];
	z.k = label;
	if (z.i == z.j) { // A -> observed word
		_check_label(z, "ipcfg::init encountered an inactive preterminal label");
		_tfreq[label]++;
		word& w = t.wd(z.i);
		(*_word)[label]->add(w, (*_word)[label]->h());
		_nonterm->add(label, _nonterm->h());
		return;
	}

	_check_label(z, "ipcfg::init encountered an inactive nonterminal label");
	shared_ptr<generator> g = generator::create();
	uniform_int_distribution<> split(z.i, z.j-1);
	z.b = split((*g)());
	vector<double> table;
	vector<int> left;
	vector<int> right;
	for (int l = 1; l <= _k; ++l) {
		double lp_l = _nonterm->lp(l, _nonterm->h());
		context *h = _nonterm->h();
		context *q = h->find(l);
		if (q)
			h = q;
		for (int r = 1; r <= _k; ++r) {
			if (!_parent_allowed(label, l, r)) continue;
			table.push_back(_rule_lp(label, l, r));
			left.push_back(l);
			right.push_back(r);
		}
	}
	if (table.empty())
		throw "ipcfg::init found no legal binary production";
	int selected = rd::ln_draw(table);
	int l = left[selected];
	int r = right[selected];

	// Keep this local update in exactly the same order and contexts as _add.
	// Thus remove() is the inverse bookkeeping operation of initialization.
	_tfreq[label]++;
	if (_span && !(z.i == 0 && z.j == t.s.size()-1)) {
		++_span_stop;
		_span_continue += z.j-z.i-1;
	}
	_rule_add(label, l, r);

	int size = t.s.size();
	int li = size*z.i+z.b-z.i*(1.+z.i)/2;
	int ri = size*(z.b+1)+z.j-(z.b+1.)*(z.b+2)/2;
	node& ln = t[li];
	ln.i = z.i;
	ln.j = z.b;
	node& rn = t[ri];
	rn.i = z.b+1;
	rn.j = z.j;
	_init_node(t, li, l);
	_init_node(t, ri, r);
}

double ipcfg::_init_logprob_and_add(tree& t, int idx) {
	node& z = t[idx];
	_check_label(z, "ipcfg::mh encountered an inactive tree label");
	if (z.i == z.j) {
		if (z.k <= 0)
			return 0.;
		word& w = t.wd(z.i);
		double lp = (*_word)[z.k]->lp(w, (*_word)[z.k]->h())+
			_nonterm->lp(z.k, _nonterm->h());
		_tfreq[z.k]++;
		(*_word)[z.k]->add(w, (*_word)[z.k]->h());
		_nonterm->add(z.k, _nonterm->h());
		return lp;
	}
	int N = t.s.size();
	int li = N*z.i+z.b-z.i*(1.+z.i)/2;
	int ri = N*(z.b+1)+z.j-(z.b+1.)*(z.b+2)/2;
	node& left = t[li];
	node& right = t[ri];
	_check_label(left, "ipcfg::mh encountered an inactive left child label");
	_check_label(right, "ipcfg::mh encountered an inactive right child label");
	if (!_parent_allowed(z.k, left.k, right.k))
		throw "ipcfg::mh encountered an illegal parent production";

	vector<double> table;
	double selected = -numeric_limits<double>::infinity();
	for (int l = 1; l <= _k; ++l) {
		double lp_l = _nonterm->lp(l, _nonterm->h());
		context *h = _nonterm->h();
		context *q = h->find(l);
		if (q) h = q;
		for (int r = 1; r <= _k; ++r) {
			if (!_parent_allowed(z.k, l, r)) continue;
			double lp = _rule_lp(z.k, l, r);
			table.push_back(lp);
			if (l == left.k && r == right.k)
				selected = lp;
		}
	}
	if (table.empty() || !isfinite(selected))
		throw "ipcfg::mh could not score the selected production";
	double norm = -numeric_limits<double>::infinity();
	for (double lp : table)
		norm = math::lse(norm, lp, !isfinite(norm));
	double lp = selected-norm-log((double)(z.j-z.i));
	if (_span && !(z.i == 0 && z.j == N-1)) {
		const double floor = numeric_limits<double>::min();
		lp += log(max(floor, _span_p))+
			(z.j-z.i-1)*log(max(floor, 1.-_span_p));
	}
	_tfreq[z.k]++;
	if (_span && !(z.i == 0 && z.j == N-1)) {
		++_span_stop;
		_span_continue += z.j-z.i-1;
	}
	_rule_add(z.k, left.k, right.k);
	return lp+_init_logprob_and_add(t, li)+_init_logprob_and_add(t, ri);
}

double ipcfg::mh_logprob_and_add(tree& t) {
	lock_guard<mutex> m(_mutex);
	double lp = _init_logprob_and_add(t, t.s.size()-1);
	if (_tfreq[_k] > 0)
		_resize();
	return lp;
}

tree ipcfg::sample(io& f, int i, tree *cur) {
	return _sample(f, i, cur, false, nullptr, nullptr);
}

tree ipcfg::mh_propose(io& f, int i, tree *cur, double& log_q, double& log_q_cur) {
	return _sample(f, i, cur, false, &log_q, &log_q_cur);
}

tree ipcfg::_sample(io& f, int i, tree *cur, bool full_cyk, double *log_q,
					double *log_q_cur) {
	cyk c(f, i);
	if (c.s.size() == 1) {
		tree t(c.s);
		t[0].k = 0;
		t[0].i = 0;
		t[0].j = 0;
		if (log_q)
			*log_q = 0.;
		if (log_q_cur)
			*log_q_cur = 0.;
		return t;
	}
	vt dp;
	_slice(c, cur, full_cyk);
	_record_slice(c);
	// inside
	int size = c.s.size();
	for (auto j = 0; j < size; ++j) {
		_calc_preterm(c, j, dp[j][j]);
	}
	for (auto l = 1; l < size; ++l) {
		for (auto j = 0; j < size-l; ++j) {
			//double mu = c.mu[j][j+l];
			_calc_nonterm(c, j, j+l, dp);
		}
	}
	// tree sampling
	tree t(c.s);
	int k = 0; // root
	int id = t.s.size()-1; // root id
	node& root = t[id];
	root.k = k;
	root.i = 0;
	root.j = size-1;
	if (log_q_cur) {
		if (!cur || cur->s.size() != size)
			throw "ipcfg::mh missing current tree for slice proposal score";
		*log_q_cur = _traceback_logprob(c, 0, size-1, 0, dp, *cur)-
			dp[0][size-1][0].v;
	}
	double traceback_lp = _traceback(c, 0, t.s.size()-1, k, dp, t);
	if (log_q)
		*log_q = traceback_lp-dp[0][size-1][0].v;
	return t;
}

tree ipcfg::parse(io& f, int i) {
	cyk c(f, i);
	if (c.s.size() == 1) {
		tree t(c.s);
		t[0].k = 0;
		t[0].i = 0;
		t[0].j = 0;
		return t;
	}
	vt dp;
	_slice(c, nullptr);
	// inside
	int size = c.s.size();
	for (auto j = 0; j < size; ++j) {
		_calc_preterm(c, j, dp[j][j]);
	}
	for (auto l = 1; l < size; ++l) {
		for (auto j = 0; j < size-l; ++j) {
			//double mu = c.mu[j][j+l];
			_calc_nonterm(c, j, j+l, dp);
		}
	}
	// tree sampling
	tree t(c.s);
	int k = 0; // root
	int id = t.s.size()-1; // root id
	node& root = t[id];
	root.k = k;
	root.i = 0;
	root.j = size-1;
	_traceback(c, 0, t.s.size()-1, k, dp, t, true);
	return t;
}

void ipcfg::add(tree& t) {
	lock_guard<mutex> m(_mutex);
	_add(t, t.s.size()-1);
	if (_tfreq[_k] > 0)
		_resize();
}

void ipcfg::_check_label(const node& z, const char *where) const {
	if (z.k < 0 || z.k > _k || z.k >= (int)_word->size() ||
	    z.k >= (int)_letter->size()) {
		fprintf(stderr, "%s: label=%d active_k=%d word_models=%zu letter_models=%zu\n",
			where, z.k, _k, _word->size(), _letter->size());
		throw where;
	}
}

double ipcfg::_rule_lp(int parent, int left, int right) const {
	if (_bottom_up) {
		double lp = _nonterm->lp(left, _nonterm->h());
		context *h = _nonterm->h();
		if (context *q = h->find(left)) h = q;
		lp += _nonterm->lp(right, h);
		h = _nonterm->h();
		if (context *q = h->find(right)) h = q;
		if (context *q = h->find(left)) h = q;
		return lp+_nonterm->lp(parent, h);
	}
	context *h = _nonterm->h();
	if (context *q = h->find(parent)) h = q;
	double lp = _nonterm->lp(left, h);
	if (context *q = h->find(left)) h = q;
	return lp+_nonterm->lp(right, h);
}

void ipcfg::_rule_add(int parent, int left, int right) {
	if (_bottom_up) {
		context *h = _nonterm->h()->make(right)->make(left);
		_nonterm->add(parent, h);
		h = _nonterm->h();
		_nonterm->add(left, h);
		h = h->make(left);
		_nonterm->add(right, h);
		return;
	}
	context *h = _nonterm->h()->make(parent);
	_nonterm->add(left, h);
	h = h->make(left);
	_nonterm->add(right, h);
}

void ipcfg::_rule_remove(int parent, int left, int right) {
	if (_bottom_up) {
		context *h = _nonterm->h()->find(right);
		if (!h || !(h = h->find(left)))
			throw "ipcfg::remove missing bottom-up parent context";
		_nonterm->remove(parent, h);
		h = _nonterm->h();
		_nonterm->remove(left, h);
		if (!(h = h->find(left)))
			throw "ipcfg::remove missing bottom-up left context";
		_nonterm->remove(right, h);
		return;
	}
	context *h = _nonterm->h()->find(parent);
	if (!h) throw "ipcfg::remove missing parent rule context";
	_nonterm->remove(left, h);
	h = h->find(left);
	if (!h) throw "ipcfg::remove missing left rule context";
	_nonterm->remove(right, h);
}

void ipcfg::_add(tree& t, int i) {
	node& z = t[i];
	_check_label(z, "ipcfg::add encountered an inactive tree label");
	_tfreq[z.k]++;
	if (z.i != z.j) { // nonterminal
		if (_span && !(z.i == 0 && z.j == t.s.size()-1)) {
			++_span_stop;
			_span_continue += z.j-z.i-1;
		}
		node& left = t[t.s.size()*z.i+z.b-z.i*(1.+z.i)/2];
		node& right = t[t.s.size()*(z.b+1)+z.j-(1.+z.b)*(z.b+2)/2];
		_rule_add(z.k, left.k, right.k);
		_add(t, t.s.size()*z.i+z.b-z.i*(1.+z.i)/2);
		_add(t, t.s.size()*(z.b+1)+z.j-(1.+z.b)*(z.b+2)/2);
	} else if (z.k > 0) { // preterminal
		word& w = t.wd(z.i);
		context *h = (*_word)[z.k]->h();
		(*_word)[z.k]->add(w, h);
		_nonterm->add(z.k, _nonterm->h());
	}
}

void ipcfg::remove(tree& t) {
	lock_guard<mutex> m(_mutex);
	_remove(t, t.s.size()-1);
	// This tree is temporarily held out for Gibbs sampling.  Its labels may
	// be the final users of a category, so shrinking here would invalidate the
	// tree before it can be resampled and re-added.
}

void ipcfg::compact() {
	lock_guard<mutex> m(_mutex);
	// Category 0 is the structural root.  This is called only at an epoch
	// boundary, after every held-out tree has been re-added.
	// _k itself is a valid latent label (loops elsewhere use 1.._k), so it
	// must be checked before popping its model.  Starting at _k-1 discards
	// label _k even when a fully-added corpus still contains it.
	for (int k = _k; k > 1 && _tfreq[k] == 0; --k) {
		_shrink();
	}
}

void ipcfg::_remove(tree& t, int i) {
	node& z = t[i];
	_check_label(z, "ipcfg::remove encountered an inactive tree label");
	if (_tfreq[z.k] <= 0)
		throw "ipcfg::remove frequency underflow";
	_tfreq[z.k]--;
	if (z.i != z.j) { // nonterminal
		if (_span && !(z.i == 0 && z.j == t.s.size()-1)) {
			--_span_stop;
			_span_continue -= z.j-z.i-1;
			if (_span_stop < 0 || _span_continue < 0)
				throw "span prior count underflow in ipcfg::remove";
		}
		node& left = t[t.s.size()*z.i+z.b-z.i*(1.+z.i)/2];
		node& right = t[t.s.size()*(z.b+1)+z.j-(1.+z.b)*(z.b+2)/2];
		_rule_remove(z.k, left.k, right.k);
		_remove(t, t.s.size()*z.i+z.b-z.i*(1.+z.i)/2);
		_remove(t, t.s.size()*(z.b+1)+z.j-(1.+z.b)*(z.b+2)/2);
	} else if (z.k > 0) { // preterminal
		word& w = t.wd(z.i);
		context *h = (*_word)[z.k]->h();
		(*_word)[z.k]->remove(w, h);
		_nonterm->remove(z.k, _nonterm->h());
	}
}

bool ipcfg::valid() const {
	if (!_nonterm || !_word || !_letter ||
	    (int)_word->size() < _k+1 || (int)_letter->size() < _k+1)
		return false;
	if (!_nonterm->valid()) {
		fprintf(stderr, "ipcfg validation: nonterminal HPYP invalid\n");
		return false;
	}
	for (int i = 0; i <= _k; ++i) {
		if (!(*_word)[i] || !(*_letter)[i])
			return false;
		if (!(*_word)[i]->valid()) {
			fprintf(stderr, "ipcfg validation: word HPYP %d invalid\n", i);
			return false;
		}
		if (!(*_letter)[i]->valid()) {
			fprintf(stderr, "ipcfg validation: letter HPYP %d invalid\n", i);
			return false;
		}
	}
	for (auto it = _tfreq.cbegin(); it != _tfreq.cend(); ++it) {
		if (it->second < 0)
			return false;
	}
	return true;
}

bool ipcfg::empty() const {
	if (!valid() || !_nonterm->empty() || _span_stop != 0 || _span_continue != 0)
		return false;
	for (int i = 0; i <= _k; ++i) {
		if (!(*_word)[i]->empty() || !(*_letter)[i]->empty())
			return false;
	}
	for (auto it = _tfreq.cbegin(); it != _tfreq.cend(); ++it) {
		if (it->second != 0)
			return false;
	}
	return true;
}

void ipcfg::estimate(int iter) {
	for (int i = 1; i < _k+1; ++i) {
		(*_word)[i]->gibbs(iter);
		(*_word)[i]->estimate(iter);
	}
	if (_shared_letter) {
		(*_letter)[0]->estimate(iter);
	} else {
		for (int i = 1; i < _k+1; ++i)
			(*_letter)[i]->estimate(iter);
	}
	_nonterm->estimate(iter);
	if (_span) {
		beta_distribution be;
		_span_p = be(_span_a+_span_stop, _span_b+_span_continue);
		cerr << "[span] p=" << _span_p << " stop=" << _span_stop
		     << " continue=" << _span_continue << endl;
	}
}

void ipcfg::poisson_correction(int n) {
	for (int i = 1; i < _k+1; ++i) {
		(*_word)[i]->poisson_correction(n);
	}
}

void ipcfg::set(int k) {
	_K = k;
	_k = min(_k, _K);
}

void ipcfg::slice(double a, double b) {
	if (a <= 0 || b <= 0) {
		return;
	}
	_a = a;
	_b = b;
}

void ipcfg::bottom_up(bool enabled) {
	_bottom_up = enabled;
}

bool ipcfg::_parent_allowed(int parent, int left, int right) const {
	return parent <= max(left, right);
}

void ipcfg::_record_slice(cyk& c) {
	int size = c.s.size();
	for (int i = 0; i < size; ++i) {
		for (int j = i; j < size; ++j) {
			if (i == 0 && j == size-1)
				continue;
			long long labels = c.k[i][j].size();
			if (i == j) {
				++_slice_terminal_cells;
				_slice_terminal_labels += labels;
			} else {
				++_slice_internal_cells;
				_slice_internal_labels += labels;
			}
		}
	}
}

void ipcfg::slice_diagnostics(long long& terminal_cells, long long& terminal_labels,
						  long long& internal_cells, long long& internal_labels) {
	terminal_cells = _slice_terminal_cells.exchange(0);
	terminal_labels = _slice_terminal_labels.exchange(0);
	internal_cells = _slice_internal_cells.exchange(0);
	internal_labels = _slice_internal_labels.exchange(0);
}

void ipcfg::span(double a, double b) {
	if (a <= 0 || b <= 0)
		return;
	_span = true;
	_span_a = a;
	_span_b = b;
	beta_distribution be;
	_span_p = be(_span_a, _span_b);
}

double ipcfg::_span_lp(cyk& c, int i, int j) {
	if (!_span || i == j || (i == 0 && j == c.s.size()-1))
		return 0.;
	const double floor = numeric_limits<double>::min();
	return log(max(floor, _span_p))+(j-i-1)*log(max(floor, 1.-_span_p));
}

double ipcfg::_traceback(cyk& c, int i, int j, int z, vt& a, tree& tr, bool best) {
	double mu = c.mu[i][j];
	if (i == j) { // pre-terminal
		word& w = c.wd(i);
		// A pre-terminal emits the observed terminal word.  This factor must
		// be part of the returned tree probability, not only of the inside DP.
		return (*_word)[z]->lp(w, (*_word)[z]->h())+
			_nonterm->lp(z, _nonterm->h());
	} else { // non-terminal
		vector<double> table;
		vector<int> left;
		vector<int> right;
		vector<int> brp; // break point
		vector<double> rule;
		for (auto k = i; k < j; ++k) {
			for (auto l = c.begin(i,k); l != c.end(i,k); ++l) {
				double lp_l = _nonterm->lp(*l, _nonterm->h());
				context *h = _nonterm->h();
				context *t = h->find(*l);
				if (t)
					h = t;
				for (auto r = c.begin(k+1,j); r != c.end(k+1,j); ++r) {
					if (!_parent_allowed(z, *l, *r))
						continue;
					double lp_r = _nonterm->lp(*r, h);
					context *s = _nonterm->h();
					context *u = s->find(*r);
					if (u) {
						s = u;
						u = s->find(*l);
						if (u)
							s = u;
					}
					double lp = _rule_lp(z, *l, *r)+_span_lp(c,i,j);
					if (lp < mu)
						continue;
					table.push_back(lp+a[i][k][*l].v+a[k+1][j][*r].v);
					rule.push_back(lp);
					left.push_back(*l);
					right.push_back(*r);
					brp.push_back(k);
				}
			}
		}
		int id = 0;
		if (best)
			id = rd::best(table);
		else
			id = rd::ln_draw(table);
		int b = brp[id];
		node& n = tr[tr.s.size()*i+j-i*(1.+i)/2];
		n.b = b;
		node& ln = tr[tr.s.size()*i+b-i*(1.+i)/2];
		node& rn = tr[tr.s.size()*(b+1)+j-(b+1.)*(2.+b)/2];
		ln.k = left[id];
		ln.i = i;
		ln.j = b;
		rn.k = right[id];
		rn.i = b+1;
		rn.j = j;
		return rule[id]+
			_traceback(c, i, b, ln.k, a, tr, best)+
			_traceback(c, b+1, j, rn.k, a, tr, best);
	}
}

double ipcfg::_traceback_logprob(cyk& c, int i, int j, int z, vt& a, tree& tr) {
	if (i == j) {
		word& w = c.wd(i);
		return (*_word)[z]->lp(w, (*_word)[z]->h())+
			_nonterm->lp(z, _nonterm->h());
	}
	int N = tr.s.size();
	node& n = tr[N*i+j-i*(1.+i)/2];
	if (n.k != z || n.b < i || n.b >= j)
		throw "ipcfg::mh current tree is incompatible with its slice lattice";
	vector<double> table, rule;
	vector<int> left, right, brp;
	double mu = c.mu[i][j];
	for (int k = i; k < j; ++k) {
		for (auto l = c.begin(i,k); l != c.end(i,k); ++l) {
			double lp_l = _nonterm->lp(*l, _nonterm->h());
			context *h = _nonterm->h();
			context *t = h->find(*l);
			if (t) h = t;
			for (auto r = c.begin(k+1,j); r != c.end(k+1,j); ++r) {
				if (!_parent_allowed(z, *l, *r)) continue;
				double lp_r = _nonterm->lp(*r, h);
				context *s = _nonterm->h();
				context *u = s->find(*r);
				if (u) {
					s = u;
					u = s->find(*l);
					if (u) s = u;
				}
				double lp = _rule_lp(z, *l, *r)+_span_lp(c,i,j);
				if (lp < mu) continue;
				table.push_back(lp+a[i][k][*l].v+a[k+1][j][*r].v);
				rule.push_back(lp);
				left.push_back(*l); right.push_back(*r); brp.push_back(k);
			}
		}
	}
	int selected = -1;
	for (int q = 0; q < (int)table.size(); ++q) {
		if (brp[q] == n.b && left[q] == tr[N*i+n.b-i*(1.+i)/2].k &&
		    right[q] == tr[N*(n.b+1)+j-(n.b+1.)*(n.b+2)/2].k) {
			selected = q;
			break;
		}
	}
	if (selected < 0)
		throw "ipcfg::mh current tree production was pruned from its slice lattice";
	int li = N*i+n.b-i*(1.+i)/2;
	int ri = N*(n.b+1)+j-(n.b+1.)*(n.b+2)/2;
	return rule[selected]+
		_traceback_logprob(c, i, n.b, tr[li].k, a, tr)+
		_traceback_logprob(c, n.b+1, j, tr[ri].k, a, tr);
}

void ipcfg::_calc_preterm(cyk& c, int j, vt& a) {
	word& w = c.wd(j);
	double mu = c.mu[j][j];
	for (auto k = c.begin(j,j); k != c.end(j,j); ++k) {
		double lp = (*_word)[*k]->lp(w, (*_word)[*k]->h())+_nonterm->lp(*k, _nonterm->h());
		if (lp >= mu) {
			a[*k].v = math::lse(a[*k].v, lp, true);
			a[*k].set(true);
		}
	}
}

void ipcfg::_calc_nonterm(cyk& c, int i, int j, vt& a) {
	double mu = c.mu[i][j];
	for (auto k = i; k < j; ++k) {
		for (auto l = c.begin(i,k); l != c.end(i,k); ++l) {
			double lp_l = _nonterm->lp(*l, _nonterm->h());
			context *h = _nonterm->h();
			context *t = h->find(*l);
			if (t)
				h = t;
			for (auto r = c.begin(k+1,j); r != c.end(k+1,j); ++r) {
				double lp_r = _nonterm->lp(*r, h);
				context *s = _nonterm->h();
				context *u = s->find(*r);
				if (u) {
					s = u;
					u = s->find(*l);
					if (u)
						s = u;
				}
				for (auto z = c.begin(i,j); z != c.end(i,j); ++z) {
					if (!_parent_allowed(*z, *l, *r))
						continue;
					double lp = _rule_lp(*z, *l, *r)+_span_lp(c,i,j);
					if (lp < mu)
						continue;
					a[i][j][*z].v = math::lse(a[i][j][*z].v, lp+a[i][k][*l].v+a[k+1][j][*r].v, !a[i][j][*z].is_init());
					if (!a[i][j][*z].is_init())
						a[i][j][*z].set(true);
				}
			}
		}
	}
}

// Recursively walk the current tree T from the root and record, for every
// span (i,j) that appears in T, a pointer to its node. Node index layout is
// idx(i,j) = N*i + j - i*(i+1)/2 (see _add/_remove). on-path nodes carry
// their assigned label in node.k (>=0); off-path cells keep the ctor default
// node.k == -1 and are never touched here.
void ipcfg::_collect_spans(tree& t, int idx, vector<vector<const node*> >& on) {
	node& z = t[idx];
	if (z.k < 0)
		return;
	on[z.i][z.j] = &z;
	if (z.i == z.j) // pre-terminal
		return;
	int N = t.s.size();
	int b = z.b;
	_collect_spans(t, N*z.i+b-z.i*(1.+z.i)/2, on);
	_collect_spans(t, N*(b+1)+z.j-(1.+b)*(b+2)/2, on);
}

void ipcfg::_slice(cyk& l, tree *cur, bool full_cyk) {
	int size = l.s.size();
	// Diagnostic full-CYK path.  Do not construct auxiliary slice variables:
	// every latent label remains available in every non-root cell and mu=-inf
	// disables the threshold check in _calc_* and _traceback.
	const bool noslice = full_cyk || (getenv("NPBNLP_NOSLICE") != NULL);
	if (noslice) {
		const double minus_inf = -numeric_limits<double>::infinity();
		for (int i = 0; i < size; ++i) {
			for (int j = i; j < size; ++j) {
				l.mu[i][j] = minus_inf;
				if (i == 0 && j == size-1) {
					l.k[i][j].insert(0);
					continue;
				}
				for (int k = 1; k <= _k; ++k)
					l.k[i][j].insert(k);
			}
		}
		return;
	}
	bool cond = (cur != nullptr && cur->s.size() == size && size > 1);
	if (cond) {
		// --- WO-005: span-independent mu conditioned on the current tree T ---
		// Deviation from the note's unscaled beta(a,b): off-path spans keep the
		// current per-span fresh draw (log(be)+table[id]) instead of beta(a,b),
		// because the unscaled threshold with the present a,b would over-prune
		// and wreck the search space. mu depends only on the fixed post-remove
		// scores, so slice-sampling validity (marginalizing mu recovers P) holds.
		vector<vector<const node*> > on(size, vector<const node*>(size, nullptr));
		_collect_spans(*cur, size-1, on); // root idx = N-1
		// pre-terminals
		for (int i = 0; i < size; ++i) {
			const node *z = on[i][i];
			if (z != nullptr && z->k >= 1 && z->k <= _k)
				_slice_preterm_cond(l, i, z->k); // on-path: mu = log(be)+score_cur
			else
				_slice_preterm(l, i);            // off-path: fresh draw
		}
		// non-terminals, span-independent (no pivot walk / shared nu)
		for (int m = 1; m < size-1; ++m) {
			for (int i = 0; i < size-m; ++i) {
				int j = i+m;
				const node *z = on[i][j];
				bool onpath = false;
				int lc = 0, rc = 0, kc = 0, mc = 0;
				if (z != nullptr && z->i == i && z->j == j && z->i != z->j) {
					int b = z->b;
					const node *ln = on[i][b];
					const node *rn = on[b+1][j];
					if (ln != nullptr && rn != nullptr) {
						lc = ln->k; rc = rn->k; kc = b; mc = z->k;
						if (mc >= 1 && mc <= _k && lc >= 1 && lc <= _k &&
						    rc >= 1 && rc <= _k && _parent_allowed(mc, lc, rc))
							onpath = true;
					}
				}
				if (onpath)
					_slice_nonterm_cond(l, i, j, lc, rc, kc, mc);
				else
					_draw(l, i, j); // off-path: per-span fresh draw
			}
		}
		// root
		const node *r = on[0][size-1];
		if (r != nullptr && r->i == 0 && r->j == size-1 && r->i != r->j) {
			int b = r->b;
			const node *ln = on[0][b];
			const node *rn = on[b+1][size-1];
			if (ln != nullptr && rn != nullptr &&
			    ln->k >= 1 && ln->k <= _k && rn->k >= 1 && rn->k <= _k)
				_slice_root_cond(l, ln->k, rn->k);
			else
				_slice_root(l);
		} else {
			_slice_root(l);
		}
		return;
	}
	// --- cur == nullptr: original path (parse), unchanged incl. RNG usage ---
	// terminal
	for (auto i = 0; i < l.s.size(); ++i) {
		_slice_preterm(l, i);
	}
	vector<double> table;
	for (auto i = 0; i < l.s.size()-1; ++i) {
		table.push_back(_marginalize(l,i,i+1));
	}
	int p = rd::ln_draw(table);
	//shared_ptr<generator> g = generator::create();
	//uniform_int_distribution<> u(0, l.s.size()-2);
	//int p = u((*g)());
	//non terminal
	for (auto m = 1; m < l.s.size()-1; ++m) {
		//int len = l.s.size()-m;
		//uniform_int_distribution<> v(0, len-1);
		//int p = v((*g)());
		// draw nu for slice non-terminal
		//double nu = _draw(l, p, p+m);
		double nu = _draw(l, p, p+m);
		// slice non-terminals
		for (auto i = 0; i < l.s.size()-m; ++i) {
			if (i == p)
				continue;
			_slice_nonterm(l, i, i+m, nu);
		}
		vector<double> cand;
		if (p > 0) {
			cand.push_back(_marginalize(l,p-1,p+m));
		}
		if (p+m < l.s.size()-1) {
			cand.push_back(_marginalize(l,p,p+m+1));
		}
		if (cand.size() > 1) {
			int id = rd::ln_draw(cand);
			if (id == 0) {
				p -= 1;
			}
		} else if (p > 0) {
			p -= 1;
		}
	}
	// root
	_slice_root(l);
}

double ipcfg::_marginalize(cyk& c, int i, int j) {
	double z = 0;
	for (auto k = i; k < j; ++k) {
		for (auto l = c.begin(i,k); l != c.end(i,k); ++l) {
			double lp_l = _nonterm->lp(*l, _nonterm->h());
			context *h = _nonterm->h();
			context *t = h->find(*l);
			if (t)
				h = t;
			for (auto r = c.begin(k+1,j); r != c.end(k+1,j); ++r) {
				double lp_r = _nonterm->lp(*r, h);
				context *s = _nonterm->h();
				context *u = s->find(*r);
				if (u) {
					s = u;
					u = s->find(*l);
					if (u)
						s = u;
				}
				for (auto m = max(*l,*r); m > 0; --m) {
					double lp = _rule_lp(m, *l, *r)+_span_lp(c,i,j);
					math::lse(z,lp,(z==0.));
				}
			}
		}
	}
	return z;
}

double ipcfg::_draw(cyk& c, int i, int j) {
	beta_distribution be;
	vector<double> table;
	vector<int> z;
	for (auto k = i; k < j; ++k) {
		for (auto l = c.begin(i,k); l != c.end(i,k); ++l) {
			double lp_l = _nonterm->lp(*l, _nonterm->h());
			context *h = _nonterm->h();
			context *t = h->find(*l);
			if (t)
				h = t;
			for (auto r = c.begin(k+1,j); r != c.end(k+1,j); ++r) {
				double lp_r = _nonterm->lp(*r, h);
				context *s = _nonterm->h();
				context *u = s->find(*r);
				if (u) {
					s = u;
					u = s->find(*l);
					if (u)
						s = u;
				}
				for (auto m = max(*l,*r); m > 0; --m) {
					double lp = _rule_lp(m, *l, *r)+_span_lp(c,i,j);
					table.push_back(lp);
					z.push_back(m);
				}
			}
		}
	}
	int id = rd::ln_draw(table);
	double mu = log(be(_a,_b))+table[id];
	c.mu[i][j] = mu;
	for (auto m = 0; m < (int)table.size(); ++m) {
		if (table[m] >= mu)
			c.k[i][j].insert(z[m]);
	}
	return mu;
}

void ipcfg::_slice_nonterm(cyk& c, int i, int j, double mu) {
	vector<double> table;
	vector<int> z;
	for (auto k = i; k < j; ++k) {
		for (auto l = c.begin(i,k); l != c.end(i,k); ++l) {
			double lp_l = _nonterm->lp(*l, _nonterm->h());
			context *h = _nonterm->h();
			context *t = h->find(*l);
			if (t)
				h = t;
			for (auto r = c.begin(k+1,j); r != c.end(k+1,j); ++r) {
				double lp_r = _nonterm->lp(*r, h);
				context *s = _nonterm->h();
				context *u = s->find(*r);
				if (u) {
					s = u;
					u = s->find(*l);
					if (u)
						s = u;
				}
				for (auto m = max(*l,*r); m > 0; --m) {
					double lp = _rule_lp(m, *l, *r)+_span_lp(c,i,j);
					table.push_back(lp);
					z.push_back(m);
				}
			}
		}
	}
	// P(B,C|A) := P(A->B C|A)
	// P(B,C|A) \propto P(B,C,A) = P(A|B,C)P(B,C)
	c.mu[i][j] = mu;
	for (auto m = 0; m < (int)table.size(); ++m) {
		if (table[m] >= mu) {
			c.k[i][j].insert(z[m]);
		}
	}
}

// WO-005 on-path non-terminal: condition mu on the current rule
// A(mc) -> B(lc) C(rc) split at kc. score_cur = log P(A|B,C) + log P(B) +
// log P(C|B) reproduces exactly what _draw/_slice_nonterm push to the table
// (_nonterm->lp(mc,s)+lp_l+lp_r). mu = log(be)+score_cur with log(be)<=0
// guarantees score_cur >= mu, so the current rule always survives the slice.
void ipcfg::_slice_nonterm_cond(cyk& c, int i, int j, int lc, int rc, int kc, int mc) {
	(void)kc; // split point is implied by lc/rc contexts; kept for clarity
	beta_distribution be;
	// score of the current on-path rule (same context construction as _draw)
	double lp_l = _nonterm->lp(lc, _nonterm->h());
	context *h = _nonterm->h();
	context *t = h->find(lc);
	if (t)
		h = t;
	double lp_r = _nonterm->lp(rc, h);
	context *s = _nonterm->h();
	context *u = s->find(rc);
	if (u) {
		s = u;
		u = s->find(lc);
		if (u)
			s = u;
	}
	double score_cur = _rule_lp(mc, lc, rc)+_span_lp(c,i,j);
	double mu = log(be(_a, _b))+score_cur;
	c.mu[i][j] = mu;
	// permitted set: enumerate every rule of this span exactly as _draw does
	for (auto k = i; k < j; ++k) {
		for (auto l = c.begin(i,k); l != c.end(i,k); ++l) {
			double lpl = _nonterm->lp(*l, _nonterm->h());
			context *hh = _nonterm->h();
			context *tt = hh->find(*l);
			if (tt)
				hh = tt;
			for (auto r = c.begin(k+1,j); r != c.end(k+1,j); ++r) {
				double lpr = _nonterm->lp(*r, hh);
				context *ss = _nonterm->h();
				context *uu = ss->find(*r);
				if (uu) {
					ss = uu;
					uu = ss->find(*l);
					if (uu)
						ss = uu;
				}
				for (auto mm = max(*l,*r); mm > 0; --mm) {
					double lp = _rule_lp(mm, *l, *r)+_span_lp(c,i,j);
					if (lp >= mu)
						c.k[i][j].insert(mm);
				}
			}
		}
	}
	assert(c.k[i][j].count(mc) > 0); // current parent label must survive
}
/*
   void ipcfg::_slice(cyk& l) {
// terminal
for (auto i = 0; i < l.s.size(); ++i) {
_slice_preterm(l, i);
}
// non terminal
for (auto m = 1; m < l.s.size()-1; ++m) {
for (auto i = 0; i < l.s.size()-m; ++i) {
_slice_nonterm(l, i, i+m);
}
}
// root
_slice_root(l);
}
*/

void ipcfg::_slice_preterm(cyk& l, int i) {
	beta_distribution be;
	//shared_ptr<generator> g = generator::create();
	word& w = l.wd(i);
	vector<double> table;
	for (auto k = 1; k < _k+1; ++k) {
		double lp = (*_word)[k]->lp(w, (*_word)[k]->h())+_nonterm->lp(k, _nonterm->h());
		table.push_back(lp);
	}
	int id = rd::ln_draw(table);
	double mu = log(be(_a, _b))+table[id];
	//double mu = table[id];
	l.mu[i][i] = mu;
	for (auto j = 0; j < (int)table.size(); ++j) {
		if (table[j] >= mu) {
			l.k[i][i].insert(j+1);
		}
	}
}

// WO-005 on-path pre-terminal: condition mu on the current label.
// table[label-1] is the score _slice_preterm/_calc_preterm assign to this
// label; mu = log(be)+score_cur keeps that label (and any richer ones) alive.
void ipcfg::_slice_preterm_cond(cyk& l, int i, int label) {
	beta_distribution be;
	word& w = l.wd(i);
	vector<double> table;
	for (auto k = 1; k < _k+1; ++k) {
		double lp = (*_word)[k]->lp(w, (*_word)[k]->h())+_nonterm->lp(k, _nonterm->h());
		table.push_back(lp);
	}
	double score_cur = table[label-1];
	double mu = log(be(_a, _b))+score_cur;
	l.mu[i][i] = mu;
	for (auto j = 0; j < (int)table.size(); ++j) {
		if (table[j] >= mu) {
			l.k[i][i].insert(j+1);
		}
	}
	assert(l.k[i][i].count(label) > 0); // current pre-terminal must survive
}

/*
   void ipcfg::_slice_nonterm(cyk& c, int i, int j) {
   beta_distribution be;
//shared_ptr<generator> g = generator::create();
vector<double> table;
vector<int> z;
for (auto k = i; k < j; ++k) {
for (auto l = c.begin(i,k); l != c.end(i,k); ++l) {
double lp_l = _nonterm->lp(*l, _nonterm->h());
context *h = _nonterm->h();
context *t = h->find(*l);
if (t)
h = t;
for (auto r = c.begin(k+1,j); r != c.end(k+1,j); ++r) {
double lp_r = _nonterm->lp(*r, h);
context *s = _nonterm->h();
context *u = s->find(*r);
if (u) {
s = u;
u = s->find(*l);
if (u)
s = u;
}
//for (auto m = 1; m < _k+1; ++m) {
for (auto m = max(*l,*r); m > 0; --m) {
double lp = _nonterm->lp(m, s)+lp_l+lp_r;
table.push_back(lp);
z.push_back(m);
}
}
}
}
// P(B,C|A) := P(A->B C|A)
// P(B,C|A) \propto P(B,C,A) = P(A|B,C)P(B,C)
// draw A ~ P(A,B,C) for a threshold at cell_{i,j}
int id = rd::ln_draw(table);
double mu = log(be(_a, _b))+table[id];
//double mu = table[id];
c.mu[i][j] = mu;
for (auto m = 0; m < (int)table.size(); ++m) {
if (table[m] >= mu) {
c.k[i][j].insert(z[m]);
}
}
}
*/

void ipcfg::_slice_root(cyk& c) {
	beta_distribution be;
	//shared_ptr<generator> g = generator::create();
	int size = c.s.size();
	vector<double> table;
	for (auto k = 0; k < size-1; ++k) {
		for (auto l = c.begin(0, k); l != c.end(0, k); ++l) {
			double lp_l = _nonterm->lp(*l, _nonterm->h());
			context *h = _nonterm->h();
			context *t = h->find(*l);
			if (t)
				h = t;
			for (auto r = c.begin(k+1, size-1); r != c.end(k+1, size-1); ++r) {
				double lp_r = _nonterm->lp(*r, h);
				context *s = _nonterm->h();
				context *u = s->find(*r);
				if (u) {
					s = u;
					u = s->find(*l);
					if (u)
						s = u;
				}
				double lp = _rule_lp(0, *l, *r);
				table.push_back(lp);
			}
		}
	}
	int id = rd::ln_draw(table);
	// table[id] is a log-domain score (== _nonterm->lp(...)+lp_l+lp_r); the
	// old form log(be(_a,_b)+table[id]) took log of (0,1)+negative => NaN,
	// which silenced root pruning entirely. Match the other slice helpers:
	// mu = log(be) + score, so the current root decomposition (score >= mu)
	// always survives while lower-scoring ones are pruned.
	double mu = log(be(_a, _b))+table[id];
	c.mu[0][size-1] = mu;
	c.k[0][size-1].insert(0);
	/*
	   for (auto m = 0; m < table.size(); ++m) {
	   if (table[m] >= mu)
	   c.k[0][size-1][0]+=1;
	   */
}

// WO-005 on-path root: root label is always 0; condition mu on the current
// root split into children lc/rc. score_cur = log P(0|lc,rc)+log P(lc)+
// log P(rc|lc) matches _slice_root's table entry; mu = log(be)+score_cur
// ensures the current root decomposition survives traceback pruning.
void ipcfg::_slice_root_cond(cyk& c, int lc, int rc) {
	beta_distribution be;
	int size = c.s.size();
	double lp_l = _nonterm->lp(lc, _nonterm->h());
	context *h = _nonterm->h();
	context *t = h->find(lc);
	if (t)
		h = t;
	double lp_r = _nonterm->lp(rc, h);
	context *s = _nonterm->h();
	context *u = s->find(rc);
	if (u) {
		s = u;
		u = s->find(lc);
		if (u)
			s = u;
	}
	double score_cur = _rule_lp(0, lc, rc);
	double mu = log(be(_a, _b))+score_cur;
	c.mu[0][size-1] = mu;
	c.k[0][size-1].insert(0);
}

void ipcfg::_resize() {
	if (_k+1 > _K)
		return;
	++_k;
	_word->resize(_k+1, shared_ptr<hpyp>(new hpyp(1)));
	if (_shared_letter) {
		_letter->push_back((*_letter)[0]);
		(*_word)[_k]->set_base((*_letter)[0].get());
	} else {
		_letter->resize(_k+1, shared_ptr<vpyp>(new vpyp(_m)));
		(*_word)[_k]->set_base((*_letter)[_k].get());
	}
}

void ipcfg::_shrink() {
	--_k;
	_word->pop_back();
	_letter->pop_back();
}

void ipcfg::_share_letters() {
	if (_letter->empty())
		throw "ipcfg::_share_letters called with no letter model";
	shared_ptr<vpyp> shared = (*_letter)[0];
	for (int i = 0; i <= _k; ++i) {
		(*_letter)[i] = shared;
		(*_word)[i]->set_base(shared.get());
	}
	_shared_letter = true;
}

void ipcfg::_unshare_letters() {
	for (int i = 0; i < (int)_letter->size(); ++i) {
		(*_letter)[i] = shared_ptr<vpyp>(new vpyp(_m));
		(*_word)[i]->set_base((*_letter)[i].get());
	}
	_shared_letter = false;
}
