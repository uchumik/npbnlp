#include "vhmm.h"
#include "convinience.h"
#include "rd.h"
#include "generator.h"
#include "math.h"
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <cstdlib>
#include <unordered_map>
#include <chrono>
#include "vhdp_context.h"
#ifdef _OPENMP
#include <omp.h>
#endif
namespace { long long N_retry=0, N_sent=0; double T_ord=0,T_slice=0,T_lat=0,T_fwd=0,T_bwd=0; using clk=std::chrono::steady_clock;
  struct Tm { double& acc; clk::time_point t0; Tm(double& a):acc(a),t0(clk::now()){} ~Tm(){ acc += std::chrono::duration<double>(clk::now()-t0).count(); } }; }

using namespace std;
using namespace npbnlp;

// Keep each dictionary entry while at least one sentence still uses its id; the
// emission models use these ids as their word keys.
static unordered_map<int,int> wfreq;

// The letter VPYP vocabulary is learned from the data.  With V=1 its deficient
// base measure has log p=0, so the seating counts carry the letter distribution.
#define VOCAB 1
#define STATES 50
#define _MIN_SLICE_ 0.01

vhmm::vhmm():_n(3),_mn(1),_m(20),_v(VOCAB),_k(0),_K(STATES),_wn(1),_fixed(false),_clamp_hits(0),_clamp_total(0),_a(1),_b(1),_pos(new vhdp(3)),_word(new vector<shared_ptr<hpyp> >),_letter(new vector<shared_ptr<vpyp> >) {
	set_alpha(5.);
	int target=10;
	while (_k<target || _word->empty()) _resize();
	_pos->cache_max(_k);
}

vhmm::vhmm(int n, int min_n, int m, int k):_n(n),_mn(min_n),_m(m),_v(VOCAB),_k(0),_K(STATES),_wn(1),_fixed(false),_clamp_hits(0),_clamp_total(0),_a(1),_b(1),_pos(new vhdp(n)),_word(new vector<shared_ptr<hpyp> >),_letter(new vector<shared_ptr<vpyp> >) {
	if (_mn < 1 || _mn > _n || _n < 1 || _k < 0) throw "invalid vhmm parameters";
	set_alpha(5.);
	int target=k;
	while (_k<target || _word->empty()) _resize();
	_pos->cache_max(_k);
}

vhmm::~vhmm() {
}

void vhmm::set_alpha(double value) {
	if (value <= 0) throw "invalid vhmm alpha";
	for (int i=0; i<_pos->n(); ++i) _pos->set_alpha(i, value);
}

// Inference uses a frozen state space: _fixed prevents _build_lattice from
// admitting k >= _k and calling _resize() to create new restaurants.
void vhmm::set_fixed() {
	_fixed = true;
}

void vhmm::emission_argmax(sentence& s) {
	for (int i=0; i<s.size(); ++i) {
		int best=1; double bv=-DBL_MAX;
		for (int k=1; k<=_k; ++k) {
			double v=_emission_word_lp(s,i,k);
			if (v>bv) { bv=v; best=k; }
		}
		s.wd(i).pos=best;
	}
}

int vhmm::n() {
	return _n;
}

int vhmm::min_n() {
	return _mn;
}

int vhmm::m() {
	return _m;
}

int vhmm::k() {
	return _k;
}

long long vhmm::clamp_hits() const {
	return _clamp_hits;
}

// Count the context tree used by the order posterior; each node represents a
// stop/pass decision at one history depth.
static void count_nodes(context *c, int depth, std::vector<long long>& n) {
	if ((int)n.size() <= depth) n.resize(depth+1, 0);
	++n[depth];
	for (auto& x : static_cast<vhdp_context*>(c)->child())
		count_nodes(x.second.get(), depth+1, n);
}

void vhmm::dump_tree_stats(const char *tag) {
	context *r=_pos->h();
	std::vector<long long> n;
	count_nodes(r,0,n);
	if (getenv("VHDP_ORDPROBE")) {
		extern double ORD_score[8], ORD_lpk[8];
		extern long long ORD_n[8], ORD_have[8], ORD_stop[8], ORD_pass[8], ORD_hn[8];
		std::cerr<<"[ord-score] "<<tag;
		for (int j=0; j<_n; ++j) if (ORD_n[j])
			std::cerr<<" n"<<j+1<<"[score="<<ORD_score[j]/ORD_n[j]
				<<" lp="<<ORD_lpk[j]/ORD_n[j]
				<<" ctx="<<(double)ORD_have[j]/ORD_n[j]
				<<" stop="<<(ORD_hn[j]?(double)ORD_stop[j]/ORD_hn[j]:0)
				<<" pass="<<(ORD_hn[j]?(double)ORD_pass[j]/ORD_hn[j]:0)<<"]";
		std::cerr<<"\n";
		for (int j=0;j<8;++j){ORD_score[j]=ORD_lpk[j]=0.;ORD_n[j]=ORD_have[j]=ORD_stop[j]=ORD_pass[j]=ORD_hn[j]=0;}
	}
	std::cerr<<"[alpha] "<<tag;
	for (int i=0; i<_pos->n(); ++i) std::cerr<<" a"<<i<<"="<<_pos->alpha(i);
	std::cerr<<"\n";
	std::cerr<<"[tree] "<<tag<<" root stop="<<r->stop()<<" pass="<<r->pass()
		<<" P(stop)="<<(double)r->stop()/(r->stop()+r->pass())<<" nodes";
	for (size_t d=0; d<n.size(); ++d) std::cerr<<" d"<<d<<":"<<n[d];
	std::cerr<<"\n";
}

void vhmm::cache_max() {
	_pos->cache_max(_k);
}

void vhmm::refresh_cache() {
	_pos->cache_max(_k);
}

namespace npbnlp { extern long long VHDP_N_cmax_d[8]; extern long long VHDP_N_cmax, VHDP_N_lp, VHDP_N_pr, VHDP_N_pr_hit, VHDP_N_gamma, VHDP_N_gamma_hit, VHDP_N_pass, VHDP_N_pass_hit, VHDP_N_binf, VHDP_N_binf_hit; }

void vhmm::dump_timing() const {
	auto pct=[](long long h,long long n){ return n? 100.0*h/n : 0.0; };
	std::cerr<<"cache_max node visits="<<VHDP_N_cmax<<" by depth";
	for(int d=0;d<4;++d) std::cerr<<" "<<d<<":"<<VHDP_N_cmax_d[d];
	std::cerr<<"\n";
	std::cerr<<"lp calls="<<VHDP_N_lp
		<<" pr="<<VHDP_N_pr<<" hit="<<pct(VHDP_N_pr_hit,VHDP_N_pr)<<"%"
		<<" gamma="<<VHDP_N_gamma<<" hit="<<pct(VHDP_N_gamma_hit,VHDP_N_gamma)<<"%"
		<<" pass="<<VHDP_N_pass<<" hit="<<pct(VHDP_N_pass_hit,VHDP_N_pass)<<"%"
		<<" beta_inf="<<VHDP_N_binf<<" hit="<<pct(VHDP_N_binf_hit,VHDP_N_binf)<<"%\n";
	std::cerr<<"retries="<<N_retry<<" sentences="<<N_sent<<" per_sent="<<(N_sent? (double)N_retry/N_sent:0)<<"\n";
	std::cerr<<"timing order="<<T_ord<<" slice="<<T_slice<<" lattice="<<T_lat<<" forward="<<T_fwd<<" backward="<<T_bwd<<"\n";
}

double vhmm::mean_beam() const {
	return _beam_positions ? (double)_beam_total/_beam_positions : 0.;
}

long long vhmm::clamp_total() const {
	return _clamp_total;
}

void vhmm::log_probability_parts(sentence& s, double& em, double& tr) {
	for (int i=0;i<=s.size();++i) {
		word& w=s.wd(i);
		em+=_emission_word_lp(s,i,w.pos);
		tr+=_pos->lp(w.pos,_pos->find_exist(s,i,s.n[i]));
	}
}

void vhmm::draw_order(sentence& s) {
	_draw_order(s);
}

double vhmm::log_probability(sentence& s) {
	double x=0.;
	for (int i=0;i<=s.size();++i) {
		word& w=s.wd(i);
		x+=_emission_word_lp(s,i,w.pos);
		x+=_pos->lp(w.pos,_pos->find_exist(s,i,s.n[i]));
	}
	return x;
}

// Set the vocabulary size used by every letter base measure; the state base
// measure is configured independently by the transition model.
/* _v is a count the letter models maintain themselves, so there is nothing to
   seed; set_k() carries the state ceiling that this also used to set.
void vhmm::set(int v, int k) {
	_v=v;
	_K=k;
	_k=min(_k,_K);
	for (auto it = _letter->begin(); it != _letter->end(); ++it)
		(*it)->set_v(_v);
}
*/

void vhmm::set_k(int k) {
	if (k <= 0)
		return;
	_K=k;
	_k=min(_k,_K);
}

void vhmm::slice(double a, double b) {
	if (a > 0 && b > 0) {
		_a=a;
		_b=b;
	}
}

int vhmm::_draw_order(sentence& s, int i) {
	int drawn = _pos->draw_n(s, i);
	int upper = min(i+2, _n);
	int lower = min(i+2, _mn);
	return max(lower, min(upper, drawn));
}

void vhmm::_draw_order(sentence& s) {
	for (int i=0; i<=s.size(); ++i) {
		int x = _draw_order(s, i);
		s.n[i] = x;
		if (i < s.size()) s.wd(i).n = x;
	}
}

// vlattice owns the orders consumed by slice, lattice construction, and FFBS.
// Updating the sentence and the lattice together keeps the sampled (state,
// order) pair consistent and preserves the order used for seating.
void vhmm::_draw_order(vlattice& l) {
	for (int i=0; i<=l.s.size(); ++i)
		l.set_order(i, _draw_order(l.s, i));
}

// Prune a context subtree when its upper bound on log p(z_t=k|c) is below the
// slice threshold; every deeper context then fails the same threshold.
static bool ctx_prune(context *c, int k, double u) {
	return c && static_cast<vhdp_context*>(c)->subtree_max(k) < u;
}

static context* ctx_find(context *c, int k) {
	return c ? static_cast<vhdp_context*>(c)->find(k) : NULL;
}

// The emission n-gram conditions on preceding observed words, never on latent
// states, and stops at the deepest existing context.  Order 1 uses the root.
context* vhmm::_emission_ctx(sentence& s, int i, int k) const {
	return (*_word)[k]->find(s,i,_wn);
}

// Draw the emission depth from the VPYLM stop/pass posterior using the node-local
// hpyp::lp(word&, context*) probability.  The emission probability marginalizes
// over depths by summing the stop probability at each context times the word
// probability after passing through its ancestors.  _word remains hpyp because
// virtual lp() dispatch changes the base distribution used by the recursion.
double vhmm::_emission_word_lp(sentence& s, int i, int k) {
	word& w=s.wd(i);
	context *c=_emission_ctx(s,i,k);
	if (_wn <= 1)
		return (*_word)[k]->lp(w,c);
	// Products are accumulated in logs: log(a/d)=log(a)-log(d), and lse sums
	// alternative depths without converting back to ordinary probabilities.
	double ln_pr=0., z=0.;
	while (c) {
		int st=c->stop(), ps=c->pass();
		double ln_stop=log(st)-log(st+ps);
		double ln_pass=log(ps)-log(st+ps);
		z=math::lse(z+ln_pass, ln_stop, (z==0.));
		ln_pr=math::lse(ln_pr+ln_pass, ln_stop+(*_word)[k]->lp(w,c), (ln_pr==0.));
		c=c->parent();
	}
	return ln_pr-z;
}

int vhmm::_draw_emission_n(sentence& s, int i, int k) {
	if (_wn <= 1)
		return 1;
	vector<double> table;
	double ln_pass=0., ln_stop=0., lp_cache=0.;
	context *c=(*_word)[k]->h();
	word& w=s.wd(i);
	int j=1;
	// Depth j's weight is stop_j times the passes of the depths ABOVE it, so the
	// candidate has to be recorded before this depth's own pass is accumulated.
	do {
		if (c) {
			int st=c->stop(), ps=c->pass();
			lp_cache=(*_word)[k]->lp(w,c);
			ln_stop=log(st)-log(st+ps);
			table.push_back(lp_cache+ln_stop+ln_pass);
			ln_pass+=log(ps)-log(st+ps);
			c=c->find(s[i-j]);
		} else {
			ln_stop=-log(2.);
			table.push_back(lp_cache+ln_stop+ln_pass);
			ln_pass+=-log(2.);
		}
	} while (i-j >= -1 && j++ < _wn);
	return 1+rd::ln_draw(table);
}

void vhmm::_set_ed(int idx, int i, int d) {
	if (idx < 0)
		return;
	if ((int)_ed.size() <= idx)
		_ed.resize(idx+1);
	if ((int)_ed[idx].size() <= i)
		_ed[idx].resize(i+1, 1);
	_ed[idx][i]=d;
}

int vhmm::_get_ed(int idx, int i) const {
	if (idx < 0 || idx >= (int)_ed.size() || i >= (int)_ed[idx].size())
		return 1;
	return _ed[idx][i];
}

void vhmm::set_word_ngram(int n) {
	if (n < 1)
		throw "invalid vhmm word ngram order";
	if (n == _wn)
		return;
	_wn=n;
	// hpyp has no setter for its context depth, so rebuild the restaurants.  This
	// is valid only before any customer has been seated.
	int target=_k;
	_word->clear(); _letter->clear(); _k=0;
	while (_k<target || _word->empty()) _resize();
}

int vhmm::word_ngram() const { return _wn; }

void vhmm::init(sentence& s) {
	init(s,-1);
}

void vhmm::init(sentence& s, int idx) {
	if (_wn > 1 && idx < 0)
		throw "vhmm::init needs the corpus index once the emission is an n-gram";
	lock_guard<mutex> l(_mutex);
	shared_ptr<wid> d=wid::create();
	for (int i=0; i<s.size(); ++i) {
		s.wd(i).id=(*d)[s.wd(i)] == 1 ? d->index(s.wd(i)) : (*d)[s.wd(i)];
		++wfreq[s.wd(i).id];
	}
	s.n[s.size()]=_mn;
	s.wd(s.size()).pos=0;
	s.wd(s.size()).n=s.n[s.size()];
	// Draw the state and its order jointly so the initial assignment follows the
	// same generative factors as later sampling.
	for (int i=0; i<s.size(); ++i) {
		word& w=s.wd(i);
		vector<double> table;
		vector<int> table_n, table_k;
		for (int k=1; k<=_k; ++k) {
			double ln_pr_em=_emission_word_lp(s,i,k);
			context *c=_pos->h();
			double ln_pass=0., ln_pr=_pos->lp(k,_pos->h());
			for (int j=1; j<=_n && i-j>=-2; ++j) {
				table.push_back(ln_pr_em+_pos->lp_order(k,c,ln_pass,ln_pr));
				table_n.push_back(j);
				table_k.push_back(k);
				if (c) c=ctx_find(c,s.wd(i-j).pos);
			}
		}
		int id=rd::ln_draw(table);
		w.pos=table_k[id];
		s.n[i]=max(_mn,min(min(i+2,_n),table_n[id]));
		w.n=s.n[i];
		context *p=_pos->make(s,i,s.n[i]);
		int ed=_draw_emission_n(s,i,w.pos);
		context *h=(*_word)[w.pos]->make(s,i,ed);
		(*_word)[w.pos]->add(w,h);
		_set_ed(idx,i,ed);
		_pos->add(w.pos,p);
		if (w.pos==_k) _resize();
	}
	word& e=s.wd(s.size());
	int eed=_draw_emission_n(s,s.size(),0);
	context *h=(*_word)[0]->make(s,s.size(),eed);
	(*_word)[0]->add(e,h);
	_set_ed(idx,s.size(),eed);
	context *p=_pos->make(s,s.size(),s.n[s.size()]);
	_pos->add(0,p);
}

void vhmm::inference_init(sentence& s) {
	shared_ptr<wid> d=wid::create();
	for (int i=0; i<s.size(); ++i)
		s.wd(i).id=(*d)[s.wd(i)] == 1 ? d->index(s.wd(i)) : (*d)[s.wd(i)];
	s.wd(s.size()).pos=0;
	// Draw each initial state and order jointly from the generative process.  No
	// customers are seated here.  _k is the highest existing state and _word has
	// _k+1 restaurants including state 0, so all states 1.._k are candidates.
	int top=_k;
	for (int i=0; i<s.size(); ++i) {
		word& w=s.wd(i);
		vector<double> table;
		vector<int> table_n, table_k;
		for (int k=1; k<=top; ++k) {
			double ln_pr_em=_emission_word_lp(s,i,k);
			context *c=_pos->h();
			double ln_pass=0., ln_pr=_pos->lp(k,_pos->h());
			for (int j=1; j<=_n && i-j>=-2; ++j) {
				table.push_back(ln_pr_em+_pos->lp_order(k,c,ln_pass,ln_pr));
				table_n.push_back(j);
				table_k.push_back(k);
				if (c) c=ctx_find(c,s.wd(i-j).pos);
			}
		}
		int id=rd::ln_draw(table);
		w.pos=table_k[id];
		s.n[i]=max(_mn,min(min(i+2,_n),table_n[id]));
		w.n=s.n[i];
	}
	// Sample the EOS order as part of the sentence's latent order sequence.
	s.n[s.size()]=_draw_order(s,s.size());
	s.wd(s.size()).n=s.n[s.size()];
}

void vhmm::add(sentence& s) {
	add(s,-1);
}

void vhmm::add(sentence& s, int idx) {
	if (_wn > 1 && idx < 0)
		throw "vhmm::add needs the corpus index once the emission is an n-gram";
	lock_guard<mutex> l(_mutex);
	shared_ptr<wid> d=wid::create();
	for (int i=0; i<s.size(); ++i) {
		s.wd(i).id=(*d)[s.wd(i)] == 1 ? d->index(s.wd(i)) : (*d)[s.wd(i)];
		++wfreq[s.wd(i).id];
	}
	for (int i=0; i<=s.size(); ++i) {
		word& w=s.wd(i);
		int ed=_draw_emission_n(s,i,w.pos);
		context *h=(*_word)[w.pos]->make(s,i,ed);
		(*_word)[w.pos]->add(w,h);
		_set_ed(idx,i,ed);
		context *p=_pos->make(s,i,s.n[i]);
		_pos->add(w.pos,p);
		if (w.pos==_k) _resize();
	}
}

void vhmm::remove(sentence& s) {
	remove(s,-1);
}

void vhmm::remove(sentence& s, int idx) {
	if (_wn > 1 && idx < 0)
		throw "vhmm::remove needs the corpus index once the emission is an n-gram";
	lock_guard<mutex> l(_mutex);
	shared_ptr<wid> d=wid::create();
	for (int i=0; i<=s.size(); ++i) {
		// Remove the customer from the same context depth recorded at seating time.
		context *h=(*_word)[s.wd(i).pos]->find(s,i,_get_ed(idx,i));
		(*_word)[s.wd(i).pos]->remove(s.wd(i),h);
	}
	for (int i=0; i<=s.size(); ++i) {
		word& w=s.wd(i);
		context *p=_pos->find(s,i,s.n[i]);
		_pos->remove(w.pos,p);
	}
	for (int i=0; i<s.size(); ++i) {
		word& w=s.wd(i);
		if (--wfreq[w.id] == 0) {
			d->remove(w);
			wfreq.erase(w.id);
		}
	}
	while (_k>0 && (*_word)[_k]->h()->c()==0)
		_shrink();
}

void vhmm::estimate(int iter) {
#ifdef _OPENMP
	int keep=omp_get_max_threads();
	omp_set_num_threads(1);
#endif
	// Resample emission hyperparameters once per epoch; iter controls transition
	// estimation and the restaurant Gibbs updates.
	for (int i=0; i<=_k; ++i) {
		(*_word)[i]->gibbs(iter);
		(*_word)[i]->estimate(1);
		(*_letter)[i]->estimate(1);
	}
	_pos->estimate(iter);
#ifdef _OPENMP
	omp_set_num_threads(keep);
#endif
}

void vhmm::poisson_correction(int x) {
	for (int i=1; i<=_k; ++i)
		(*_word)[i]->poisson_correction(x);
}

void vhmm::_slice(vlattice& l) {
	for (int t=0; t<(int)l.k.size(); ++t) {
		int order=l.order(t);
		context *c=_pos->find_exist(l.s,t,order);
		int current=t==l.s.size()?0:min(_k,l.s.wd(t).pos);
		if (t<l.s.size()) {
			++_clamp_total;
			if (l.s.wd(t).pos>_k) ++_clamp_hits;
		}
		beta_distribution be;
		double x=_pos->lp(current,c)+log(max(_MIN_SLICE_,be(_a,_b)));
		l.slice(t,x);
		l.cur[t]=current;
	}
}

int vhmm::_build_lattice(vlattice& l, bool best) {
	// Keep the union of states admitted by successive (order,u) draws.  A vector
	// therefore needs an explicit membership check before insertion.
	int size=l.s.size()+1;
	l.k[size-1].push_back(0); // EOS state (id 0)
	for (int i=0; i<size-1; ++i) {
		int n=l.order(i);
		context *root=_pos->h();
		// Test the boundary state in the context used for the slice variable:
		// u_i is based on lp(z_i, find_exist(i,n_i)), not the root.  Since lp is
		// decreasing in k at a fixed context, the loop stops after the valid states.
		context *cx=_pos->find_exist(l.s,i,n);
		if (getenv("VHMM_CTXDEPTH")) {
			int d=0; context *probe=_pos->h();
			for (int m=1; m<n; ++m) { context *nx=static_cast<vhdp_context*>(probe)->find(l.s.wd(i-m).pos); if (!nx) break; probe=nx; ++d; }
			fprintf(stderr,"[ctx] order=%d reached=%d\n",n,d+1);
		}
		// State _k exists and is trained.  When the model is unfixed the second loop
		// owns that boundary state and may grow the model; when fixed the first loop
		// must include it explicitly.
		//
		// Unfixed, both loops run under _grow: they share the bound _k, and another
		// worker growing the model between them would otherwise leave the states it
		// added untested here.  Fixed, _k cannot move, so no lock is needed.
		auto admit=[&](int i,int n,context *cx) {
			for (int k=1; k<=_K && (_fixed ? k<=_k : k<_k); ++k) {
				if (_pos->max_lp(k,n-1)>=l.u(i) &&
				    find(l.k[i].begin(),l.k[i].end(),k)==l.k[i].end())
					l.k[i].push_back(k);
			}
			if (_fixed)
				return;
			for (int k=_k; k<=_K && _pos->lp(k,cx)>=l.u(i); ++k) {
				if (find(l.k[i].begin(),l.k[i].end(),k)==l.k[i].end())
					l.k[i].push_back(k);
				_resize_locked();
			}
		};
		if (_fixed) {
			admit(i,n,cx);
		} else {
			lock_guard<mutex> lk(_grow);
			admit(i,n,cx);
		}
		if (l.k[i].empty())
			return 1;
		_beam_total+=l.k[i].size();
		++_beam_positions;
	}
	if (best) {
		for (int i=0; i<size-1; ++i) {
			int n=l.order(i);
			context *c=_pos->find_exist(l.s,i,n);
			int state=l.k[i][0];
			for (int x : l.k[i])
				if (_pos->lp(x,c)>_pos->lp(state,c)) state=x;
			l.k[i].clear();
			l.k[i].push_back(state);
		}
	}
	return 0;
}

sentence vhmm::sample(sentence& s) {
	return _sample(s,false);
}

sentence vhmm::_sample(sentence& s, bool best) {
	vlattice l(s);
	int guard=0;
	do {
		_draw_order(l);
		_slice(l);
		if (++guard>100) {
			_mn=1;
			_draw_order(l);
			_slice(l);
			if (_build_lattice(l,best))
				throw "vhmm: lattice is empty even at min order 1";
			break;
		}
	} while (_build_lattice(l,best));
	int size=l.s.size()+1;
	vector<vt> alpha(size);
	for (int i=0; i<size; ++i) {
		for (auto it=l.begin(i); it!=l.end(i); ++it)
			_forward(l,l.u(i),i,*it,l.order(i-1),l.order(i),alpha);
		double z=_marginalize(alpha[i]);
		if (std::isfinite(z))
			_scale(alpha[i],z);
		if (getenv("VHMM_PROBE")) {
			int ninit=0;
			for (auto it=alpha[i].begin(); it!=alpha[i].end(); ++it)
				if (it->second && it->second->is_init()) ++ninit;
			cerr<<"[fwd] i="<<i<<" order="<<l.order(i)<<" prev_order="<<l.order(i-1)
				<<" |lat|="<<distance(l.begin(i),l.end(i))<<" init="<<ninit
				<<" children="<<distance(alpha[i].begin(),alpha[i].end())<<" u="<<l.u(i)<<"\n";
		}
	}
	int k=0; // EOS state (id 0)
	for (int i=size-1; i>0; --i) {
		k=_backward(l,l.u(i),i,k,l.order(i-1),l.order(i),alpha);
		l.s.wd(i-1).pos=k;
	}
	return l.s;
}

sentence vhmm::parse(sentence& s) {
	// Inference alternates MCMC draws of the order and states.  Their transition
	// factors depend on each other, so the lattice must retain all slice-admitted
	// states rather than collapse each column to a Viterbi argmax.
	return _sample(s,false);
}

// Log density of the slice variable under Beta(a,b).  Since
// mu_t=p(z_t|c) r with r~Beta(a,b), the transition factor cancels and the
// remaining density is Beta_pdf(mu_t/p(z_t|c);a,b) under mu_t<p(z_t|c).
// The cutoff makes the ratio lie in (0,1), so the density is evaluated in logs.
double vhmm::_ln_slice_density(double u, double ln_pr_tr) const {
	if (_a == 1. && _b == 1.)
		return 0.;
	double ln_ratio = u - ln_pr_tr;
	if (ln_ratio > 0.)
		ln_ratio = 0.;
	double ln_1m = log1p(-exp(ln_ratio));
	if (!std::isfinite(ln_1m))
		ln_1m = -1e300;
	return (_a-1.)*ln_ratio + (_b-1.)*ln_1m - (lgamma(_a)+lgamma(_b)-lgamma(_a+_b));
}

// Sum a forward-probability subtree into its parents, then rescale.
// alpha is stored as a tree indexed by the path of previous states, because the
// order at t may need to condition on several of them.  When t is LOWER order
// than t-1, the forward at t reads a shallower node, which must already hold the
// sum over the deeper paths below it -- that is this marginalisation.  (The other
// direction, t higher than t-1, is covered by _prior_lp: the states the shorter
// history never conditioned on get their generation probability multiplied in at
// the leaf.)  This is the sum over latent histories required by the marginal
// probability at the shallower context; math::lse performs that sum on the log
// axis.
// A node with no initialised children is a leaf and keeps the value the forward
// accumulated into it.
double vhmm::_marginalize(vt& node) {
	bool init=false;
	double z=0.;
	for (auto it=node.begin(); it!=node.end(); ++it) {
		if (!it->second || !it->second->is_init())
			continue;
		double child=_marginalize(*it->second);
		z=math::lse(z,child,!init);
		init=true;
	}
	if (init) {
		node.v=z;
		node.set(true);
	}
	return node.v;
}

// Divide the whole subtree by z (subtract in logs) so alpha stays bounded.
void vhmm::_scale(vt& node, double z) {
	if (node.is_init())
		node.v-=z;
	for (auto it=node.begin(); it!=node.end(); ++it)
		if (it->second)
			_scale(*it->second,z);
}




// Forward/backward recursion for FFBS over the slice lattice.  The alpha tree
// follows the history needed by the current order.

// vt::operator[] creates on miss; the recursion needs a lookup that does not.
// context::find is non-virtual and vhdp_context stores children in its own map;
// an unqualified call through context* would use the wrong prediction context.

static vt* vt_find(vt& node, int i) {
	for (auto it=node.begin(); it!=node.end(); ++it)
		if (it->first == i)
			return it->second.get();
	return NULL;
}

double vhmm::_emission_lp(vlattice& l, int i, int k) {
	return _emission_word_lp(l.s,i,k);
}

void vhmm::_forward(vlattice& l, double u, int i, int k, int m, int n, vector<vt>& a) {
	context *c=_pos->h();
	// The lattice has already selected candidates using the slice context; do not
	// repeat the boundary test at the root, which could reject an admitted state.
	if (n > 1) {
		if (i == 0) {
			context *_c=ctx_find(c,0);
			if (_c) c=_c;

			if (_pos->lp(k,c)<u)
				return;
			a[i][k].v=_emission_lp(l,i,k);
			a[i][k].set(true);
		} else {
			double ln_pr_em=_emission_lp(l,i,k);
			for (auto pr=l.begin(i-1); pr!=l.end(i-1); ++pr) {
				context *_c=ctx_find(c,*pr);
				double ln_pr_tr=_pos->lp(k,c);
				if (!_c && ln_pr_tr<u)
					continue;
				else if (_c)
					ln_pr_tr=_pos->lp(k,_c);
				vt *_b=vt_find(a[i-1],*pr);
				if (_b && _b->is_init()) {
					vector<int> prefix;
					_forward(_c,l,u,ln_pr_em,ln_pr_tr,i-1,k,*pr,m-1,n-1,_b,a[i][k],prefix);
				}
			}
		}
		return;
	} else { // independent state
		if (_pos->lp(k,c)<u)
			return;
		a[i][k].v=_emission_lp(l,i,k);
		a[i][k].set(true);
		return;
	}
}

void vhmm::_forward(context *c, vlattice& l, double u, double ln_pr_em, double ln_pr_tr, int i, int k, int j, int m, int n, vt *b, vt& a, vector<int>& prefix) {
	if (ctx_prune(c,k,u))
		return;
	if (n > 1) {
		for (auto pr=l.begin(i-1); pr!=l.end(i-1); ++pr) {
			bool pushed=false;
			vt *_b=b;
			if (m > 1) {
				_b=vt_find(*b,*pr);
				if (!_b || !_b->is_init())
					continue;
			} else {
				prefix.push_back(*pr);
				pushed=true;
			}
			context *_c=NULL;
			double _ln_pr_tr=ln_pr_tr;
			if (c) _c=ctx_find(c,*pr);
			if (!_c && ln_pr_tr<u) {
				if (pushed) prefix.pop_back();
				continue;
			} else {
				if (_c) _ln_pr_tr=_pos->lp(k,_c);
				_forward(_c,l,u,ln_pr_em,_ln_pr_tr,i-1,k,*pr,m-1,n-1,_b,a[j],prefix);
			}
			if (pushed) prefix.pop_back();
		}
		return;
	} else { // n == 1, end of the recursion
		if (ln_pr_tr<u)
			return;
		// Generate states omitted by the shorter history and add their transition
		// probabilities; this marginalizes the missing transitions with the proper
		// normalization.
		double prior=0.;
		context *p=_pos->h();
		bool fixed=false;
		for (int h=(int)prefix.size()-1; h>=0; --h) {
			prior+=_pos->lp(prefix[h],p);
			if (!fixed) {
				context *pp=ctx_find(p,prefix[h]);
				if (pp) p=pp; else fixed=true;
			}
		}
		a.v=math::lse(a.v,ln_pr_em+b->v+prior,!a.is_init());
		a.set(true);
		return;
	}
}

int vhmm::_backward(vlattice& l, double u, int i, int k, int m, int n, vector<vt>& a) {
	if (n > 1) {
		vector<double> table;
		vector<int> states;
		for (auto pr=l.begin(i-1); pr!=l.end(i-1); ++pr) {
			context *c=_pos->h();
			context *_c=ctx_find(c,*pr);
			double ln_pr_tr=_pos->lp(k,c);
			if (!_c && ln_pr_tr<u)
				continue;
			else if (_c)
				ln_pr_tr=_pos->lp(k,_c);
			vt *_b=vt_find(a[i-1],*pr);
			if (!_b || !_b->is_init())
				continue;
			bool cut=false;
			double lp=_backward(_c,l,u,ln_pr_tr,i-1,k,m-1,n-1,_b,cut);
			if (cut)
				continue;
			table.push_back(lp);
			states.push_back(*pr);
		}
		if (table.empty())
			throw "vhmm: no previous state survived the beam";
		return states[rd::ln_draw(table)];
	} else { // state i is independent
		vector<double> table;
		vector<int> states;
		for (auto pr=l.begin(i-1); pr!=l.end(i-1); ++pr) {
			vt *_b=vt_find(a[i-1],*pr);
			if (!_b || !_b->is_init())
				continue;
			table.push_back(_b->v);
			states.push_back(*pr);
		}
		if (table.empty()) {
			if (getenv("VHMM_PROBE")) {
				cerr<<"[bwd-indep-empty] i="<<i<<" k="<<k<<" m="<<m<<" n="<<n
					<<" |lat(i-1)|="<<distance(l.begin(i-1),l.end(i-1))
					<<" order(i-1)="<<l.order(i-1)<<" order(i)="<<l.order(i)
					<<" u(i)="<<l.u(i)<<"\n";
				int ninit=0;
				for (auto pr=l.begin(i-1); pr!=l.end(i-1); ++pr) {
					vt *bb=vt_find(a[i-1],*pr);
					if (bb && bb->is_init()) ++ninit;
				}
				cerr<<"   initialised alpha nodes at i-1: "<<ninit<<" (children of a[i-1]: "<<distance(a[i-1].begin(),a[i-1].end())<<")\n";
			}
			throw "vhmm: no previous state survived the beam";
		}
		return states[rd::ln_draw(table)];
	}
}

double vhmm::_backward(context *c, vlattice& l, double u, double ln_pr_tr, int i, int k, int m, int n, vt *b, bool& cutoff) {
	if (ctx_prune(c,k,u)) {
		cutoff=true;
		return 0.;
	}
	if (n > 1) {
		double z=0.;
		bool init=false;
		for (auto pr=l.begin(i-1); pr!=l.end(i-1); ++pr) {
			vt *_b=b;
			if (m > 1) {
				_b=vt_find(*b,*pr);
				if (!_b || !_b->is_init())
					continue;
			}
			context *_c=NULL;
			double _ln_pr_tr=ln_pr_tr;
			if (c) _c=ctx_find(c,*pr);
			if (_c) _ln_pr_tr=_pos->lp(k,_c);
			if (!_c && ln_pr_tr<u)
				continue;
			else {
				bool cut=false;
				double lp=_backward(_c,l,u,_ln_pr_tr,i-1,k,m-1,n-1,_b,cut);
				if (cut) continue;
				z=math::lse(z,lp,!init);
				init=true;
			}
		}
		cutoff=!init;
		return z;
	} else { // n == 1
		cutoff=(ln_pr_tr<u);
		return b->v;
	}
}

void vhmm::_resize() {
	lock_guard<mutex> lk(_grow);
	_resize_locked();
}

// For callers that already hold _grow; taking it again on a non-recursive mutex
// would deadlock.
void vhmm::_resize_locked() {
	// _build_lattice may grow the restaurants while other threads read them.
	// Reserve _K+2 entries before parallel sampling so push_back cannot invalidate
	// those reads; _grow serializes writers.
	if ((int)_word->capacity() < _K+2) {
		_word->reserve(_K+2);
		_letter->reserve(_K+2);
	}
	// _v stays at the hpyp default of 1 and the models count it up as characters
	// reach the root.  With _v=1 the character base is deficient (log p=0), so
	// seating counts alone determine the letter distribution.
	int cv = _v;
	if (_word->empty()) {
		_word->push_back(shared_ptr<hpyp>(new hpyp(_wn)));
		_letter->push_back(shared_ptr<vpyp>(new vpyp(_m)));
		(*_letter)[0]->set_v(cv); (*_word)[0]->set_base((*_letter)[0].get());
		return;
	}
	if (_k+1>_K) return;
	++_k;
	_word->push_back(shared_ptr<hpyp>(new hpyp(_wn)));
	_letter->push_back(shared_ptr<vpyp>(new vpyp(_m)));
	(*_letter)[_k]->set_v(cv); (*_word)[_k]->set_base((*_letter)[_k].get());
}
void vhmm::_shrink() { if (_k>0) { --_k; _word->pop_back(); _letter->pop_back(); } }

void vhmm::save(const char *file) {
	FILE *fp=fopen(file,"wb"); if (!fp) throw "failed to open save file in vhmm::save";
	if (fwrite(&_n,sizeof(int),1,fp)!=1 || fwrite(&_mn,sizeof(int),1,fp)!=1 || fwrite(&_m,sizeof(int),1,fp)!=1 || fwrite(&_v,sizeof(int),1,fp)!=1 || fwrite(&_k,sizeof(int),1,fp)!=1 || fwrite(&_K,sizeof(int),1,fp)!=1) throw "failed to write vhmm parameters";
	_pos->save(fp); for (int i=0;i<=_k;++i) { (*_word)[i]->save(fp); (*_letter)[i]->save(fp); }
	// Store the emission order after the restaurants with a format marker.
	int marker=1;
	if (fwrite(&marker,sizeof(int),1,fp)!=1 || fwrite(&_wn,sizeof(int),1,fp)!=1) throw "failed to write vhmm word ngram order";
	fclose(fp);
}

void vhmm::load(const char *file) {
	FILE *fp=fopen(file,"rb"); if (!fp) throw "failed to open model file in vhmm::load";
	if (fread(&_n,sizeof(int),1,fp)!=1 || fread(&_mn,sizeof(int),1,fp)!=1 || fread(&_m,sizeof(int),1,fp)!=1 || fread(&_v,sizeof(int),1,fp)!=1 || fread(&_k,sizeof(int),1,fp)!=1 || fread(&_K,sizeof(int),1,fp)!=1) throw "failed to read vhmm parameters";
	_pos=shared_ptr<vhdp>(new vhdp(_n)); _word->clear(); _letter->clear(); _pos->load(fp);
	for (int i=0;i<=_k;++i) {
		_word->push_back(shared_ptr<hpyp>(new hpyp(_wn))); _letter->push_back(shared_ptr<vpyp>(new vpyp(_m)));
		(*_word)[i]->load(fp); (*_letter)[i]->load(fp);
		// _v is learned and restored by hpyp::load; do not overwrite it here.
		(*_word)[i]->set_base((*_letter)[i].get());
	}
	// Restaurants require _wn at construction time, so the trailer verifies the
	// caller's order; a missing trailer denotes order 1.
	int marker=0, wn=1;
	if (fread(&marker,sizeof(int),1,fp)==1 && marker==1) {
		if (fread(&wn,sizeof(int),1,fp)!=1) throw "failed to read vhmm word ngram order";
	}
	if (wn != _wn) throw "vhmm --word_ngram does not match the model";
	fclose(fp);
}
