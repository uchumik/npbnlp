#include "ghmm.h"
#include "gseq.h"
#include "rd.h"
#include <getopt.h>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <map>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace npbnlp;

static int max_order=3, min_order=1, states=10, max_states=50, epochs=500, threads=4;
static int estimate_iter=20, sample_mode=0, have_seed=0, dump_model=0;
static double alpha=5., slice_a=1., slice_b=1.;
static double niw_kappa0=0.1, niw_nu0=-1., niw_lambda=-1.;
static unsigned int seed_value=0;
static string train_file, parse_file, model_file("ghmm.model");

static void usage(int argc, char **argv) {
	cerr << argv[0] << " --train FILE|--parse FILE [options]\n";
	cerr << "  --model FILE       where to save/load the model\n";
	cerr << "  --max_order N      maximum transition order (default 3)\n";
	cerr << "  --min_order N      minimum transition order (default 1)\n";
	cerr << "  --states N         initial states (default 10)\n";
	cerr << "  --max_states N     ceiling on the state count (default 50)\n";
	cerr << "  --epoch N          sweeps (default 500)\n";
	cerr << "  --threads N        block size for the parallel sweep (default 4)\n";
	cerr << "  --alpha A          initial HDP concentration (default 5)\n";
	cerr << "  --niw_kappa0 K     NIW prior strength on the mean (default 0.1)\n";
	cerr << "  --niw_nu0 N        NIW degrees of freedom (default d+2)\n";
	cerr << "  --niw_lambda L     NIW scale; default is taken from the data\n";
	cerr << "  --niw_sample       draw (mu, Sigma) explicitly instead of\n";
	cerr << "                     marginalising to a Student-t\n";
	cerr << "  --seed N           seed the sampler\n";
	cerr << "run with OMP_WAIT_POLICY=passive when using --threads > 1\n";
	cerr << "\ninput: one observation per line, whitespace-separated reals;\n";
	cerr << "       a blank line ends a sequence, '#' starts a comment.\n";
	exit(1);
}

static void read_param(int argc, char **argv) {
	int idx=0, c=0;
	static option opts[]={{"train",1,0,0},{"parse",1,0,0},{"model",1,0,0},
		{"max_order",1,0,0},{"min_order",1,0,0},{"states",1,0,0},{"max_states",1,0,0},
		{"epoch",1,0,0},{"threads",1,0,0},{"alpha",1,0,0},{"slice_a",1,0,0},{"slice_b",1,0,0},
		{"niw_kappa0",1,0,0},{"niw_nu0",1,0,0},{"niw_lambda",1,0,0},{"niw_sample",0,0,0},
		{"estimate_iter",1,0,0},{"seed",1,0,0},{"dump_model",0,0,0},{0,0,0,0}};
	while ((c=getopt_long(argc,argv,"",opts,&idx))!=-1) {
		if (c!=0) usage(argc,argv);
		string o(opts[idx].name);
		if (o=="train") train_file=optarg;
		else if (o=="parse") parse_file=optarg;
		else if (o=="model") model_file=optarg;
		else if (o=="max_order") max_order=atoi(optarg);
		else if (o=="min_order") min_order=atoi(optarg);
		else if (o=="states") states=atoi(optarg);
		else if (o=="max_states") max_states=atoi(optarg);
		else if (o=="epoch") epochs=atoi(optarg);
		else if (o=="threads") threads=atoi(optarg);
		else if (o=="alpha") alpha=atof(optarg);
		else if (o=="slice_a") slice_a=atof(optarg);
		else if (o=="slice_b") slice_b=atof(optarg);
		else if (o=="niw_kappa0") niw_kappa0=atof(optarg);
		else if (o=="niw_nu0") niw_nu0=atof(optarg);
		else if (o=="niw_lambda") niw_lambda=atof(optarg);
		else if (o=="niw_sample") sample_mode=1;
		else if (o=="estimate_iter") estimate_iter=atoi(optarg);
		else if (o=="seed") { seed_value=(unsigned int)atoi(optarg); have_seed=1; }
		else if (o=="dump_model") dump_model=1;
	}
	if (train_file.empty() && parse_file.empty() && !dump_model)
		usage(argc,argv);
}

// state <TAB> order <TAB> x_1 x_2 ..., a blank line between sequences, which is
// the prototype's ghmm::dump format.
static void dump(gsentence& g) {
	for (int i=0; i<g.size(); ++i) {
		fvector& x=g.vec(i);
		cout << x.pos << "\t" << x.n;
		for (int j=0; j<x.size(); ++j)
			cout << "\t" << x[j];
		cout << "\n";
	}
	cout << "\n";
}

static void configure(ghmm& h, gio& f) {
	h.set_alpha(alpha);
	h.set_k(max_states);
	h.slice(slice_a, slice_b);
	h.set_sample_mode(sample_mode!=0);
	if (niw_lambda > 0. || niw_nu0 > 0.) {
		int d=f.dim();
		vector<double> mu0(d, 0.);
		double nu0 = niw_nu0 > 0. ? niw_nu0 : d+2;
		double lambda = niw_lambda > 0. ? niw_lambda : 1.;
		h.set_prior(mu0, niw_kappa0, nu0, lambda);
	} else {
		// mean and scale from the data, but the requested kappa either way
		h.init_prior(f, niw_kappa0);
	}
	h.set_corpus(f);
	if (sample_mode) h.seed_sample();
}

static int train() {
	gio f(train_file.c_str());
	ghmm h(max_order, min_order, states, f.dim());
	configure(h, f);
	int size=f.size();
	for (int i=0; i<size; ++i) h.init(i);
	h.refresh_cache();
	for (int e=0; e<epochs; ++e) {
		h.refresh_cache();
		int block=threads<1?1:threads;
		for (int id=0; id<size; id+=block) {
			int last=min(id+block,size);
			for (int j=id;j<last;++j) h.remove(j);
			h.cache_max();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(block)
#endif
			for (int j=id;j<last;++j) {
				try { h.sample(j); }
				catch (const char *ex) { cerr<<"ghmm: "<<ex<<" (sequence "<<j<<")\n"; throw; }
			}
			for (int j=id;j<last;++j) h.add(j);
		}
		h.estimate(estimate_iter);
		map<int,long long> use;
		for (int j=0;j<size;++j) { h.store(j,f[j]); for (int i=0;i<f[j].size();++i) ++use[f[j].vec(i).pos]; }
		cerr<<"epoch "<<e+1<<" state="<<h.k()<<" live="<<use.size()<<"\n";
	}
	for (int j=0;j<size;++j) { h.store(j,f[j]); dump(f[j]); }
	if (getenv("GHMM_DUMP_PARAMS")) h.dump_posterior(stderr);
	h.save(model_file.c_str());
	return 0;
}

static int parse() {
	gio f(parse_file.c_str());
	ghmm h(max_order, min_order, states, f.dim());
	h.load(model_file.c_str());
	// Inference must not invent states: load() replaces _k without rebuilding the
	// containers the constructor sized, so an unfixed model could grow past them.
	h.set_fixed();
	h.set_corpus(f);
	h.refresh_cache();
	int size=f.size();
	for (int i=0; i<size; ++i) {
		h.inference_init(i); // never seats anyone: the loaded model stays as it is
		// Inference is MCMC, as in vhmm: the order and the states depend on each
		// other, so there is no cheap Viterbi over the pair.
		for (int j=0; j<20; ++j) h.parse(i);
		h.store(i, f[i]);
		dump(f[i]);
	}
	return 0;
}

int main(int argc, char **argv) {
	read_param(argc, argv);
	try {
#ifdef _OPENMP
		// Before reseeding: generator::reseed() sizes its per-thread state from
		// omp_get_max_threads(), and the workers index it by thread number.
		omp_set_num_threads(threads<1?1:threads);
		omp_set_dynamic(0);
#endif
		if (have_seed) { seed::create()->set(seed_value); generator::reseed(); }
		if (dump_model) {
			// Load and print the model's own state, for checking a round trip.
			ghmm h(max_order, min_order, states, 1);
			h.load(model_file.c_str());
			h.set_fixed();
			h.dump_posterior(stdout);
			return 0;
		}
		if (!train_file.empty()) return train();
		return parse();
	} catch (const char *ex) {
		cerr << ex << endl;
		return 1;
	}
}
