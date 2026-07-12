// sne: unsupervised NER via the Switching NPYLM (WO-006 phase 2 CLI).
// Pipeline mirrors ne.cc: raw text -> phsmm tokenizer -> word lattice, then a
// blocked-Gibbs loop (remove -> sample -> add) over the snpylm generative model.
// Unsupervised only: no --pretrain / supervised anchoring.
#include"snpylm.h"
#include"phsmm.h"
#include"rd.h"
#include"util.h"
#include"io.h"
#include"chunktype.h"
#include<chrono>
#include<getopt.h>
#include<cstring>
#include<iostream>
#ifdef _OPENMP
#include<omp.h>
#endif

#define check(opt,arg) (strcmp(opt,arg) == 0)
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

using namespace std;
using namespace npbnlp;

static int n = 2;
static int hn = 1;   // NE surface chunk-PYP order (unigram over spans)
static int hl = 8;   // letter VPYP order
static int k = 10;
static int K = 20;
static int threads = 4;
static int epoch = 500;
static int snap = 5;
static double a = 1;
static double b = 5;
static double gamma_ = 10.0;
static double alpha_ = 1.0;
static int init_random = 0; // 0 = all-O seed (default); 1 = prior-sample init
static int anneal = 0;      // >0: warm up emission temperature over N epochs
static double tau0 = 2.0;   // initial emission temperature for the warm-up
static string prefix("snpylm");
static string train;
static string test;
static string model("snpylm.model");
static string cdic("sne.dic");
static string tokenizer("phsmm.model");
static string wdic("ma.dic");
static shared_ptr<phsmm> toklm;

void progress(const char *s, int i, double pct) {
	int val = (int)(pct * 100);
	int lpad = (int)(pct * PBWIDTH);
	int rpad = PBWIDTH - lpad;
	printf("\r%s %4d %3d%% [%.*s%*s]", s, i, val, lpad, PBSTR, rpad, "");
	fflush(stdout);
}

void usage(int argc, char **argv) {
	cout << "[Usage]" << *argv << " [options]\n";
	cout << "[example]\n";
	cout << *argv << " --train file --tokenizer phsmm.model --wdic ma.dic --model file_to_save --cdic chunk.dic\n";
	cout << *argv << " --parse file --tokenizer phsmm.model --wdic ma.dic --model modelfile --cdic chunk.dic\n";
	cout << "[options]\n";
	cout << "-n, --order=int(background template n-gram order, default 2; >=2 required)\n";
	cout << "-k, --class=int(number of NE classes, default 20)\n";
	cout << "-e, --epoch=int(default 500)\n";
	cout << "-t, --threads=int(default 4)\n";
	cout << "-a double(default 1), Beta(a,b) slice parameter\n";
	cout << "-b double(default 5), Beta(a,b) slice parameter\n";
	cout << "--letter_order=int(letter VPYP order, default 8)\n";
	cout << "--surface_order=int(NE surface chunk-PYP order, default 1)\n";
	cout << "--sgamma=double(Beta(1,gamma) gate prior, default 10)\n";
	cout << "--salpha=double(GEM concentration, default 1)\n";
	cout << "--init_random(initialize by sampling the prior instead of all-O seed)\n";
	cout << "--anneal=int(warm-up epochs that damp the NE emission, default 0=off)\n";
	cout << "--tau0=double(initial emission temperature for --anneal, default 2)\n";
	cout << "--prefix=str(snapshot prefix)\n";
	exit(1);
}

int read_long_param(const char *opt, const char *arg) {
	if (check(opt, "train")) train = arg;
	else if (check(opt, "parse")) test = arg;
	else if (check(opt, "prefix")) prefix = arg;
	else if (check(opt, "model")) model = arg;
	else if (check(opt, "wdic")) wdic = arg;
	else if (check(opt, "cdic")) cdic = arg;
	else if (check(opt, "tokenizer")) tokenizer = arg;
	else if (check(opt, "order")) n = atoi(arg);
	else if (check(opt, "letter_order")) hl = atoi(arg);
	else if (check(opt, "surface_order")) hn = atoi(arg);
	else if (check(opt, "class")) { K = atoi(arg); k = min(k, K); }
	else if (check(opt, "epoch")) epoch = atoi(arg);
	else if (check(opt, "threads")) threads = atoi(arg);
	else if (check(opt, "sgamma")) gamma_ = atof(arg);
	else if (check(opt, "salpha")) alpha_ = atof(arg);
	else if (check(opt, "init_random")) init_random = 1;
	else if (check(opt, "anneal")) anneal = atoi(arg);
	else if (check(opt, "tau0")) tau0 = atof(arg);
	return 1;
}

int read_param(int argc, char **argv) {
	if (argc < 2) {
		usage(argc, argv);
		return 1;
	}
	int c;
	while (true) {
		static struct option long_options[] = {
			{"train", required_argument, 0, 0},
			{"parse", required_argument, 0, 0},
			{"prefix", required_argument, 0, 0},
			{"cdic", required_argument, 0, 0},
			{"wdic", required_argument, 0, 0},
			{"model", required_argument, 0, 0},
			{"tokenizer", required_argument, 0, 0},
			{"order", required_argument, 0, 0},
			{"letter_order", required_argument, 0, 0},
			{"surface_order", required_argument, 0, 0},
			{"class", required_argument, 0, 0},
			{"epoch", required_argument, 0, 0},
			{"threads", required_argument, 0, 0},
			{"sgamma", required_argument, 0, 0},
			{"salpha", required_argument, 0, 0},
			{"init_random", no_argument, 0, 0},
			{"anneal", required_argument, 0, 0},
			{"tau0", required_argument, 0, 0},
			{0, 0, 0, 0}
		};
		int option_index = 0;
		c = getopt_long(argc, argv, "n:k:e:t:a:b:", long_options, &option_index);
		if (c == -1)
			break;
		switch (c) {
			case 0:
				if (long_options[option_index].flag != 0)
					break;
				read_long_param(long_options[option_index].name, optarg);
				break;
			case 'n': n = atoi(optarg); break;
			case 'k': K = atoi(optarg); k = min(k, K); break;
			case 'e': epoch = atoi(optarg); break;
			case 't': threads = atoi(optarg); break;
			case 'a': a = atof(optarg); break;
			case 'b': b = atof(optarg); break;
			case '?':
			default:
				usage(argc, argv);
		}
	}
	if (optind < argc) {
		cerr << "non-option ARGV-elements: ";
		while (optind < argc)
			cerr << argv[optind++] << " ";
		cerr << endl;
		usage(argc, argv);
		return 1;
	}
	return 0;
}

// print each chunk as `surface:id:class` (colon-delimited, class 0 = O) so that
// ne_evaluate.py's read_parsed_data (splits on ':', class = field[2]) works.
void dump(nsentence& s) {
	for (auto i = 0; i < s.size(); ++i) {
		chunk& c = s.ch(i);
		for (auto w = 0; w < c.len; ++w) {
			word& wd = c.wd(w);
			for (auto j = 0; j < wd.len; ++j) {
				char buf[5] = {0};
				io::i2c(wd[j], buf);
				cout << buf;
			}
		}
		cout << ":" << c.id << ":" << c.k << "\n";
	}
	cout << "\n";
}

// tokenize raw text with the phsmm tokenizer; assign wid ids to every word so
// the template n-gram over word ids is stable across train/parse (verbatim from
// ne.cc except the pos re-indexing, which sne does not need).
int tokenize(io& f, vector<sentence>& c) {
	shared_ptr<wid> d = wid::create();
	d->load(wdic.c_str());
	if (!toklm) {
		toklm = shared_ptr<phsmm>(new phsmm);
		toklm->load(tokenizer.c_str());
	}
	phsmm& lm = *toklm;
	c.resize(f.head.size()-1);
#ifdef _OPENMP
	threads = min(omp_get_max_threads(), threads);
	omp_set_num_threads(threads);
#pragma omp parallel for ordered schedule(dynamic)
#endif
	for (auto i = 0; i < (int)f.head.size()-1; ++i) {
		c[i] = lm.parse(f, i);
#ifdef _OPENMP
#pragma omp ordered
#endif
		progress("tokenizing", i, (double)(i+1)/(f.head.size()-1));
	}
	for (auto s = c.begin(); s != c.end(); ++s) {
		for (auto j = 0; j < s->size(); ++j) {
			word& w = s->wd(j);
			if (w.id == 1)
				w.id = d->index(w);
		}
	}
	int rpad = 2*PBWIDTH;
	printf("\r%*s", rpad, "");
	d->save(wdic.c_str());
	return 0;
}

// cold start: every word is its own length-1 O chunk (k=0). The sampler then
// re-segments; O is always length 1 so this is a valid lattice-representable path.
int init_corpus(nio& f, vector<nsentence>& corpus) {
	for (auto i = 0; i < (int)f.head.size()-1; ++i) {
		nsentence s;
		int head = f.head[i];
		int tail = f.head[i+1];
		for (auto w = head; w < tail; ++w) {
			chunk c(*f.raw, w, 1);
			c.k = 0;
			s.c.emplace_back(c);
		}
		s.n.resize(s.c.size()+1, 0);
		corpus.emplace_back(s);
	}
	return 0;
}

int snapshot(snpylm& lm, int iter) {
	char mfile[512] = {};
	char dfile[512] = {};
	snprintf(mfile, sizeof(mfile), "%s_iter%03d.model", prefix.c_str(), iter);
	snprintf(dfile, sizeof(dfile), "%s_iter%03d.dic", prefix.c_str(), iter);
	lm.save(mfile);
	shared_ptr<cid> d = cid::create();
	d->save(dfile);
	return 0;
}

// per-epoch degeneracy diagnostic: NE-chunk rate over the whole corpus, printed
// alongside snpylm::stats() so a collapse to all-O is caught early.
void epoch_stats(snpylm& lm, vector<nsentence>& corpus, int i) {
	long long nchunk = 0, nne = 0;
	for (auto& s : corpus) {
		for (auto j = 0; j < s.size(); ++j) {
			++nchunk;
			if (s.ch(j).k >= 1)
				++nne;
		}
	}
	fprintf(stderr, "[epoch %d] chunks=%lld ne=%lld ne_rate=%.4f\n",
			i, nchunk, nne, nchunk ? (double)nne/nchunk : 0.0);
	lm.stats();
}

int mcmc(nio& f, vector<nsentence>& corpus) {
	snpylm lm(n, hn, hl, K);
	lm.slice(a, b);
	lm.set_gamma(gamma_);
	lm.set_alpha(alpha_);
	static const bool stat = (getenv("NPBNLP_SNPYLM_STATS") != NULL);
#ifdef _OPENMP
	omp_set_num_threads(threads);
#endif
	// all-O seed (default): add the initial all-O segmentation so the background
	// LM starts rich and NE must be *earned* by a clearly cheaper H_k surface.
	// --init_random skips the seed and samples the prior at epoch 0 instead.
	bool seeded = false;
	if (!init_random) {
		for (auto i = 0; i < (int)corpus.size(); ++i)
			lm.add(corpus[i]);
		lm.estimate(20);
		seeded = true;
	}
	for (auto i = 0; i < epoch; ++i) {
		// annealing warm-up: emission temperature tau0 -> 1 over `anneal` epochs.
		if (anneal > 0)
			lm.set_temp((i < anneal) ? (tau0 - (tau0-1.0)*i/anneal) : 1.0);
		bool do_remove = seeded || i > 0;
		vector<int> rd(corpus.size(), 0);
		rd::shuffle(rd.data(), corpus.size());
		int j = 0;
		while (j < (int)corpus.size()) {
			if (do_remove) {
				for (auto t = 0; t < threads; ++t) {
					if (j+t < (int)corpus.size())
						lm.remove(corpus[rd[j+t]]);
				}
			}
#ifdef _OPENMP
#pragma omp parallel
			{
				auto t = omp_get_thread_num();
				if (j+t < (int)corpus.size()) {
					try {
						nsentence s = lm.sample(f, rd[j+t], &corpus[rd[j+t]]);
						corpus[rd[j+t]] = s;
					} catch (const char *ex) {
						cerr << ex << endl;
					}
				}
			}
#else
			for (auto t = 0; t < threads; ++t) {
				if (j+t < (int)corpus.size()) {
					try {
						nsentence s = lm.sample(f, rd[j+t], &corpus[rd[j+t]]);
						corpus[rd[j+t]] = s;
					} catch (const char *ex) {
						cerr << ex << endl;
					}
				}
			}
#endif
			for (auto t = 0; t < threads; ++t) {
				if (j+t < (int)corpus.size())
					lm.add(corpus[rd[j+t]]);
			}
			j += threads;
			progress("epoch", i, (double)(j+1)/corpus.size());
		}
		lm.estimate(20);
		if (stat)
			epoch_stats(lm, corpus, i);
		if ((i+1)%snap == 0)
			snapshot(lm, i);
	}
	cout << endl;
	lm.save(model.c_str());
	shared_ptr<cid> d = cid::create();
	d->save(cdic.c_str());
	return 0;
}

int parse(nio& f) {
	shared_ptr<cid> d = cid::create();
	d->load(cdic.c_str());
	snpylm lm;
	try {
		lm.load(model.c_str());
		lm.slice(a, b);
	} catch (const char *ex) {
		throw ex;
	}
#ifdef _OPENMP
	omp_set_num_threads(threads);
#pragma omp parallel for ordered schedule(dynamic)
#endif
	for (auto i = 0; i < (int)f.head.size()-1; ++i) {
		nsentence s = lm.parse(f, i);
#ifdef _OPENMP
#pragma omp ordered
#endif
		dump(s);
	}
	return 0;
}

int main(int argc, char **argv) {
	try {
		read_param(argc, argv);
		if (!train.empty()) {
			io g(train.c_str());
			vector<sentence> ws;
			tokenize(g, ws);
			nio f(ws);
			vector<nsentence> corpus;
			init_corpus(f, corpus);
			mcmc(f, corpus);
		}
		if (!test.empty()) {
			io g(test.c_str());
			vector<sentence> ws;
			tokenize(g, ws);
			nio f(ws);
			parse(f);
		}
	} catch (const char *ex) {
		cerr << ex << endl;
		return 1;
	}
	return 0;
}
