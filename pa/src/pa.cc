#include"ipcfg.h"
#include"rd.h"
#include"util.h"
#include<cstdio>
#include<cstring>
#include<cstdlib>
#include<getopt.h>
#include<unordered_set>
#include<map>
#include<cmath>
#include<limits>
#ifdef _OPENMP
#include<omp.h>
#endif

#define check(opt,arg) (strcmp(opt,arg) == 0)
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60
#define NPYLM_EPOCH 100

using namespace std;
using namespace npbnlp;

static int m = 20;
static int k = 20;
static int K = 100; // base measure for transition
static int threads = 4;
static int epoch = 500;
static int dmp = 0;
static double a = 1;
static double b = 1;
static int span = 0;
static double span_a = 1;
static double span_b = 1;
static int split = 0;
static int split_fixed = 0;
static double split_a = 1;
static double split_b = 1;
static double split_q = .5;
static int wngram = 2;
static int dot = 0;
static int mh = 0;
static int random_seed = -1;
static int ckpt = 0;
static string train;
static string test;
static string model("ipcfg.model");
static string dic("pa.dic");
//static int node_id = 0;

class dot_node {
	public:
		int id;
		int k;
		int left;
		int right;
		string label;
};

void progress(const char *s, int i, double pct) {
	int val = (int) (pct * 100);
	int lpad = (int) (pct * PBWIDTH);
	int rpad = PBWIDTH - lpad;
	printf("\r%s %4d %3d%% [%.*s%*s]", s, i, val, lpad, PBSTR, rpad, "");
	fflush(stdout);
}

void usage(int argc, char **argv) {
	cout << "[Usage]" << *argv << " [options]\n";
	cout << "[example]\n";
	cout << *argv << " --train file --model file_to_save --dic dicfile\n";
	cout << *argv << " --parse file --model modelfile --dic dicfile\n";
	cout << "[options]\n";
	cout << "-m, --letter_order=int(default 20)\n";
	cout << "-k, --max_category=int(default 100)\n";
	cout << "-e, --epoch=int(default 500)\n";
	cout << "-t, --threads=int(default 4)\n";
	cout << "--span=flag enable a Beta-geometric prior over non-root internal spans\n";
	cout << "--span_alpha=float prior alpha for span stop probability(default 1)\n";
	cout << "--span_beta=float prior beta for span continue probability(default 1)\n";
	cout << "--split=flag enable a truncated geometric prior over split points (root included)\n";
	cout << "--split_alpha=float Beta(a,b) prior alpha for the split probability q(default 1)\n";
	cout << "--split_beta=float Beta(a,b) prior beta for the split probability q(default 1)\n";
	cout << "  larger q concentrates on L=1 (a one-word left child) = right branching,\n";
	cout << "  so use alpha>beta for right-branching languages (English), beta>alpha for\n";
	cout << "  left-branching ones (Japanese); the default 1/1 is uninformative and lets\n";
	cout << "  the branching bias be learned from data.\n";
	cout << "--split_q_init=float initial value of q for the sampler(default 0.5); this is a\n";
	cout << "  starting point, not prior knowledge -- inject knowledge via --split_alpha/--split_beta\n";
	cout << "--split_fixed=flag hold q at --split_q_init instead of learning it (ablation)\n";
	cout << "--wngram=int order of the pre-terminal word emission P(w_i|w_{i-1}..,A)(default 2);\n";
	cout << "  1 is the historical unigram emission P(w_i|A). Ignored with a warning when the\n";
	cout << "  loaded model was trained at another order\n";
	cout << "Parent labels are constrained to be no greater than a child label.\n";
	cout << "--mh=flag use slice-CYK proposals with a sequential-HPYP MH correction (threads=1)\n";
	cout << "--seed=int use a deterministic random seed for reproducible experiments\n";
	cout << "--ckpt=int    save model/dic every N epochs (0=only at the end)\n";
	cout << "--dot=flag output in dot format for graphviz" << endl;
	exit(1);
}

int read_long_param(const char *opt, const char *arg) {
	if (check(opt, "train")) {
		train = arg;
	} else if (check(opt, "parse")) {
		test = arg;
	} else if (check(opt, "model")) {
		model = arg;
	} else if (check(opt, "dic")) {
		dic = arg;
	} else if (check(opt, "letter_order")) {
		m = atoi(arg);
	} else if (check(opt, "max_category")) {
		K = atoi(arg);
		k = min(k, K);
	} else if (check(opt, "epoch")) {
		epoch = atoi(arg);
	} else if (check(opt, "threads")) {
		threads = atoi(arg);
	} else if (check(opt, "dump")) {
		dmp = atoi(arg);
	} else if (check(opt, "dot")) {
		dot = 1;
	} else if (check(opt, "span_alpha")) {
		span_a = atof(arg);
	} else if (check(opt, "span_beta")) {
		span_b = atof(arg);
	} else if (check(opt, "split_alpha")) {
		split_a = atof(arg);
	} else if (check(opt, "split_beta")) {
		split_b = atof(arg);
	} else if (check(opt, "split_q_init")) {
		split_q = atof(arg);
	} else if (check(opt, "wngram")) {
		wngram = atoi(arg);
	} else if (check(opt, "seed")) {
		random_seed = atoi(arg);
	} else if (check(opt, "ckpt")) {
		ckpt = atoi(arg);
	} else {
		return 1;
	}
	return 1;
}

int read_param(int argc, char **argv) {
	if (argc < 2) {
		usage(argc, argv);
		return 1;
	}
	int c;
	while (1) {
		static struct option long_options[] =
		{
			{"train", required_argument, 0, 0},
			{"parse", required_argument, 0, 0},
			{"dic", required_argument, 0, 0},
			{"model", required_argument, 0, 0},
			{"letter_order", required_argument, 0, 0},
			{"max_category", required_argument, 0, 0},
			{"epoch", required_argument, 0, 0},
			{"threads", required_argument, 0, 0},
			{"dump", required_argument, 0, 0},
			{"span", no_argument, &span, 1},
			{"span_alpha", required_argument, 0, 0},
			{"span_beta", required_argument, 0, 0},
			{"split", no_argument, &split, 1},
			{"split_alpha", required_argument, 0, 0},
			{"split_beta", required_argument, 0, 0},
			{"split_q_init", required_argument, 0, 0},
			{"split_fixed", no_argument, &split_fixed, 1},
			{"wngram", required_argument, 0, 0},
			{"mh", no_argument, &mh, 1},
			{"seed", required_argument, 0, 0},
			{"ckpt", required_argument, 0, 0},
			// flag option
			{"dot", no_argument, &dot, 1},
			{0, 0, 0, 0}
		};
		int option_index = 0;
		c = getopt_long(argc, argv, "m:k:e:t:", long_options, &option_index);
		if (c == -1)
			break;
		switch (c) {
			case 0:
				if (long_options[option_index].flag != 0)
					break;
				read_long_param(long_options[option_index].name, optarg);
				break;
			case 'm':
				m = atoi(optarg);
				break;
			case 'k':
				K = atoi(optarg);
				k = min(k, K);
				break;
			case 'e':
				epoch = atoi(optarg);
				break;
			case 't':
				threads = atoi(optarg);
				break;
			case '?':
			default:
				usage(argc, argv);
		}
	}
	if (optind < argc) {
		cerr << "non-option ARGV-elements: ";
		while (optind < argc) {
			cerr << argv[optind++] << " ";
		}
		cerr << endl;
		usage(argc, argv);
		return 1;
	}
	return 0;
}

void dump_node(tree& t, int i) {
	node& c = t[i];
	if (c.i != c.j) {
		cout << "(";
		int left = t.s.size()*c.i+c.b-c.i*(1.+c.i)/2;
		int right = t.s.size()*(c.b+1)+c.j-(1.+c.b)*(2+c.b)/2;
		cout << c.k << " ";
		dump_node(t, left);
		dump_node(t, right);
		cout << ")";
	} else if (c.k > 0 && c.i == c.j) {
		cout << "(" << c.k << " " << t.wd(c.i) << ")";
	}
}

void tree_node(tree& t, int i, vector<dot_node>& n) {
	node& c = t[i];
	if (c.i != c.j) {
		int left = t.s.size()*c.i+c.b-c.i*(1.+c.i)/2;
		int right = t.s.size()*(c.b+1)+c.j-(1.+c.b)*(2+c.b)/2;
		dot_node nd;
		nd.id = i;
		nd.k = c.k;
		nd.left = left;
		nd.right = right;
		char buf[1024] = {0};
		sprintf(buf, "%d:", c.k);
		nd.label = buf;
		for (auto i = c.i; i <= c.j; ++i) {
			word& w = t.wd(i);
			for (auto l = 0; l < w.len; ++l) {
				char wbuf[5] = {0};
				io::i2c(w[l], wbuf);
				nd.label += wbuf;
			}
			nd.label += " ";
		}
		n.push_back(nd);
		tree_node(t, left, n);
		tree_node(t, right, n);
	} else if (c.k > 0 && c.i == c.j) {
		dot_node nd;
		nd.id = i;
		nd.k = c.k;
		nd.left = -1;
		nd.right = -1;
		word& w = t.wd(c.i);
		char k[1024] = {0};
		sprintf(k, "%d:", c.k);
		nd.label = k;
		for (auto i = 0; i < w.len; ++i) {
			char buf[5] = {0};
			io::i2c(w[i], buf);
			nd.label += buf;
		}
		n.push_back(nd);
	}
}

void dump_dot(tree& t, int n) {
	string str;
	for (auto i = 0; i < t.s.size(); ++i) {
		word& w = t.s.wd(i);
		for (auto j = 0; j < w.len; ++j) {
			char buf[5] = {0};
			io::i2c(w[j], buf);
			if (strcmp(buf, "#") == 0)
				str += "\\";
			str += buf;
		}
		str += " ";
	}
	vector<dot_node> nodes;
	//cout << "digraph {" << endl;
	//cout << "node [fontname=IPAPGothic]" << endl;
	//cout << "edge [fontname=IPAPGothic]" << endl;
	cout << "subgraph cluster_" << n << "{" << endl;
	cout << "label=\"" << str << "\""<< endl;
	tree_node(t, t.s.size()-1, nodes);
	// label
	for (auto i = nodes.begin(); i != nodes.end(); ++i) {
		if (i->label[i->label.size()-1] == '\\') {
			i->label += "\\";
		}
		cout << "n" << n << "_" << i->id << " [label=\"" << i->label << "\"]" << endl;
	}
	// edge
	for (auto i = nodes.begin(); i != nodes.end(); ++i) {
		if (i->left >= 0)
			cout << "n" << n << "_" << i->id << " -> " << "n" << n << "_" << i->left << endl;
		if (i->right >= 0)
			cout << "n" << n << "_" << i->id << " -> " << "n" << n << "_" << i->right << endl;
	}
	cout << "}" << endl;
	//cout << "}" << endl;
}

void dump(tree& t, int n) {
	if (dot) {
		dump_dot(t, n);
	} else {
		dump_node(t, t.s.size()-1);
		cout << endl;
	}
}

void dump_all(vector<tree>& corpus) {
	if (dot) {
		cout << "digraph {" << endl;
		cout << "node [fontname=IPAPGothic]" << endl;
		cout << "edge [fontname=IPAPGothic]" << endl;
	}
	for (auto i = 0; i < (int)corpus.size(); ++i) {
		dump(corpus[i], i);
	}
	if (dot)
		cout << "}" << endl;
}

void assign_balanced_labels(tree& tr, int index, int categories, long long& cursor) {
	node& z = tr[index];
	if (z.k < 0)
		return;
	if (z.k != 0)
		z.k = 1+(cursor++ % categories);
	if (z.i == z.j)
		return;
	int size = tr.s.size();
	assign_balanced_labels(tr, size*z.i+z.b-z.i*(1.+z.i)/2, categories, cursor);
	assign_balanced_labels(tr, size*(z.b+1)+z.j-(z.b+1.)*(z.b+2)/2, categories, cursor);
}

void report_epoch_diagnostics(int iter, vector<tree>& corpus, ipcfg& g) {
	map<int, long long> internal, terminal, top_child;
	long long ni = 0, nt = 0, nh = 0;
	for (auto& tr : corpus) {
		for (const auto& z : tr.c) {
			if (z.k <= 0)
				continue;
			if (z.i == z.j) {
				++terminal[z.k]; ++nt;
			} else {
				++internal[z.k]; ++ni;
			}
		}
		if (tr.s.size() > 1) {
			const node& root = tr[tr.s.size()-1];
			int n = tr.s.size();
			const node& left = tr[n*root.i+root.b-root.i*(1.+root.i)/2];
			const node& right = tr[n*(root.b+1)+root.j-(root.b+1.)*(root.b+2)/2];
			if (left.k > 0) { ++top_child[left.k]; ++nh; }
			if (right.k > 0) { ++top_child[right.k]; ++nh; }
		}
	}
	auto summarize = [](const map<int, long long>& h, long long n, int& active,
						 double& entropy, double& max_share, int& winner) {
		active = 0; entropy = 0.; max_share = 0.; winner = 0;
		for (const auto& p : h) {
			if (p.second == 0) continue;
			++active;
			double q = (double)p.second/n;
			entropy -= q*log(q);
			if (q > max_share) { max_share = q; winner = p.first; }
		}
	};
	int ai, at, ah, wi, wt, wh;
	double hi, ht, hh, mi, mt, mh;
	summarize(internal, ni, ai, hi, mi, wi);
	summarize(terminal, nt, at, ht, mt, wt);
	summarize(top_child, nh, ah, hh, mh, wh);
	long long sc_t, sl_t, sc_i, sl_i;
	g.slice_diagnostics(sc_t, sl_t, sc_i, sl_i);
	cerr << "[ipcfg-diag] epoch=" << (iter+1)
		 << " k=" << g.category_count()
		 << " internal_n=" << ni << " internal_active=" << ai
		 << " internal_H=" << hi << " internal_max=" << mi << " internal_argmax=" << wi
		 << " preterm_n=" << nt << " preterm_active=" << at
		 << " preterm_H=" << ht << " preterm_max=" << mt << " preterm_argmax=" << wt
		 << " top_child_n=" << nh << " top_child_active=" << ah
		 << " top_child_H=" << hh << " top_child_max=" << mh << " top_child_argmax=" << wh
		 << " slice_preterm_mean=" << (sc_t ? (double)sl_t/sc_t : 0.)
		 << " slice_internal_mean=" << (sc_i ? (double)sl_i/sc_i : 0.) << endl;
	// Pre-terminal failure mode for the word n-gram emission: when the mean
	// number of surviving labels per terminal cell falls to 1, the word alone
	// determines its class and the grammar has nothing left to decide.
	cerr << "[preterm] epoch=" << (iter+1)
		 << " wngram=" << g.word_ngram()
		 << " slice_terminal_labels=" << sl_t
		 << " slice_terminal_cells=" << sc_t
		 << " labels_per_cell=" << (sc_t ? (double)sl_t/sc_t : 0.) << endl;
}

int mcmc() {
	io f(train.c_str());
	vector<tree> corpus;
	corpus.resize(f.head.size()-1);
	ipcfg g(m);
	g.set(K);
	// Must precede every add(): word_ngram() rebuilds the class HPYPs.
	g.word_ngram(wngram);
	g.slice(a, b);
	if (span)
		g.span(span_a, span_b);
	if (split)
		g.split(split_a, split_b, split_q, split_fixed != 0);
	if (mh && threads != 1) {
		cerr << "[ipcfg-mh] forcing --threads 1: proposal scoring and snapshot restoration are sequential" << endl;
		threads = 1;
	}
#ifdef _OPENMP
	threads = min(omp_get_max_threads(), threads);
	omp_set_num_threads(threads);
#endif
	// Model-based initialization is intentionally sequential: each generated
	// production immediately changes the HPYP used for the next one.  It does
	// not call the slice sampler.  The following MCMC sweep always starts by
	// removing these trees, as in ma/src/ma.cc.
	vector<int> init_rd(corpus.size(), 0);
	rd::shuffle(init_rd.data(), corpus.size());
	for (int j = 0; j < (int)corpus.size(); ++j)
		corpus[init_rd[j]] = g.init(f, init_rd[j]);
	g.estimate(20);
	g.poisson_correction(1000);
	report_epoch_diagnostics(-1, corpus, g);
	{
		// Deeper contexts route fewer customers to the root, so fewer tokens
		// reach the base measure whose witnesses the Poisson correction is
		// estimated from.  Reported once, right after initialization.
		vector<long long> bc;
		g.base_corpus_sizes(bc);
		cerr << "[preterm] base_corpus";
		for (int c = 0; c < (int)bc.size(); ++c)
			cerr << " " << c << ":" << bc[c];
		cerr << endl;
	}
	for (auto i = 0; i < epoch; ++i) {
		long long mh_attempts = 0;
		long long mh_accepts = 0;
		double mh_log_alpha_sum = 0.;
		double mh_log_alpha_min = numeric_limits<double>::infinity();
		double mh_log_alpha_max = -numeric_limits<double>::infinity();
		vector<int> rd(corpus.size(), 0);
		//int rd[corpus.size()] = {0};
		rd::shuffle(rd.data(), corpus.size());
		int j = 0;
		while (j < (int)corpus.size()) {
			// remove
			for (auto t = 0; t < threads; ++t) {
				if (j+t < (int)corpus.size()) {
					g.remove(corpus[rd[j+t]]);
				}
			}
			if (mh) {
				int id = rd[j];
				// The old tree has already been removed.  Snapshot this common
				// M_- state before either sequential target evaluation mutates it.
				unique_ptr<ipcfg> base = g.snapshot();
				double log_q_new = 0.;
				double log_q_old = 0.;
				// Both q values are evaluated on this one current-tree-conditioned
				// slice lattice: root inside supplies its normalizer and traceback
				// supplies the selected derivation probability.
				tree proposal = g.mh_propose(f, id, &corpus[id], log_q_new, log_q_old);
				double log_p_old = base->mh_logprob_and_add(corpus[id]);
				double log_p_new = g.mh_logprob_and_add(proposal);
				double log_alpha = log_p_new-log_p_old+log_q_old-log_q_new;
				uniform u;
				bool accept = log_alpha >= 0. || log(u(0., 1.)) < log_alpha;
				++mh_attempts;
				mh_log_alpha_sum += log_alpha;
				mh_log_alpha_min = min(mh_log_alpha_min, log_alpha);
				mh_log_alpha_max = max(mh_log_alpha_max, log_alpha);
				if (accept) {
					++mh_accepts;
					corpus[id] = proposal;
				} else {
					// The tentative tree was added to g only to evaluate the
					// sequential target.  Remove that exact seating and re-add the
					// retained tree; this is the ordinary HPYP seating update for a
					// held-out sentence and preserves every base-corpus witness.
					g.remove(proposal);
					g.add(corpus[id]);
				}
			} else {
			#ifdef _OPENMP
#pragma omp parallel
			{ // sample segmentations
				auto t = omp_get_thread_num();
				if (j+t < (int)corpus.size()) {
					try {
						// pass the current tree so the slice variable is
						// conditioned on it (WO-005). corpus[rd] is only
						// overwritten after sample() returns, so the current
						// assignment stays alive throughout sampling.
						tree tr = g.sample(f, rd[j+t], &corpus[rd[j+t]]);
						//dump(tr, rd[j+t]);
						corpus[rd[j+t]] = tr;
					} catch (const char *ex) {
						cerr << ex << endl;
					}
				}
			}
#else
			for (auto t = 0; t < threads; ++t) {
				if (j+t < (int)corpus.size()) {
					try {
						tree tr = g.sample(f, rd[j+t], &corpus[rd[j+t]]);
						corpus[rd[j+t]] = tr;
					} catch (const char *ex) {
						cerr << ex << endl;
					}
				}
			}
			#endif
			}
			// add
			if (!mh) {
				for (auto t = 0; t < threads; ++t) {
					if (j+t < (int)corpus.size()) {
						g.add(corpus[rd[j+t]]);
					}
				}
			}
			j += threads;
#ifdef _OPENMP
#pragma omp ordered
#endif
			progress("epoch", i, (double)(j+1)/corpus.size());
		}
		// All trees are in the model again.  Only now is it safe to discard
		// trailing empty latent categories.
		g.compact();
		// estimate hyperparameter
		g.estimate(20);
		g.poisson_correction(1000);
		report_epoch_diagnostics(i, corpus, g);
		if (mh)
			cerr << "[ipcfg-mh] epoch=" << (i+1) << " accept=" << mh_accepts
				 << "/" << mh_attempts << " rate="
				 << (mh_attempts ? (double)mh_accepts/mh_attempts : 0.)
				 << " log_alpha_mean=" << (mh_attempts ? mh_log_alpha_sum/mh_attempts : 0.)
				 << " min=" << (mh_attempts ? mh_log_alpha_min : 0.)
				 << " max=" << (mh_attempts ? mh_log_alpha_max : 0.) << endl;
		if (ckpt > 0 && (i+1) % ckpt == 0) {
			// crash-safe: write to a temp path, then rename over the target
			string mtmp = model+".tmp";
			string dtmp = dic+".tmp";
			g.save(mtmp.c_str());
			shared_ptr<wid> cd = wid::create();
			cd->save(dtmp.c_str());
			if (rename(mtmp.c_str(), model.c_str()) != 0 ||
			    rename(dtmp.c_str(), dic.c_str()) != 0)
				throw "failed to rename checkpoint in ipcfg train";
			cerr << "[ipcfg-ckpt] epoch=" << (i+1) << " saved" << endl;
		}
		if (dmp && (i+1)%dmp == 0) {
			cout << endl;
			//for (auto s = corpus.begin(); s != corpus.end(); ++s)
			/*
			   for (auto s = 0; s < corpus.size(); ++s)
			   dump(corpus[s], s);
			   */
			dump_all(corpus);
		}
	}
	cout << endl;
	g.save(model.c_str());
	shared_ptr<wid> d = wid::create();
	d->save(dic.c_str());
	return 0;
}

int parse() {
	io f(test.c_str());
	shared_ptr<wid> d = wid::create();
	d->load(dic.c_str());
	ipcfg g(m);
	try {
		// Record the requested order so load() can report a conflict; the
		// order stored in the model always wins (see ipcfg::_load).
		g.word_ngram(wngram);
		g.load(model.c_str());
		// load() restores the training alphabet size.  Do not overwrite it
		// with the command-line default during inference.
	} catch (const char *ex) {
		throw ex;
	}
	if (dot) {
		cout << "digraph {" << endl;
		cout << "node [fontname=IPAPGothic]" << endl;
		cout << "edge [fontname=IPAPGothic]" << endl;
	}
	// parse() samples, so the dynamic schedule below makes the order in which
	// sentences consume the RNG depend on thread timing.  --seed promises a
	// reproducible run, so honour it the way mcmc() honours --mh.
	if (random_seed >= 0 && threads != 1) {
		cerr << "[ipcfg] forcing --threads 1: --seed requires a deterministic parse order" << endl;
		threads = 1;
	}
#ifdef _OPENMP
	threads = min(omp_get_max_threads(), threads);
	omp_set_num_threads(threads);
#pragma omp parallel for ordered schedule(dynamic)
#endif
	for (auto i = 0; i < (int)f.head.size()-1; ++i) {
		tree t = g.parse(f, i);
#ifdef _OPENMP
#pragma omp ordered
#endif
		dump(t, i);
	}
	if (dot)
		cout << "}" << endl;
	return 0;
}

int main(int argc, char **argv) {
	try {
		read_param(argc, argv);
		if (random_seed >= 0)
			seed::set((unsigned int)random_seed);
		if (!train.empty()) {
			mcmc();
		}
		if (!test.empty()) {
			parse();
		}
	} catch (const char *ex) {
		cerr << ex << endl;
		return 1;
	}
	return 0;
}
