#include"nnpylm.h"
#include"nphsmm.h"
#include"npylm.h"
#include"phsmm.h"
#include"ihmm.h"
#include"rd.h"
#include"util.h"
#include"cio.h"
#include<chrono>
#include"chunktype.h"
#include<getopt.h>
#ifdef _OPENMP
#include<omp.h>
#endif

#define check(opt,arg) (strcmp(opt,arg) == 0)
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60
#define NNPYLM_EPOCH 0

using namespace std;
using namespace npbnlp;

static int n = 1;
static int m = 3;
static int l = 20;
static int k = 10;
static int K = 50;
static int threads = 4;
static int epoch = 500;
static int pre_epoch = 20;
static int snap = 5;
static int dmp = 0;
//static int tokenized = 0;
static int vocab = 0;
static int original = 0;
static int posbase = 0;
static int sweight = 1;
static int ctx = 0;
static int ctxgate = 0;
static int wclass = 0;
static double wbeta = 1.0;
static std::shared_ptr<phsmm> toklm; // tokenizer kept alive for the lexical fill-in of the pos base
static double a = 1;
static double b = 5;
static string prefix("nphsmm");
static string pretrain;
static string train;
static string test;
static string model("nphsmm.model");
static string cdic("ne.dic");
static string tokenizer("phsmm.model");
static string wdic("ma.dic");

static unordered_map<string, int> pos_index;
static int pos_id = 1;
static unordered_map<string, int> label_index;
static int label_id = 2;

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
	cout << *argv << " --train file --tokenizer phsmm.model --wdic ma.dic --model file_to_save --cdic chunk.dic\n";
	cout << *argv << " --parse file --tokenizer phsmm.model --wdic ma.dic --model modelfile --cdic chunk.dic\n";
	cout << "[options]\n";
	cout << "-n, --chunk_order=int(default 1)\n";
	cout << "-m, --word_order=int(default 3)\n";
	cout << "-l, --letter_order=int(default 20)\n";
	cout << "-k, --class=int(default 50)\n";
	cout << "-e, --epoch=int(default 500)\n";
	cout << "-t, --threads=int(default 4)\n";
	cout << "-v, --vocab=int(means letter variations. default 0: train from data)\n";
	//cout << "--tokenized=bool(default 0)\n";
	cout << "-a double(default 1), parameter of beta distribution for slice\n";
	cout << "-b double(default 5), parameter of beta distribution for slice\n";
	cout << "--pretrain=file(use as pretraining dataset in training\n";
	cout << "--prefix=str(use as prefix for saving snapshot of model\n";
	cout << "--original(use original forward computation)\n";
	cout << "--posbase(use pos-pattern base measure for chunk emission)\n";
	cout << "--sweight=int(weight of supervised data. default 1)\n";
	cout << "--ctx=int(context-distribution factor window radius. default 0=off)\n";
	cout << "--ctxgate(class-normalize the context factor as a softmax gate; needs --ctx)\n";
	cout << "--wclass(per-word tokenizer-class channel: chunk class emits word.pos via theta)\n";
	cout << "--wbeta=float(temperature of the per-word class channel, default 1.0)\n";
	exit(1);
}

int read_long_param(const char *opt, const char *arg) {
	if (check(opt, "train")) {
		train = arg;
	} else if (check(opt, "pretrain")) {
		pretrain = arg;
	} else if (check(opt, "prefix")) {
		prefix = arg;
	} else if (check(opt, "parse")) {
		test = arg;
	} else if (check(opt, "model")) {
		model = arg;
	} else if (check(opt, "wdic")) {
		wdic = arg;
	} else if (check(opt, "cdic")) {
		cdic = arg;
	} else if (check(opt, "tokenizer")) {
		tokenizer = arg;
	} else if (check(opt, "chunk_order")) {
		n = atoi(arg);
	} else if (check(opt, "word_order")) {
		m = atoi(arg);
	} else if (check(opt, "letter_order")) {
		l = atoi(arg);
	} else if (check(opt, "class")) {
		K = atoi(arg);
		k = min(k, K);
	} else if (check(opt, "epoch")) {
		epoch = atoi(arg);
	} else if (check(opt, "threads")) {
		threads = atoi(arg);
	} else if (check(opt, "dump")) {
		dmp = atoi(arg);
	} else if (check(opt, "vocab")) {
		vocab = atoi(arg);
	} else if (check(opt, "original")) {
		original = 1;
	} else if (check(opt, "posbase")) {
		posbase = 1;
	} else if (check(opt, "sweight")) {
		sweight = atoi(arg);
	} else if (check(opt, "ctx")) {
		ctx = atoi(arg);
	} else if (check(opt, "ctxgate")) {
		ctxgate = 1;
	} else if (check(opt, "wclass")) {
		wclass = 1;
	} else if (check(opt, "wbeta")) {
		wbeta = atof(arg);
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
			{"pretrain", required_argument, 0, 0},
			{"prefix", required_argument, 0, 0},
			{"cdic", required_argument, 0, 0},
			{"wdic", required_argument, 0, 0},
			{"model", required_argument, 0, 0},
			{"tokenizer", required_argument, 0,0},
			{"chunk_order", required_argument, 0, 0},
			{"word_order", required_argument, 0, 0},
			{"letter_order", required_argument, 0, 0},
			{"class", required_argument, 0, 0},
			{"epoch", required_argument, 0, 0},
			{"threads", required_argument, 0, 0},
			{"dump", required_argument, 0, 0},
			{"vocab", required_argument, 0, 0},
			{"original", no_argument, 0, 0},
			{"posbase", no_argument, 0, 0},
			{"sweight", required_argument, 0, 0},
			{"ctx", required_argument, 0, 0},
			{"ctxgate", no_argument, 0, 0},
			{"wclass", no_argument, 0, 0},
			{"wbeta", required_argument, 0, 0},
			//{"tokenized", no_argument, &tokenized, 1},
			{0, 0, 0, 0}
		};
		int option_index = 0;
		c = getopt_long(argc, argv, "n:m:l:k:e:t:v:a:b:", long_options, &option_index);
		if (c == -1)
			break;
		switch (c) {
			case 0:
				if (long_options[option_index].flag != 0)
					break;
				read_long_param(long_options[option_index].name, optarg);
				break;
			case 'n':
				n = atoi(optarg);
				break;
			case 'm':
				m = atoi(optarg);
				break;
			case 'l':
				l = atoi(optarg);
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
			case 'v':
				vocab = atoi(optarg);
				break;
			case 'a':
				a = atof(optarg);
				break;
			case 'b':
				b = atof(optarg);
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

void dump(nsentence& s) {
	for (auto i = 0; i < s.size(); ++i)
		cout << s.ch(i) << endl;
	cout << endl;
}

int tokenize(io& f, vector<sentence>& c) {
	shared_ptr<wid> d = wid::create();
	d->load(wdic.c_str());
	//npylm lm;
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
		progress("tokenizing",i,(double)(i+1)/(f.head.size()-1));
	}
	// indexing
	for (auto s = c.begin(); s != c.end(); ++s) {
		for (auto j = 0; j < s->size(); ++j) {
			word& w = s->wd(j);
			if (w.id == 1) {
				w.id = d->index(w);
			}
		}
	}
	int rpad = 2*PBWIDTH;
	printf("\r%*s", rpad,"");
	d->save(wdic.c_str());
	return 0;
}

int nphsmm_pretrain(nphsmm& lm, vector<nsentence>& corpus) {
	int n = corpus.size();
	for (auto i = 0; i < pre_epoch; ++i) {
		vector<int> rd(n, 0);
		rd::shuffle(rd.data(), n);
		for (auto j = 0; j < n; ++j) {
			if (i > 0)
				lm.remove(corpus[rd[j]]);
			lm.add(corpus[rd[j]]);
		}
		lm.estimate(20);
		progress("pretrain nphsmm", i, (double)(i+1)/pre_epoch);
	}
	int rpad = 2*PBWIDTH;
	printf("\r%*s", rpad,"");
	return 0;
}

int nnpy_pretrain(nnpylm& lm, vector<nsentence>& corpus) {
	// pretraining
	int n = corpus.size();
	for (auto i = 0; i < pre_epoch; ++i) {
		vector<int> rd(n, 0);
		rd::shuffle(rd.data(), n);
		for (auto j = 0; j < n; ++j) {
			if (i > 0)
				lm.remove(corpus[rd[j]]);
			lm.add(corpus[rd[j]]);
		}
		lm.estimate(20);
		progress("pretrain nnpylm", i, (double)(i+1)/pre_epoch);
	}
	int rpad = 2*PBWIDTH;
	printf("\r%*s", rpad,"");
	return 0;
}

int init(nio& f, vector<nsentence>& corpus, vector<nsentence>& supervised) {
	nnpylm chunker(n, m, l);
	/*
	if (corpus.size() < f.head.size()-1) {
		corpus.resize(f.head.size()-1);
	}
	*/
	if (!supervised.empty())
		nnpy_pretrain(chunker, supervised);
#ifdef _OPENMP
	omp_set_num_threads(threads);
#endif
	for (int i = 0; i < NNPYLM_EPOCH; ++i) {
		vector<int> rd(corpus.size(), 0);
		//int rd[corpus.size()] = {0};
		rd::shuffle(rd.data(), corpus.size());
		int j = 0;
		while (j < (int)corpus.size()) {
			if (i > 0) {
				for (auto t = 0; t < threads; ++t) {
					if (j+t < (int)corpus.size())
						chunker.remove(corpus[rd[j+t]]);
				}
			}
#ifdef _OPENMP
#pragma omp parallel
			{
				if (i > 0) {
					auto t = omp_get_thread_num();
					if (j+t < (int)corpus.size()) {
						try {
							nsentence s  = chunker.sample(f, rd[j+t]);
							corpus[rd[j+t]] = s;
						} catch (const char *ex) {
							cerr << ex << endl;
						}
					}
				}
			}
#else
			if (i > 0) {
				for (auto t = 0; t < threads; ++t) {
					if (j+t < (int)corpus.size()) {
						try {
							nsentence s = chunker.sample(f, rd[j+t]);
							corpus[rd[j+t]] = s;
						} catch (const char *ex) {
							cerr << ex << endl;
						}
					}
				}
			}
#endif
			for (auto t = 0; t < threads; ++t) {
				if (j+t < (int)corpus.size()) {
					chunker.add(corpus[rd[j+t]]);
				}
			}
			j += threads;
#ifdef _OPENMP
#pragma omp ordered
#endif
			progress("init", i, (double)(i+1)/NNPYLM_EPOCH);
		}
		chunker.estimate(20);
		/*
		   if (i)
		   chunker.poisson_correction(5000);
		   */
	}
	int rpad = 2*PBWIDTH;
	printf("\r%*s", rpad, "");
	/*
	   shared_ptr<cid> d = cid::create();
	   d->save(cdic.c_str());
	   */
	return 0;
}

int snapshot(nphsmm& model, int iter) {
	char mfile[512] = {};
	char dfile[512] = {};
	sprintf(mfile, "%s_iter%03d.model", prefix.c_str(), iter);
	sprintf(dfile, "%s_iter%03d.dic", prefix.c_str(), iter);
	model.save(mfile);
	shared_ptr<cid> d = cid::create();
	d->save(dfile);
	return 0;
}

int mcmc(nio& f, vector<nsentence>& corpus, vector<nsentence>& supervised) {
	nphsmm lm(n, m, l, k);
	if (original) lm.set_original(true);
	if (posbase) lm.set_posbase(true);
	if (ctx > 0) lm.set_ctx(ctx);
	if (ctxgate) lm.set_ctxgate(true);
	if (wclass) lm.set_wclass(true);
	if (wbeta != 1.0) lm.set_wbeta(wbeta);
	if (toklm) lm.set_lex([](word& w, int p) { return toklm->lexlp(w, p); }, toklm->k());
	//lm.set(vocab, K);
	if (!pretrain.empty()) {
		if (K < label_id) {
			K = label_id;
		}
	}
	lm.set_k(K);
	lm.slice(a, b);
	if (!supervised.empty()) {
		nphsmm_pretrain(lm, supervised);
		// supervised anchoring: keep N-1 extra permanent copies of the labeled counts
		for (auto r = 1; r < sweight; ++r) {
			for (auto s = supervised.begin(); s != supervised.end(); ++s)
				lm.add(*s);
		}
	}
#ifdef _OPENMP
	omp_set_num_threads(threads);
#endif
	//shared_ptr<cid> d = cid::create();
	//d->load(cdic.c_str());
	vector<int> rid(corpus.size(), 0);
	//int rid[corpus.size()] = {0};
	/*
	rd::shuffle(rid.data(), corpus.size());
	for (auto i = 0; i < (int)corpus.size(); ++i)
		lm.init(corpus[rid[i]]);
	lm.estimate(20);
	*/
	//lm.poisson_correction(100);
	static const bool pht = (getenv("NPBNLP_PHASE_TIME") != NULL);
	for (auto i = 0; i < epoch; ++i) {
		auto e0 = std::chrono::steady_clock::now();
		vector<int> rd(corpus.size(), 0);
		//int rd[corpus.size()] = {0};
		rd::shuffle(rd.data(), corpus.size());
		int j = 0;
		while (j < (int)corpus.size()) {
			if (i > 0)
				for (auto t = 0; t < threads; ++t) {
					if (j+t < (int)corpus.size())
						lm.remove(corpus[rd[j+t]]);
				}
#ifdef _OPENMP
#pragma omp parallel
			{
				auto t = omp_get_thread_num();
				if (j+t < (int)corpus.size()) {
					try {
						nsentence s = lm.sample(f, rd[j+t]);
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
						nsentence s = lm.sample(f, rd[j+t]);
						corpus[rd[j+t]] = s;
					} catch (const char *ex) {
						cerr << ex << endl;
					}
				}
			}
#endif
			for (auto t = 0; t < threads; ++t) {
				if (j+t < (int)corpus.size()) {
					lm.add(corpus[rd[j+t]]);
				}
			}
			j += threads;
#ifdef _OPENMP
#pragma omp ordered
#endif
			progress("epoch",i, (double)(j+1)/corpus.size());
		}
		auto e1 = std::chrono::steady_clock::now();
		lm.estimate(20);
		auto e2 = std::chrono::steady_clock::now();
		if (pht)
			fprintf(stderr, "[epoch %d time] mcmc=%llds estimate=%llds\n", i,
					(long long)std::chrono::duration_cast<std::chrono::seconds>(e1-e0).count(),
					(long long)std::chrono::duration_cast<std::chrono::seconds>(e2-e1).count());
		//lm.poisson_correction(5000);
		if (dmp && (i+1)%dmp == 0) {
			cout << endl;
			for (auto s = corpus.begin(); s != corpus.end(); ++s)
				dump(*s);
		}
		if ((i+1)%snap == 0) {
			snapshot(lm, i);
		}
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
	nphsmm lm;
	if (original) lm.set_original(true);
	if (posbase) lm.set_posbase(true); // note: load() overrides from the model file
	if (toklm) lm.set_lex([](word& w, int p) { return toklm->lexlp(w, p); }, toklm->k());
	try {
		lm.load(model.c_str());
		//lm.set(vocab, K);
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

int chunking(vector<vector<word> >& supervised, vector<vector<string> >& labels, vector<nsentence>& corpus) {
	for (auto i = 0; i < (int)supervised.size(); ++i) {
		int chunk_head = 0;
		int chunk_len = 0;
		int chunk_k = 0;
		nsentence s;
		for (auto j = 0; j < (int)supervised[i].size(); ++j) {
			string& label = labels[i][j];
			if (label[0] == 'B') {
				string ne(label, 2, string::npos);
				if (label_index.find(ne) == label_index.end()) {
					label_index[ne] = label_id++;
				}
				chunk_head = j;
				chunk_len = 1;
				chunk_k = label_index[ne];
			} else if (label[0] == 'I') {
				chunk_len++;
			} else if (label[0] == 'O') {
			//} else { // if (label[0] == 'O') {
				//if (chunk_len > 0) {
				if (chunk_len > 0 && chunk_k > 1) { // for partial annotation
					chunk c(supervised[i], chunk_head, chunk_len);
					c.k = chunk_k;
					c.type = chunktype::get(c);
					s.c.emplace_back(c);
				}
			/*
				if (label_index.find(label) == label_index.end()) {
					label_index[label] = label_id++;
				}
				*/
				chunk_head = j;
				chunk_len = 1;
				chunk_k = 1;
				//chunk_k = label_index[label];
			}
		}
		chunk c(supervised[i], chunk_head, chunk_len);
		c.k = chunk_k;
		c.type = chunktype::get(c);
		s.c.emplace_back(c);
		s.n.resize(s.c.size()+1, 0);
		corpus.emplace_back(s);
	}
	return 0;
}

int load_label(cio& file, vector<vector<word> >& corpus, vector<vector<string> >& labels) {
	int size = file.chunk->size();
	shared_ptr<wid> dic = wid::create();
	for (auto i = 0; i < size; ++i) {
		io& f = (*file.chunk)[i];
		vector<word> s;
		vector<string> lb;
		for (auto j = 0; j < (int)f.head.size()-1; ++j) {
			int head = f.head[j];
			int tail = f.head[j+1];
			word w(*f.raw);
			w.head = head;
			int p = util::find(9, *f.raw, head, tail);
			w.len = p - head;
			w.id = dic->index(w);
			int y = util::find(9, *f.raw, p+1, tail);
			string pos;
			for (auto k = p+1; k < y; ++k) {
				char buf[5] = {0};
				io::i2c((*f.raw)[k], buf);
				pos += buf;
			}
			if (pos_index.find(pos) == pos_index.end()) {
				pos_index[pos] = pos_id++;
			}
			w.pos = pos_index[pos];
			w.m.resize(w.len+1, 0);
			s.emplace_back(w);
			int l = util::find(9, *f.raw, y+1, tail);
			string pron;
			for (auto k = y+1; k < l; ++k) {
				char buf[5] = {0};
				io::i2c((*f.raw)[k], buf);
				pron += buf;
			}
			string label;
			for (auto k = l+1; k < tail; ++k) {
				char buf[5] = {0};
				io::i2c((*f.raw)[k], buf);
				label += buf;
			}
			//if (label[0] == 'O') {
			//	lb.emplace_back(pos);
			//} else {
				lb.emplace_back(label);
			//}
		}
		corpus.emplace_back(s);
		labels.emplace_back(lb);
	}
	return 0;
}

int init_corpus(nio& f, vector<nsentence>& corpus) {
	for (auto i = 0; i < (int)f.head.size()-1; ++i) {
		nsentence s;
		chunk c(*f.raw, f.head[i], f.head[i+1]-f.head[i]);
		c.k = 1;
		c.type = chunktype::get(c);
		s.c.emplace_back(c);
		s.n.resize(s.c.size()+1, 0);
		corpus.emplace_back(s);
	}
	return 0;
}

// unify the pos id space: supervised words carry gold-POS indexes while the
// unlabeled corpus carries tokenizer latent classes. For --posbase the pos-seq
// base measure needs ONE id space, so re-assign supervised pos by the tokenizer:
//   pos(w) = argmax_p lexlp(w|p) + poslp(p),  cached per word id.
static void reassign_pos(vector<nsentence>& supervised) {
	if (!toklm)
		return;
	unordered_map<int, int> cache;
	int kk = toklm->k();
	for (auto s = supervised.begin(); s != supervised.end(); ++s) {
		for (auto i = 0; i < s->size(); ++i) {
			chunk& c = s->ch(i);
			for (auto j = 0; j < c.len; ++j) {
				word& w = c.wd(j);
				auto it = (w.id > 1) ? cache.find(w.id) : cache.end();
				if (it != cache.end()) {
					w.pos = it->second;
					continue;
				}
				double best = -1e300;
				int arg = 1;
				for (auto p = 1; p <= kk; ++p) {
					double lp = toklm->lexlp(w, p)+toklm->poslp(p);
					if (lp > best) {
						best = lp;
						arg = p;
					}
				}
				w.pos = arg;
				if (w.id > 1)
					cache[w.id] = arg;
			}
		}
	}
}

int main(int argc, char **argv) {
	try {
		read_param(argc, argv);
		if (!train.empty()) {
			io g(train.c_str());
			cio *p = NULL;
			vector<vector<word> > words;
			vector<vector<string> > labels;
			if (!pretrain.empty()) {
				p = new cio(pretrain.c_str());
				load_label(*p, words, labels);
			}
			vector<nsentence> supervised;
			chunking(words, labels, supervised);
			vector<sentence> ws;
			tokenize(g, ws);
			nio f(ws);
			vector<nsentence> corpus;
			init_corpus(f, corpus);
			//vector<nsentence> corpus(f.head.size()-1);
			//init(f, corpus, supervised);
			// unify the pos-id space: pretrain words carry gold POS indices while
			// the unlabeled corpus carries tokenizer latent classes. posbase and the
			// per-word class channel both index by word.pos, so remap supervised
			// words to the tokenizer MAP class to share one space.
			if (posbase || wclass) reassign_pos(supervised);
			mcmc(f, corpus, supervised);
			delete p;
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
