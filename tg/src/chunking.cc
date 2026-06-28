#include<getopt.h>
#include<cstdlib>
#include<cstdio>
#include"phsmm.h"
#include"nnpylm.h"
#include"chunktype.h"
#include"ihmm.h"
#include"util.h"
#include"rd.h"
#include"cio.h"

#define check(opt,arg) (strcmp(opt,arg) == 0)
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

using namespace npbnlp;
using namespace std;

static int n = 1;
static int m = 3;
static int l = 20;
static int threads = 4;
static int epoch = 100;
static int pre_epoch = 20;
static int dmp = 0;
//static int tokenized = 0;
static int vocab = 0;
static string pretrain;
static string train;
static string test;
static string tokenizer("phsmm.model");
static string model("nnpylm.model");
static string cdic("chunk.dic");
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
	cout << *argv << " --train file --tokenizer phsmm.model --model file_to_save --cdic dicfile --wdic ma.dic\n";
	cout << *argv << " --parse file --tokenizer phsmm.model --model modelfile --cdic dicfile --wdic ma.dic\n";
	cout << "[options]\n";
	cout << "-n, --order=int(default 2)\n";
	cout << "-m, --word_order=int(default 3)\n";
	cout << "-l, --letter_order=int(default 20)\n";
	cout << "-e, --epoch=int(default 500)\n";
	cout << "-t, --threads=int(default 4)\n";
	cout << "-v, --vocab=int(means letter variations. default 0: train from data)\n";
	cout << "--pretrain =file(use as pretraining dataset in training\n";
	//cout << "--tokenized=bool(default 0)\n";
	exit(1);
}


int read_long_param(const char *opt, const char *arg) {
	if (check(opt, "train")) {
		train = arg;
	} else if (check(opt, "pretrain")) {
		pretrain = arg;
	} else if (check(opt, "parse")) {
		test = arg;
	} else if (check(opt, "model")) {
		model = arg;
	} else if (check(opt, "cdic")) {
		cdic = arg;
	} else if (check(opt, "wdic")) {
		wdic = arg;
	} else if (check(opt, "order")) {
		n = atoi(arg);
	} else if (check(opt, "word_order")) {
		m = atoi(arg);
	} else if (check(opt, "letter_order")) {
		l = atoi(arg);
	} else if (check(opt, "epoch")) {
		epoch = atoi(arg);
	} else if (check(opt, "threads")) {
		threads = atoi(arg);
	} else if (check(opt, "dump")) {
		dmp = atoi(arg);
	} else if (check(opt, "tokenizer")) {
		tokenizer = arg;
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
			{"model", required_argument, 0, 0},
			{"cdic", required_argument, 0,0},
			{"wdic", required_argument, 0,0},
			{"order", required_argument, 0,0},
			{"word_order", required_argument, 0, 0},
			{"letter_order", required_argument, 0, 0},
			{"epoch", required_argument, 0,0},
			{"threads", required_argument, 0,0},
			{"dump", required_argument, 0,0},
			{"tokenizer", required_argument, 0,0},
			//{"tokenized", no_argument, &tokenized, 1},
			{0, 0, 0, 0}
		};
		int option_index = 0;
		c = getopt_long(argc, argv, "n:m:l:e:t:d:", long_options, &option_index);
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

void dump(nsentence& s) {
	for (auto i = 0; i < s.size(); ++i)
		cout << s.ch(i) << endl;
	cout << endl;
}

int tokenize(io& f, vector<sentence>& c) {
	shared_ptr<wid> d = wid::create();
	d->load(wdic.c_str());
	phsmm lm;
	lm.load(tokenizer.c_str());
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
	//d->save(wdic.c_str());
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

int init(nio& f, vector<nsentence>& corpus) {
	for (auto i = 0; i < (int)f.head.size()-1; ++i) {
		nsentence s;
		chunk c(*f.raw, f.head[i], f.head[i+1]-f.head[i]);
		c.k = 1;
		//c.type = chunktype::get(c);
		c.type = chunktype2::get(c);
		s.c.emplace_back(c);
		s.n.resize(s.c.size()+1, 0);
		corpus.emplace_back(s);
	}
	return 0;
}

int mcmc(vector<nsentence>& supervised) {
	io g(train.c_str());
	vector<sentence> ws;
	tokenize(g, ws);
	nio f(ws);
	/*
	   for (auto i = 0; i < f.head.size()-1; ++i) {
	   for (auto j = f.head[i]; j < f.head[i+1]; ++j) {
	   cout << (*f.raw)[j] << endl;
	   }
	   }
	vector<nsentence> corpus(f.head.size()-1);
	   */
	vector<nsentence> corpus;
	init(f, corpus);

	phsmm lm;
	lm.load(tokenizer.c_str());
	m = lm.n();
	l = lm.m();
	/*
	if (tokenized) {
	} else {
		m = lm.n();
		l = lm.m();
	}
	*/
	nnpylm chunker(n, m, l);
	if (!pretrain.empty())
		nnpy_pretrain(chunker, supervised);
	/*
	if (vocab)
		chunker.set(vocab);
		*/
#ifdef _OPENMP
	omp_set_num_threads(threads);
#endif
	for (auto i = 0; i < epoch; ++i) {
		vector<int> rd(corpus.size(), 0);
		//int rd[corpus.size()] = {0};
		rd::shuffle(rd.data(), corpus.size());;
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
							nsentence s = chunker.sample(f, rd[j+t]);
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
				if (j+t < (int)corpus.size())
					chunker.add(corpus[rd[j+t]]);
			}
			j += threads;
#ifdef _OPENMP
#pragma omp ordered
#endif
			progress("epoch",i, (double)(j+1)/corpus.size());
		}
		chunker.estimate(20);
		/*
		   if (i)
		   chunker.poisson_correction(5000);
		   */
		if (dmp && (i+1)%dmp == 0) {
			cout << endl;
			for (auto s = corpus.begin(); s != corpus.end(); ++s)
				dump(*s);
		}
	}
	cout << endl;
	chunker.save(model.c_str());
	shared_ptr<cid> d = cid::create();
	d->save(cdic.c_str());
	return 0;
}

int parse() {
	io g(test.c_str());
	vector<sentence> ws;
	tokenize(g, ws);
	shared_ptr<cid> d = cid::create();
	d->load(cdic.c_str());
	nio f(ws);
	/*
	   npylm lm;
	   lm.load(tokenizer.c_str());
	   */
	nnpylm chunker;//(n, lm.n(), lm.m());
	chunker.load(model.c_str());
#ifdef _OPENMP
	omp_set_num_threads(threads);
#pragma omp parallel for ordered schedule(dynamic)
#endif
	for (auto i = 0; i < (int)f.head.size()-1; ++i) {
		nsentence s = chunker.parse(f, i);
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
				if (chunk_len > 0) {
					chunk c(supervised[i], chunk_head, chunk_len);
					c.k = chunk_k;
					c.type = chunktype::get(c);
					s.c.emplace_back(c);
				}
				//cout << c << " len:" << chunk_len << " k:" << chunk_k << endl;
				chunk_head = j;
				chunk_len = 1;
				chunk_k = 1;
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
			lb.emplace_back(label);
		}
		corpus.emplace_back(s);
		labels.emplace_back(lb);
	}
	return 0;
}

int main(int argc, char **argv) {
	try {
		read_param(argc, argv);
		if (!train.empty()) {
			cio *p = NULL;
			vector<vector<word> > words;
			vector<vector<string> > labels;
			if (!pretrain.empty()) {
				p = new cio(pretrain.c_str());
				load_label(*p, words, labels);
			}
			vector<nsentence> supervised;
			chunking(words, labels, supervised);
			mcmc(supervised);
			delete p;
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
