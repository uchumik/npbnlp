#include "seed.h"
#include "vhmm.h"
#include <cstring>
#include "rd.h"
#include "util.h"
#include <algorithm>
#include <set>
#include <getopt.h>
#include <iostream>
#include <map>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace npbnlp;

static int poisson=0, estimate_iter=20;
static int max_order=3, min_order=1, letter_order=20, states=10, max_states=50, epochs=500, threads=4;
static int parse_iter=20;
// Order 1 makes the emission depend only on the root; larger orders condition on
// preceding observed words.
static int word_ngram=1;
static double slice_a=1, slice_b=1, alpha=5.;
static bool dump_tsv=false, have_seed=false;
static unsigned int random_seed=0;
static string train_file, parse_file, model_file("vhmm.model"), dic_file("vtg.dic");

static void usage(char **argv) {
	cerr << argv[0] << " --train FILE|--parse FILE [--max_order N --min_order N --alpha A --seed N --dump_tsv --parse_iter N --word_ngram N]\n";
	cerr << "run with OMP_WAIT_POLICY=passive when using --threads > 1\n";
	exit(1);
}

static void set_seed() {
	if (!have_seed)
		return;
	seed::create()->set(random_seed);
	generator::reseed();
}

static int params(int argc,char **argv) {
	static option opts[]={{"train",1,0,0},{"parse",1,0,0},{"model",1,0,0},{"dic",1,0,0},{"max_order",1,0,0},{"min_order",1,0,0},{"letter_order",1,0,0},{"pos",1,0,0},{"epoch",1,0,0},{"threads",1,0,0},{"slice_a",1,0,0},{"slice_b",1,0,0},{"alpha",1,0,0},{"seed",1,0,0},{"dump_tsv",0,0,0},{"poisson",1,0,0},{"estimate_iter",1,0,0},{"parse_iter",1,0,0},{"word_ngram",1,0,0},{0,0,0,0}};
	int index=0;
	int c;
	while ((c=getopt_long(argc,argv,"",opts,&index))!=-1) {
		if (c!='?') {
			string o=opts[index].name;
			if (o=="train") train_file=optarg;
			else if (o=="parse") parse_file=optarg;
			else if (o=="model") model_file=optarg;
			else if (o=="dic") dic_file=optarg;
			else if (o=="max_order") max_order=atoi(optarg);
			else if (o=="min_order") min_order=atoi(optarg);
			else if (o=="letter_order") letter_order=atoi(optarg);
			else if (o=="pos") { states=atoi(optarg); max_states=states; }
			else if (o=="epoch") epochs=atoi(optarg);
			else if (o=="threads") threads=atoi(optarg);
			else if (o=="poisson") poisson=atoi(optarg);
			else if (o=="estimate_iter") estimate_iter=atoi(optarg);
			else if (o=="slice_a") slice_a=atof(optarg);
			else if (o=="slice_b") slice_b=atof(optarg);
			else if (o=="alpha") alpha=atof(optarg);
			else if (o=="seed") { random_seed=strtoul(optarg,NULL,10); have_seed=true; }
			else if (o=="dump_tsv") dump_tsv=true;
			else if (o=="parse_iter") parse_iter=atoi(optarg);
			else if (o=="word_ngram") word_ngram=atoi(optarg);
		}
	}
	if (train_file.empty() && parse_file.empty())
		usage(argv);
	return 0;
}

static void dump(sentence& s) {
	for (int i=0;i<s.size();++i) {
		if (dump_tsv) {
			for (int j=0;j<s.wd(i).len;++j) { char buf[5]={0}; io::i2c(s.wd(i)[j],buf); cout<<buf; }
			cout<<"\t"<<s.wd(i).id<<"\t"<<s.wd(i).pos<<"\t"<<s.n[i]<<"\n";
		} else cout<<s.wd(i)<<" ";
	}
	cout<<"\n";
}

static int train(io& f, vector<sentence>& corpus) {
	vhmm hmm(max_order,min_order,letter_order,states); hmm.set_word_ngram(word_ngram); hmm.set_alpha(alpha); hmm.set_k(max_states); hmm.slice(slice_a,slice_b);
	for (int i=0;i<(int)corpus.size();++i) hmm.init(corpus[i],i);
	hmm.refresh_cache();
	for (int e=0;e<epochs;++e) {
		hmm.refresh_cache();
		// Remove a block before sampling, refresh the transition cache while its
		// customers are absent, sample the block from the read-only model, then
		// seat the sampled sentences serially.
		int size=(int)corpus.size();
		int block=threads<1?1:threads;
		for (int id=0; id<size; id+=block) {
			int last=min(id+block,size);
			for (int j=id;j<last;++j) hmm.remove(corpus[j],j);
			hmm.cache_max();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(block)
#endif
			for (int j=id;j<last;++j) {
				try { corpus[j]=hmm.sample(corpus[j]); }
				catch (const char *e) { cerr<<"vtg: "<<e<<" (sentence "<<j<<")\n"; throw; }
			}
			for (int j=id;j<last;++j) hmm.add(corpus[j],j);
		}
		hmm.estimate(estimate_iter); if (poisson) hmm.poisson_correction(1000); double lp=0.; for (auto& s:corpus) lp+=hmm.log_probability(s); int live=0; {std::map<int,int> h; for (auto& s2:corpus) for (int i=0;i<s2.size();++i) h[s2.wd(i).pos]++; live=h.size();} cerr<<"epoch "<<e+1<<" state="<<hmm.k()<<" live="<<live<<" log P "<<lp<<"\n";
		{ map<int,long long> oh; for (auto& s2:corpus) for (int i=0;i<s2.size();++i) ++oh[s2.wd(i).n]; cerr<<"[ord] epoch "<<e+1; for (auto& x:oh) cerr<<" "<<x.first<<":"<<x.second; cerr<<"\n"; }
		{
			map<int,long long> use; map<int,set<int> > succ; map<int,set<int> > vocab;
			for (auto& s2:corpus) for (int i=0;i<s2.size();++i) {
				int k=s2.wd(i).pos; ++use[k]; vocab[k].insert(s2.wd(i).id);
				if (i+1<s2.size()) succ[k].insert(s2.wd(i+1).pos);
			}
			double tot=0; for (auto& x:use) tot+=x.second;
			double H=0; for (auto& x:use) { double q=x.second/tot; if (q>0) H-=q*log(q); }
			map<int,map<int,long long> > bi; long long bn=0;
			for (auto& s2:corpus) for (int i=0;i+1<s2.size();++i) { ++bi[s2.wd(i).pos][s2.wd(i+1).pos]; ++bn; }
			double Hc=0;
			for (auto& a2:bi) { double c2=0; for (auto& b2:a2.second) c2+=b2.second;
				double hh=0; for (auto& b2:a2.second) { double q=b2.second/c2; if(q>0) hh-=q*log(q); }
				Hc += (c2/bn)*hh; }
			double ms=0; for (auto& x:succ) ms+=x.second.size();
			double mv=0; for (auto& x:vocab) mv+=x.second.size();
			cerr<<"[state] epoch "<<e+1<<" live="<<use.size()<<" H="<<H
				<<" succ/state="<<(succ.empty()?0:ms/succ.size())
				<<" types/state="<<(vocab.empty()?0:mv/vocab.size())
				<<" H(next|s)="<<Hc<<" eff_succ="<<exp(Hc)<<"\n";
		}
		{ char b[32]; snprintf(b,sizeof(b),"epoch%d",e+1); hmm.dump_tree_stats(b); }
	}
	map<int,long long> pos_count, order_count;
	long long token_count=0;
	for (auto& s:corpus) for (int i=0;i<s.size();++i) {
		++pos_count[s.wd(i).pos];
		++order_count[s.wd(i).n];
		++token_count;
	}
	if (const char *tsv = getenv("VTG_DUMP_TRAIN")) {
		FILE *fp = fopen(tsv, "w");
		if (fp) {
			for (auto& s2 : corpus) {
				for (int i=0; i<s2.size(); ++i) {
					word& w = s2.wd(i);
					for (int j=0; j<w.len; ++j) { char buf[5]={0}; io::i2c(w[j], buf); fputs(buf, fp); }
					fprintf(fp, "\t%d\n", w.pos);
				}
			}
			fclose(fp);
		}
	}
	vector<pair<int,long long> > top(pos_count.begin(),pos_count.end());
	sort(top.begin(),top.end(),[](const pair<int,long long>& a,const pair<int,long long>& b) { return a.second>b.second; });
	cerr<<"pos distinct "<<pos_count.size()<<" top5";
	for (int i=0;i<(int)top.size() && i<5;++i) cerr<<" "<<top[i].first<<":"<<top[i].second<<"/"<<token_count;
	cerr<<"\norder histogram";
	for (auto& x:order_count) cerr<<" "<<x.first<<":"<<x.second;
	cerr<<"\n";
	hmm.dump_timing();
	cerr<<"mean beam "<<hmm.mean_beam()<<"\n"; cerr<<"clamp hits "<<hmm.clamp_hits()<<" / "<<hmm.clamp_total()<<"\n";
	hmm.save(model_file.c_str()); wid::create()->save(dic_file.c_str()); return 0;
}

static int parse() {
	io f(parse_file.c_str()); wid::create()->load(dic_file.c_str()); vhmm hmm(max_order,min_order,letter_order,states); hmm.set_word_ngram(word_ngram); hmm.load(model_file.c_str()); hmm.set_fixed();
	vector<sentence> corpus;
	util::store_sentences(f,corpus);
	// Inference does not change customers, so one transition-cache build serves all
	// sentences.
	hmm.refresh_cache();
	if (getenv("VTG_TREE")) hmm.dump_tree_stats("after-load");
	if (const char *tsv = getenv("VTG_LOAD_ASSIGN")) {
		FILE *fp = fopen(tsv, "r");
		if (!fp) throw "cannot open VTG_LOAD_ASSIGN";
		char buf[4096];
		for (int i=0;i<(int)corpus.size();++i) {
			sentence s=corpus[i];
			hmm.inference_init(s);
			for (int j=0;j<s.size();++j) {
				if (!fgets(buf,sizeof(buf),fp)) throw "VTG_LOAD_ASSIGN is shorter than the corpus";
				char *tab=strrchr(buf,'\t');
				if (!tab) throw "VTG_LOAD_ASSIGN is not surface<TAB>class";
				s.wd(j).pos=atoi(tab+1);
			}
			hmm.draw_order(s);
			dump(s);
			corpus[i]=s;
		}
		fclose(fp);
	} else
	for (int i=0;i<(int)corpus.size();++i) {
		sentence s=corpus[i];
		hmm.inference_init(s);
		if (getenv("VTG_EMIT_ARGMAX")) {
			hmm.emission_argmax(s);
			dump(s);
			corpus[i]=s;
			continue;
		}
		for (int j=0;j<parse_iter;++j) s=hmm.parse(s);
		dump(s);
		corpus[i]=s;
	}
	{ double lp=0., em=0., tr=0.; for (auto& s2:corpus) { lp+=hmm.log_probability(s2); hmm.log_probability_parts(s2,em,tr); } cerr<<"total log P "<<lp<<"  emission "<<em<<"  transition "<<tr<<"\n"; }
	cerr<<"mean beam "<<hmm.mean_beam()<<"\n";
	cerr<<"clamp hits "<<hmm.clamp_hits()<<" / "<<hmm.clamp_total()<<"\n";
	return 0;
}

int main(int argc,char **argv) {
	try {
		params(argc,argv);
#ifdef _OPENMP
		// Before set_seed(): generator::reseed() sizes its per-thread state from
		// omp_get_max_threads(), and the workers index it by thread number.
		omp_set_num_threads(threads<1?1:threads);
		omp_set_dynamic(0);
#endif
		set_seed();
		if (!train_file.empty()) {
			io f(train_file.c_str());
			vector<sentence> c;
			util::store_sentences(f,c);
			return train(f,c);
		}
		return parse();
	} catch (const char *e) {
		cerr << e << endl;
		return 1;
	}
}
