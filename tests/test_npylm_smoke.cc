#include"npylm.h"
#include"io.h"
#include"util.h"
#include<cassert>
#include<cstdio>
#include<vector>

using namespace std;
using namespace npbnlp;

int main(int argc, char **argv) {
	if (argc < 2) {
		fprintf(stderr, "usage: %s data_file\n", argv[0]);
		return 1;
	}
	try {
		io f(argv[1]);
		vector<sentence> corpus;
		util::store_sentences(f, corpus);
		if (corpus.empty()) {
			fprintf(stderr, "empty corpus\n");
			return 1;
		}

		npylm lm(2, 20);
		for (auto i = 0; i < 2; ++i) {
			for (auto j = 0; j < (int)corpus.size(); ++j) {
				if (i > 0)
					lm.remove(corpus[j]);
				sentence s = lm.sample(f, j);
				corpus[j] = s;
				lm.add(corpus[j]);
			}
			lm.estimate(20);
		}

		int check_n = min(3, (int)f.head.size()-1);
		for (auto i = 0; i < check_n; ++i) {
			sentence s = lm.parse(f, i);
			if (s.size() <= 0) {
				fprintf(stderr, "parse of sentence %d produced no words\n", i);
				return 1;
			}
			int total = 0;
			for (auto k = 0; k < s.size(); ++k)
				total += s.wd(k).len;
			int expect = f.head[i+1]-f.head[i];
			if (total != expect) {
				fprintf(stderr, "sentence %d: segmented length %d != input length %d\n", i, total, expect);
				return 1;
			}
		}
	} catch (const char *ex) {
		fprintf(stderr, "exception: %s\n", ex);
		return 1;
	}
	printf("test_npylm_smoke OK\n");
	return 0;
}
