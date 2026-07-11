#include"hpyp.h"
#include"context.h"
#include"io.h"
#include<cassert>
#include<cmath>
#include<cstdio>
#include<cstring>
#include<vector>

using namespace std;
using namespace npbnlp;

int main(int argc, char **argv) {
	if (argc < 2) {
		fprintf(stderr, "usage: %s data_file\n", argv[0]);
		return 1;
	}
	const char *model_file = "/tmp/qc_hpyp.bin";
	try {
		io f(argv[1]);
		vector<int> seq;
		for (auto& c : *f.raw)
			seq.push_back((int)c);
		if ((int)seq.size() < 10) {
			fprintf(stderr, "corpus too small for roundtrip test\n");
			return 1;
		}

		hpyp lm(2);
		for (auto i = 1; i < (int)seq.size(); ++i) {
			context *h = lm.h();
			h = h->make(seq[i-1]);
			lm.add(seq[i], h);
		}
		lm.estimate(20);

		double d0 = lm.discount(0);
		double d1 = lm.discount(1);
		double s0 = lm.strength(0);
		double s1 = lm.strength(1);

		// sample a handful of (context, key) probes
		vector<int> probe_idx;
		for (auto i = 1; i < (int)seq.size() && (int)probe_idx.size() < 8; i += (int)seq.size()/8+1)
			probe_idx.push_back(i);
		vector<double> lp_before;
		for (auto& i : probe_idx) {
			context *h = lm.h();
			context *c = h->find(seq[i-1]);
			if (!c)
				c = h;
			lp_before.push_back(lm.lp(seq[i], c));
		}

		lm.save(model_file);

		hpyp lm2(2);
		lm2.load(model_file);

		double d0b = lm2.discount(0);
		double d1b = lm2.discount(1);
		double s0b = lm2.strength(0);
		double s1b = lm2.strength(1);
		if (memcmp(&d0, &d0b, sizeof(double)) != 0 ||
		    memcmp(&d1, &d1b, sizeof(double)) != 0 ||
		    memcmp(&s0, &s0b, sizeof(double)) != 0 ||
		    memcmp(&s1, &s1b, sizeof(double)) != 0) {
			fprintf(stderr, "discount/strength mismatch after save/load\n");
			fprintf(stderr, "d0=%.17g d0'=%.17g d1=%.17g d1'=%.17g\n", d0, lm2.discount(0), d1, lm2.discount(1));
			fprintf(stderr, "s0=%.17g s0'=%.17g s1=%.17g s1'=%.17g\n", s0, lm2.strength(0), s1, lm2.strength(1));
			return 1;
		}

		for (auto j = 0; j < (int)probe_idx.size(); ++j) {
			int i = probe_idx[j];
			context *h = lm2.h();
			context *c = h->find(seq[i-1]);
			if (!c)
				c = h;
			double lpr = lm2.lp(seq[i], c);
			if (lpr != lp_before[j]) {
				fprintf(stderr, "lp mismatch at probe %d: before=%.17g after=%.17g\n", i, lp_before[j], lpr);
				return 1;
			}
		}
	} catch (const char *ex) {
		fprintf(stderr, "exception: %s\n", ex);
		return 1;
	}
	printf("test_hpyp_roundtrip OK\n");
	return 0;
}
