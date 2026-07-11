#include"hpyp.h"
#include"context.h"
#include<cassert>
#include<cmath>
#include<cstdio>
#include<vector>

using namespace std;
using namespace npbnlp;

int main(int argc, char **argv) {
	try {
		hpyp lm(1);
		context *h = lm.h();

		vector<int> keys;
		for (auto k = 1; k <= 20; ++k) {
			int reps = (k % 3) + 1;
			for (auto r = 0; r < reps; ++r)
				keys.push_back(k);
		}

		for (auto round = 0; round < 3; ++round) {
			// add everything
			for (auto& k : keys) {
				lm.add(k, h);
				double lpr = lm.lp(k, h);
				if (!isfinite(lpr)) {
					fprintf(stderr, "round %d: lp(%d) not finite after add: %f\n", round, k, lpr);
					return 1;
				}
			}
			// remove in reverse order
			for (auto it = keys.rbegin(); it != keys.rend(); ++it) {
				lm.remove(*it, h);
			}
			if (h->c() != 0) {
				fprintf(stderr, "round %d: customer count not zero after full removal: %d\n", round, h->c());
				return 1;
			}
			if (h->t() != 0) {
				fprintf(stderr, "round %d: table count not zero after full removal: %d\n", round, h->t());
				return 1;
			}
			// lm should still answer finite lp on an empty restaurant
			double lpr = lm.lp(1, h);
			if (!isfinite(lpr)) {
				fprintf(stderr, "round %d: lp not finite on empty restaurant: %f\n", round, lpr);
				return 1;
			}
		}
	} catch (const char *ex) {
		fprintf(stderr, "exception: %s\n", ex);
		return 1;
	}
	printf("test_hpyp_symmetry OK\n");
	return 0;
}
