// Regression test for the combination cache in rd/src/negative_binomial.cc.
// Verify the negative_binomial combination cache: the value must not depend on
// call order (the old code keyed inserts by (y+x-1,y) but looked up by (x,y),
// so a later call with x'=y+x-1 picked up the earlier call's coefficient).
#include"negative_binomial.h"
#include<cstdio>
#include<cmath>
using namespace npbnlp;

static double ref_density(double p, int x, int y) {
	// C(y+x-1, y) * p^x * (1-p)^y, computed without any cache
	double c = 1;
	for (int i = 1; i <= y; ++i)
		c = c * (double)(x - 1 + i) / (double)i;
	return c * pow(p, x) * pow(1.0 - p, y);
}

int main() {
	double p = 0.3;
	int bad = 0;
	// exercise the exact collision the bug produced, plus a sweep
	negative_binomial nb;
	int probe[][2] = {{3,2},{4,2},{5,2},{2,3},{4,3},{6,3},{1,1},{7,4}};
	for (auto& q : probe) {
		double got = nb.density(p, q[0], q[1]);
		double want = ref_density(p, q[0], q[1]);
		if (fabs(got - want) > 1e-9 * (1.0 + fabs(want))) {
			printf("MISMATCH x=%d y=%d got=%g want=%g\n", q[0], q[1], got, want);
			++bad;
		}
	}
	for (int y = 0; y <= 12; ++y) {
		for (int x = 1; x <= 12; ++x) {
			double got = nb.density(p, x, y);
			double want = ref_density(p, x, y);
			if (fabs(got - want) > 1e-9 * (1.0 + fabs(want))) {
				printf("MISMATCH x=%d y=%d got=%g want=%g\n", x, y, got, want);
				++bad;
			}
		}
	}
	if (bad) {
		printf("nb_check FAILED (%d mismatches)\n", bad);
		return 1;
	}
	printf("nb_check OK\n");
	return 0;
}
