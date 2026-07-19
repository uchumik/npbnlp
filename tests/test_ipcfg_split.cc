#include"ipcfg.h"
#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<cstring>
#include<cstdint>
#include<vector>

using namespace npbnlp;

// Sum_{L=1}^{w-1} P(b|i,j,q) must be exactly 1: the geometric mass is
// renormalised over the truncated support, so a missing normaliser shows up
// here immediately.
static bool check_normalisation() {
	const double qs[] = {.1, .5, .9};
	for (int qi = 0; qi < 3; ++qi) {
		for (int w = 2; w <= 10; ++w) {
			ipcfg g;
			g.split(1., 1., qs[qi], true);
			if (!g.split_enabled()) {
				fprintf(stderr, "split() refused q=%f\n", qs[qi]);
				return false;
			}
			double z = 0.;
			for (int b = 0; b < w-1; ++b) // i=0, j=w-1, L=b+1
				z += exp(g.split_logprob(0, w-1, b));
			if (fabs(z-1.) > 1e-9) {
				fprintf(stderr, "split prior is not normalised: q=%f w=%d sum=%.12f\n",
					qs[qi], w, z);
				return false;
			}
		}
	}
	// Disabled prior must be identically zero in the log domain.
	ipcfg off;
	if (off.split_logprob(0, 5, 2) != 0.) {
		fprintf(stderr, "disabled split prior is not a no-op\n");
		return false;
	}
	return true;
}

int main(int argc, char **argv) {
	if (argc != 2) {
		fprintf(stderr, "usage: %s corpus\n", argv[0]);
		return 2;
	}
	try {
		if (!check_normalisation())
			return 1;

		io f(argv[1]);
		// init()/add() -> remove() must return every split statistic to zero.
		ipcfg g;
		g.set(8);
		g.span(1., 1.);
		g.split(1., 1., .7);
		tree t = g.init(f, 0);
		if (g.split_count() <= 0) {
			fprintf(stderr, "no split statistics were recorded by init\n");
			return 1;
		}
		g.remove(t);
		if (g.split_count() != 0 || g.split_left_excess() != 0) {
			fprintf(stderr, "split counts did not round-trip to zero: n=%lld sum=%lld\n",
				g.split_count(), g.split_left_excess());
			return 1;
		}
		// empty() also verifies that the width histogram is all zero.
		if (!g.empty()) {
			fprintf(stderr, "iPCFG did not round-trip to empty with the split prior\n");
			return 1;
		}

		// Sampling with the split prior enabled must keep the current tree
		// inside the slice lattice (the _slice_*_cond asserts) and stay finite.
		g.add(t);
		g.estimate(1);
		if (!(g.split_probability() > 0.) || !(g.split_probability() < 1.)) {
			fprintf(stderr, "invalid split posterior draw: q=%f\n", g.split_probability());
			return 1;
		}
		double q_before = g.split_probability();
		tree resampled = g.sample(f, 0, &t);
		double log_q = 0., log_q_cur = 0.;
		tree proposal = g.mh_propose(f, 0, &t, log_q, log_q_cur);
		if (!std::isfinite(log_q) || !std::isfinite(log_q_cur)) {
			fprintf(stderr, "non-finite slice proposal score with the split prior\n");
			return 1;
		}
		(void)resampled;
		(void)proposal;

		// version 6 round-trips.
		g.save("/tmp/qc_ipcfg_split.model");
		ipcfg restored;
		restored.load("/tmp/qc_ipcfg_split.model");
		if (!restored.split_enabled() ||
		    restored.split_probability() != q_before ||
		    restored.split_count() != g.split_count() ||
		    restored.split_left_excess() != g.split_left_excess()) {
			fprintf(stderr, "split prior was not restored from a version-6 model\n");
			return 1;
		}
		g.remove(t);

		// A version-5 model (no split tail) must load with the prior disabled.
		ipcfg v5;
		v5.set(8);
		v5.span(1., 1.);
		tree t5 = v5.init(f, 0);
		v5.save("/tmp/qc_ipcfg_split_v5.model");
		v5.remove(t5);
		FILE *fp = fopen("/tmp/qc_ipcfg_split_v5.model", "rb");
		if (!fp) {
			fprintf(stderr, "cannot reopen the version-5 fixture\n");
			return 1;
		}
		fseek(fp, 0, SEEK_END);
		long size = ftell(fp);
		fseek(fp, 0, SEEK_SET);
		std::vector<unsigned char> buf(size);
		if (fread(buf.data(), 1, size, fp) != (size_t)size) {
			fclose(fp);
			fprintf(stderr, "cannot read the version-5 fixture\n");
			return 1;
		}
		fclose(fp);
		// With no split recorded the version-6 tail is a fixed 48 bytes:
		// enabled(int) a,b,q(3 doubles) n,sum(2 long long) hist_size(int).
		const long tail = 4+8*3+8*2+4;
		if (size <= tail) {
			fprintf(stderr, "version-5 fixture is too small\n");
			return 1;
		}
		// Patch the version field, which directly follows the "PAGP" magic.
		long magic_at = -1;
		for (long i = size-tail-8; i >= 0; --i) {
			uint32_t m = 0;
			memcpy(&m, buf.data()+i, sizeof(uint32_t));
			if (m == 0x50414750) {
				magic_at = i;
				break;
			}
		}
		if (magic_at < 0) {
			fprintf(stderr, "cannot locate the tail magic\n");
			return 1;
		}
		uint32_t five = 5;
		memcpy(buf.data()+magic_at+4, &five, sizeof(uint32_t));
		fp = fopen("/tmp/qc_ipcfg_split_v5.model", "wb");
		if (!fp || fwrite(buf.data(), 1, size-tail, fp) != (size_t)(size-tail)) {
			if (fp) fclose(fp);
			fprintf(stderr, "cannot write the version-5 fixture\n");
			return 1;
		}
		fclose(fp);
		ipcfg legacy;
		legacy.load("/tmp/qc_ipcfg_split_v5.model");
		if (legacy.split_enabled() || legacy.split_count() != 0 ||
		    legacy.split_left_excess() != 0) {
			fprintf(stderr, "a version-5 model did not load with the split prior disabled\n");
			return 1;
		}
	} catch (const char *ex) {
		fprintf(stderr, "exception: %s\n", ex);
		return 1;
	}
	printf("test_ipcfg_split OK\n");
	return 0;
}
