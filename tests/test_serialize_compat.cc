#include"nphsmm.h"
#include<cassert>
#include<cstdio>
#include<cstdlib>
#include<vector>

using namespace std;
using namespace npbnlp;

static long file_size(const char *path) {
	FILE *fp = fopen(path, "rb");
	if (!fp)
		return -1;
	fseek(fp, 0, SEEK_END);
	long sz = ftell(fp);
	fclose(fp);
	return sz;
}

static bool truncate_copy(const char *src, const char *dst, long strip_bytes) {
	long sz = file_size(src);
	if (sz < 0 || sz <= strip_bytes)
		return false;
	long keep = sz-strip_bytes;
	FILE *in = fopen(src, "rb");
	FILE *out = fopen(dst, "wb");
	if (!in || !out)
		return false;
	vector<char> buf(keep);
	if ((long)fread(buf.data(), 1, keep, in) != keep) {
		fclose(in);
		fclose(out);
		return false;
	}
	bool ok = (fwrite(buf.data(), 1, keep, out) == (size_t)keep);
	fclose(in);
	fclose(out);
	return ok;
}

int main(int argc, char **argv) {
	const char *full_file = "/tmp/qc_nphsmm_full.bin";
	const char *trunc_file = "/tmp/qc_nphsmm_trunc.bin";
	try {
		nphsmm lm;
		lm.save(full_file);

		nphsmm lm2;
		lm2.load(full_file);

		// simulate a legacy model saved before the wclass tail block existed:
		// the tail is wcf(int) + pv(int) + wn(int) = 12 bytes when wn == 0
		// (default-constructed model has _wclass == false, _wc empty).
		if (!truncate_copy(full_file, trunc_file, 12)) {
			fprintf(stderr, "failed to build truncated legacy-format copy\n");
			return 1;
		}

		nphsmm lm3;
		lm3.load(trunc_file); // must also succeed: legacy fallback (wclass off)
	} catch (const char *ex) {
		fprintf(stderr, "exception: %s\n", ex);
		return 1;
	}
	printf("test_serialize_compat OK\n");
	return 0;
}
