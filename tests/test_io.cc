#include"io.h"
#include<cassert>
#include<cstdio>
#include<fstream>
#include<sstream>
#include<string>

using namespace std;
using namespace npbnlp;

int main(int argc, char **argv) {
	if (argc < 2) {
		fprintf(stderr, "usage: %s data_file\n", argv[0]);
		return 1;
	}
	try {
		io f(argv[1]);
		assert(!f.head.empty());
		assert(f.head[0] == 0);
		for (auto i = 1; i < (int)f.head.size(); ++i) {
			if (f.head[i] <= f.head[i-1]) {
				fprintf(stderr, "head is not strictly increasing at %d: %d <= %d\n", i, f.head[i], f.head[i-1]);
				return 1;
			}
		}
		if (f.head.back() != (int)f.raw->size()) {
			fprintf(stderr, "last head (%d) != raw->size() (%d)\n", f.head.back(), (int)f.raw->size());
			return 1;
		}

		// u8size / u8strlen sanity check
		const char *s = "\xE3\x81\x82" "a"; // "あ" + "a"
		if (io::u8strlen(s) != 2) {
			fprintf(stderr, "u8strlen(\"%s\") = %u, expected 2\n", s, io::u8strlen(s));
			return 1;
		}
		if (io::u8size(s) != 3) {
			fprintf(stderr, "u8size of first char = %u, expected 3\n", io::u8size(s));
			return 1;
		}

		// build an equivalent io from an istringstream and compare heads
		ifstream in(argv[1]);
		if (!in) {
			fprintf(stderr, "couldn't reopen %s\n", argv[1]);
			return 1;
		}
		stringstream buf;
		buf << in.rdbuf();
		istringstream iss(buf.str());
		io g(iss);
		if (g.head.size() != f.head.size()) {
			fprintf(stderr, "istringstream io head size %d != file io head size %d\n", (int)g.head.size(), (int)f.head.size());
			return 1;
		}
		for (auto i = 0; i < (int)f.head.size(); ++i) {
			if (f.head[i] != g.head[i]) {
				fprintf(stderr, "head mismatch at %d: file=%d stream=%d\n", i, f.head[i], g.head[i]);
				return 1;
			}
		}
	} catch (const char *ex) {
		fprintf(stderr, "exception: %s\n", ex);
		return 1;
	}
	printf("test_io OK\n");
	return 0;
}
