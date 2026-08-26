#include "vlattice.h"
#include <cmath>

using namespace std;
using namespace npbnlp;

#define ZERO 1e-36

static word eos;
static vector<int> bos(1, 0);

vlattice::vlattice(sentence& x):s(x) {
	mu.resize(s.size()+1, 0);
	k.resize(s.size()+1);
	cur.resize(s.size()+1, 0);
	n.resize(s.size()+1, 1);
	for (int j=0; j<s.size(); ++j)
		n[j] = s.wd(j).n > 0 ? s.wd(j).n : 1;
}

vlattice::vlattice(io& f, int i):s(*f.raw, f.head[i], f.head[i+1]) {
	mu.resize(s.size()+1, 0);
	k.resize(s.size()+1);
	cur.resize(s.size()+1, 0);
	n.resize(s.size()+1, 1);
	for (int j=0; j<s.size(); ++j)
		n[j] = s.wd(j).n > 0 ? s.wd(j).n : 1;
}

vlattice::~vlattice() {
}

word& vlattice::wd(int i) {
	if (i < 0 || i >= s.size())
		return eos;
	return s.wd(i);
}

int vlattice::size(int i) {
	if (i < 0 || i >= (int)k.size())
		return 1;
	return k[i].size();
}

void vlattice::slice(int i, double x) {
	if (i >= 0 && i < (int)mu.size())
		mu[i] = x;
}

double vlattice::u(int i) {
	if (i < 0 || i >= (int)mu.size())
		return log(ZERO);
	return mu[i];
}

vector<int>::iterator vlattice::begin(int i) {
	if (i < 0 || i >= (int)k.size())
		return bos.begin();
	return k[i].begin();
}

vector<int>::iterator vlattice::end(int i) {
	if (i < 0 || i >= (int)k.size())
		return bos.end();
	return k[i].end();
}

int vlattice::order(int i) {
	if (i < 0 || i >= (int)n.size())
		return 1;
	return n[i];
}

void vlattice::set_order(int i, int x) {
	if (i < 0 || i >= (int)n.size())
		return;
	n[i] = x;
	// The sentence carries the sampled order out of _sample, including EOS.
	if (i <= s.size()) {
		s.n[i] = x;
		s.wd(i).n = x;
	}
}
