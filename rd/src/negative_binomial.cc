#include"negative_binomial.h"
#include<memory>
#include<iostream>

using namespace std;
using namespace npbnlp;

// Per-thread combination cache. density() runs inside OpenMP parallel regions
// (phsmm::parse -> _type_prior, driven by the parallel tokenizing loops in
// ma.cc / sne.cc) and an unordered_map cannot be written concurrently, so this
// table must not be shared. Lazily built in density(): a thread_local cannot be
// initialised from the constructor, which only runs on the thread that creates
// the (static) negative_binomial object.
static thread_local shared_ptr<cmb_lookup> nb_lookup = nullptr;
negative_binomial::negative_binomial() {
}

negative_binomial::~negative_binomial() {
}

double negative_binomial::density(double p, int x, int y) {
	if (x <= 0)
		return 0;
	if (!nb_lookup)
		nb_lookup = shared_ptr<cmb_lookup>(new cmb_lookup);
	// the entry must be keyed by what we look it up with. Keying it by
	// (y+x-1, y) instead made every lookup miss, and worse, a later call with
	// x' = y+x-1 hit that entry and got C(y+x-1, y) where C(y+x'-1, y) was due.
	auto key = make_pair(x, y);
	auto it = nb_lookup->find(key);
	int cmb = 0;
	if (it != nb_lookup->end()) {
		cmb = it->second;
	} else {
		cmb = combination(y+x-1, y);
		(*nb_lookup)[key] = cmb;
	}
	double d = 1;
	for (auto i = 0; i < x; ++i) {
		d *= p;
	}
	for (auto i = 0; i < y; ++i) {
		d *= (1.-p);
	}
	return d*cmb;
}

double negative_binomial::cdf(double p, int x, int y) {
	double cdf = 0;
	if (y < 0)
		return cdf;
	for (auto k = 0; k <= y; ++k)
		cdf += density(p, x, k);
	return cdf;
}

int negative_binomial::combination(int x, int y) {
	vector<vector<int> > D(x+1, vector<int>(x+1, 0));
	for (auto i = 0; i < x+1; ++i) {
		D[i][0] = 1;
		D[i][i] = 1;
	}

	for (auto i = 1; i < x+1; ++i) {
		for (auto j = 1; j < i; ++j) {
			D[i][j] = D[i-1][j-1]+D[i-1][j];
		}
	}
	return D[x][y];
}
