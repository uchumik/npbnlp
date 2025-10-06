#include"negative_binomial.h"
#include<memory>
#include<iostream>

using namespace std;
using namespace npbnlp;

static shared_ptr<cmb_lookup> nb_lookup = nullptr;
negative_binomial::negative_binomial() {
	if (!nb_lookup)
		nb_lookup = shared_ptr<cmb_lookup>(new cmb_lookup);
}

negative_binomial::~negative_binomial() {
}

double negative_binomial::density(double p, int x, int y) {
	auto it = nb_lookup->find(make_pair(x, y));
	int cmb = 0;
	if (it != nb_lookup->end()) {
		cmb = it->second;
	} else {
		cmb = combination(y+x-1, y);
		(*nb_lookup)[make_pair(y+x-1,y)] = cmb;
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
