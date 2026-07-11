#include"clattice.h"
#include"chartype.h"
#include"wordtype.h"
#include"chunktype.h"
#include<cstdlib>
#include<cstdio>
#ifdef _OPENMP
#include<omp.h>
#endif

using namespace std;
using namespace npbnlp;

static chunk eos;
static vector<int> bos(1, 0);

clattice::clattice(nio& f, int i) {
	int head = f.head[i];
	int tail = f.head[i+1];
	vector<type> wt;
	for (auto j = head; j < tail; ++j) {
		wt.push_back(wordtype::get((*f.raw)[j]));
	}
	c.resize(wt.size());
	k.resize(wt.size());
	shared_ptr<cid> dic = cid::create();
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (auto j = 0; j < (int)wt.size(); ++j) {
		type t = wt[j];
		for (auto k = j; k >= 0 && j-k < _chsize(t, wt[k]); --k) {
			bool is_break = true;
			switch ((*f.raw)[head+k][0]) {
				case 12539: // head = '・'
					is_break = false;
					;
				case 65292: // head = '，'
					is_break = false;
					;
					/*
				case 65285: // head = '％'
					is_break = false;
					;
					*/
				case 65286: // head = '＆'
					is_break = false;
					;
				case 65294: // head = '．'
					is_break = false;
					;
				case 46: // head = '.'
					is_break = false;
					;
				case 44: // head = ','
					is_break = false;
					;
				case 38: // head = '&'
					is_break = false;
					;
					/*
				case 37: // head = '%'
					is_break = false;
					;
					*/
					/*
				case 65311: // head = '？'
					;
				case 65281: // head = '！'
					;
				case 12290: // head = '。'
					;
				case 12289: // head = '、'
					;
				case 63: // head = '?'
					;
				case 33: // head = '!'
					;
					is_break = true;
					break;
					*/
				default:
					break;
			}
			switch ((*f.raw)[head+j][0]) {
				case 65285: // head = '％'
					is_break = false;
					;
				case 37: // head = '%'
					is_break = false;
					;
				default:
					break;
			}
			if (wt[k] == U_PUNC && k != j && is_break)
				break;
			else if (t == U_DIGIT && wt[j] == U_PUNC && is_break)
				break;
			chunk ch(*f.raw, head+k, 1+j-k);
			ch.id = (*dic)[ch];
			ch.type = t;
			c[j].push_back(ch);
		}
		k[j].resize(c[j].size());
	}
}

clattice::~clattice() {
}

chunk& clattice::ch(int i, int len) {
	if (i < 0 || i >= (int)c.size())
		return eos;
	if (len-1 >= (int)c[i].size())
		throw "invalid chunk size";
	return c[i][len-1];
}

chunk* clattice::cp(int i, int len) {
	if (i < 0 || i >= (int)c.size())
		return &eos;
	if (len-1 >= (int)c[i].size())
		throw "invalid chunk size";
	return &c[i][len-1];
}

int clattice::size(int i) {
	if (i < 0 || i >= (int)c.size())
		return 1;
	return c[i].size();
}

vector<int>::iterator clattice::begin(int i, int j) {
	if (i < 0 || i >= (int)k.size())
		return bos.begin();
	return k[i][j].begin();
}

vector<int>::iterator clattice::end(int i, int j) {
	if (i < 0 || i >= (int)k.size())
		return bos.end();
	return k[i][j].end();
}

int clattice::_chsize(type& t, type& u) {
	switch (u) {
		case U_HIRAGANA:
			if (t == U_KATAKANA)
				t = U_HIRA_KATA;
			else if (t == U_HANJI)
				t = U_HIRA_HANJI;
			else if (t == U_KATA_HANJI)
				t = U_HIRA_KATA_HANJI;
			else if (t == U_KATA_OR_HIRA)
				t = U_HIRAGANA;
			else if (t == U_PUNC)
				break;
			else if (t == U_DIGIT)
				break;
			else if (t == U_HIRA_HANJI || t == U_HIRA_KATA || t == U_HIRA_KATA_HANJI) {
			} else if (u != t)
				t = U_MISC;
			break;
		case U_KATAKANA:
			if (t == U_HIRAGANA)
				t = U_HIRA_KATA;
			else if (t == U_HANJI)
				t = U_KATA_HANJI;
			else if (t == U_HIRA_HANJI)
				t = U_HIRA_KATA_HANJI;
			else if (t == U_KATA_OR_HIRA)
				t = U_KATAKANA;
			else if (t == U_PUNC)
				break;
			else if (t == U_DIGIT)
				break;
			else if (t == U_HIRA_KATA || t == U_KATA_HANJI || t == U_HIRA_KATA_HANJI) {
			} else if (u != t)
				t = U_MISC;
			break;
		case U_HANJI:
			if (t == U_HIRAGANA)
				t = U_HIRA_HANJI;
			else if (t == U_KATAKANA)
				t = U_KATA_HANJI;
			else if (t == U_HIRA_KATA || t == U_KATA_OR_HIRA)
				t = U_HIRA_KATA_HANJI;
			else if (t == U_PUNC)
				break;
			else if (t == U_DIGIT)
				break;
			else if (t == U_HIRA_HANJI || t == U_KATA_HANJI || t == U_HIRA_KATA_HANJI) {
			} else if (u != t)
				t = U_MISC;
			break;
		case U_KATA_OR_HIRA:
			if (t != U_HIRAGANA && t != U_KATAKANA && t != U_HIRA_KATA && t != U_KATA_HANJI && t != U_HIRA_HANJI && t != U_HIRA_KATA_HANJI)
				t = U_MISC;
			else if (t == U_PUNC)
				break;
			else if (t == U_DIGIT)
				break;
			else if (u != t)
				t = U_MISC;
			break;
		case U_HIRA_KATA:
			if (t == U_HIRAGANA || t == U_KATAKANA)
				t = u;
			else if (t == U_HANJI)
				t = U_HIRA_KATA_HANJI;
			else if (t == U_PUNC)
				break;
			else if (t == U_DIGIT)
				break;
			else if (t != u)
				t = U_MISC;
			break;
		case U_HIRA_HANJI:
			if (t == U_HIRAGANA || t == U_HANJI)
				t = u;
			else if (t == U_KATAKANA)
				t = U_HIRA_KATA_HANJI;
			else if (t == U_PUNC)
				break;
			else if (t == U_DIGIT)
				break;
			else if (t != u)
				t = U_MISC;
			break;
		case U_KATA_HANJI:
			if (t == U_KATAKANA || t == U_HANJI)
				t = u;
			else if (t == U_HIRAGANA)
				t = U_HIRA_KATA_HANJI;
			else if (t == U_PUNC)
				break;
			else if (t == U_DIGIT)
				break;
			else if (t != u)
				t = U_MISC;
			break;
		case U_HIRA_KATA_HANJI:
			if (t == U_HIRAGANA || t == U_KATAKANA || t == U_HANJI)
				t = u;
			else if (t == U_PUNC)
				break;
			else if (t == U_DIGIT)
				break;
			else if (t != u)
				t = U_MISC;
			break;
		case U_LATIN:
			if (t == U_KATAKANA)
				break;
			else if (t == U_PUNC)
				t = U_LATIN;
			else if (t == U_DIGIT)
				break;
			else if (t != u)
				t = U_MISC;
			break;
		case U_DIGIT:
			t = U_DIGIT;
			/*
			if (t == U_PUNC || t == U_SYNBOL)
				t = U_DIGIT;
				*/
			//else if (t != u)
			//	t = U_MISC;
			break;
		case U_SYNBOL:
			if (t == U_LATIN || t == U_DIGIT)
			//if (t == U_LATIN || t == U_DIGIT || t == U_PUNC)
			//if (t == U_LATIN || t == U_DIGIT || t == U_KATAKANA)
				break;
			/*
			else if (t == U_HIRAGANA || t == U_KATAKANA || t == U_HANJI || t == U_HIRA_HANJI || t == U_HIRA_KATA || t == U_KATA_HANJI || t == U_HIRA_KATA_HANJI || t == U_KATA_OR_HIRA)
				return 1;
				*/
			else if (t != u)
				t = U_MISC;
			break;
		case U_PUNC:
			if (t == U_LATIN || t == U_DIGIT || t == U_KATAKANA)
				break;
			else if (t != U_MISC)
				t = U_PUNC;
			break;
		default:
			if (t != U_PUNC && t != u) {
			//if (t != u) {
				t = U_MISC;
			}

	}
	switch (t) {
		case U_ARABIC:
			return C_ARABIC;
		case U_GREEK:
			return C_GREEK;
		case U_HANGUL:
			return C_HANGUL;
		case U_HEBREW:
			return C_HEBREW;
		case U_LATIN:
			return C_LATIN;
		case U_MYANMAR:
			return C_MYANMAR;
		case U_THAI:
			return C_THAI;
		case U_DIGIT:
			return C_DIGIT;
		case U_HIRAGANA:
			return C_HIRAGANA;
		case U_KATAKANA:
			return C_KATAKANA;
		case U_HANJI:
			return C_HANJI;
		case U_HIRA_KATA:
			return C_HIRA_KATA;
		case U_HIRA_HANJI:
			return C_HIRA_HANJI;
		case U_KATA_HANJI:
			return C_KATA_HANJI;
		case U_HIRA_KATA_HANJI:
			return C_HIRA_KATA_HANJI;
		case U_PUNC:
			return C_PUNC;
		case U_SYNBOL:
			return C_SYNBOL;
		default:
			return C_MISC;
	}
}

clattice2::clattice2(nio& f, int i, std::vector<int>& chsize) {
	int head = f.head[i];
	int tail = f.head[i+1];
	vector<type> wt;
	for (auto j = head; j < tail; ++j) {
		wt.push_back(wordtype::get((*f.raw)[j]));
	}
	c.resize(wt.size());
	k.resize(wt.size());
	shared_ptr<cid> dic = cid::create();
	for (auto j = 0; j < (int)wt.size(); ++j) {
		chtype t = chunktype2::start(wt[j]);
		for (auto k = j; k >= 0 && j-k < chsize[t]; --k) {
			chunk ch(*f.raw, head+k, 1+j-k);
			ch.id = (*dic)[ch];
			ch.type = t;
			c[j].push_back(ch);
			t = chunktype2::transition(t, wt[k-1], wt[k]);
			if (t < 0)
				break;
		}
		k[j].resize(c[j].size());
	}
	// diagnostics (env gated): dump tokenizer word boundaries and every chunk
	// candidate span in absolute char offsets, for measuring gold-NE lattice
	// coverage (does the transition table / _clength drop NE spans?).
	if (getenv("NPBNLP_LATTICE_COVER")) {
		int nw = (int)wt.size();
		std::vector<int> cum(nw+1, 0);
		for (int p = 0; p < nw; ++p)
			cum[p+1] = cum[p] + (*f.raw)[head+p].len; // word char length
		fprintf(stderr, "tok %d", i);
		for (int p = 0; p <= nw; ++p)
			fprintf(stderr, " %d", cum[p]);
		fprintf(stderr, "\n");
		for (int j = 0; j < nw; ++j)
			for (auto& ch : c[j]) {
				int s = j - (ch.len - 1); // start word index
				fprintf(stderr, "cov %d %d-%d\n", i, cum[s], cum[j+1]);
			}
	}
}

clattice2::~clattice2() {
}

chunk& clattice2::ch(int i, int len) {
	if (i < 0 || i >= (int)c.size())
		return eos;
	if (len-1 >= (int)c[i].size())
		throw "invalid chunk size";
	return c[i][len-1];
}

chunk* clattice2::cp(int i, int len) {
	if (i < 0 || i >= (int)c.size())
		return &eos;
	if (len-1 >= (int)c[i].size())
		throw "invalid chunk size";
	return &c[i][len-1];
}

int clattice2::size(int i) {
	if (i < 0 || i >= (int)c.size())
		return 1;
	return c[i].size();
}

vector<int>::iterator clattice2::begin(int i, int j) {
	if (i < 0 || i >= (int)k.size())
		return bos.begin();
	return k[i][j].begin();
}

vector<int>::iterator clattice2::end(int i, int j) {
	if (i < 0 || i >= (int)k.size())
		return bos.end();
	return k[i][j].end();
}

