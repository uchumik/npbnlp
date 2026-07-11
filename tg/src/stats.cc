#include"util.h"
#include"cio.h"
#include"nio.h"
#include"wordtype.h"
#include"chunktype.h"
#include<algorithm>
#include<set>

using namespace std;
using namespace npbnlp;

static int pos_id = 1;
static unordered_map<string, int> pos_to_index;
static unordered_map<int, string> index_to_pos;
static int label_id = 2;
static unordered_map<string, int> label_to_index;
static unordered_map<int, string> index_to_label;

struct vcmp {
	bool operator() (const vector<int>& a, const vector<int>& b) const {
		if (a.size() != b.size())
			return false;
		for (auto i = 0; a[i] == b[i] && i < (int)a.size(); ++i);
		return true;
	}
};

struct vhash {
	size_t operator() (const vector<int>& a) const {
		size_t seed = a.size();
		for (auto v : a) {
			auto x = v;
			x = ((x >> 16) ^ x) * 0x45d9f3b;
			x = ((x >> 16) ^ x) * 0x45d9f3b;
			x = ((x >> 16) ^ x);
			seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}
		return seed;
	}
};

struct scmp {
	bool operator() (const set<int>& a, const set<int>& b) const {
		for (auto i : a) {
			if (b.find(i) == b.end())
				return false;
		}
		return true;
	}
};

struct shash {
	size_t operator() (const set<int>& a) const {
		size_t seed = a.size();
		for (auto v : a) {
			auto x = v;
			x = ((x >> 16) ^ x) * 0x45d9f3b;
			x = ((x >> 16) ^ x) * 0x45d9f3b;
			x = ((x >> 16) ^ x);
			seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}
		return seed;
	}
};

using tvec_dist = unordered_map<vector<int>, int, vhash, vcmp>;
using tset_dist = unordered_map<set<int>, int, shash, scmp>;

int load_label(cio& file, vector<vector<word> >& words, vector<vector<string> >& labels) {
	int size = file.chunk->size();
	shared_ptr<wid> dic = wid::create();
	for (auto i = 0; i < size; ++i) {
		io& f = (*file.chunk)[i];
		vector<word> s;
		vector<string> lb;
		for (auto j = 0; j < (int)f.head.size()-1; ++j) {
			int head = f.head[j];
			int tail = f.head[j+1];
			word w(*f.raw);
			w.head = head;
			int p = util::find(9, *f.raw, head, tail);
			w.len = p - head;
			w.id = dic->index(w);
			int y = util::find(9, *f.raw, p+1, tail);
			string pos;
			for (auto k = p+1; k < y; ++k) {
				char buf[5] = {0};
				io::i2c((*f.raw)[k], buf);
				pos += buf;
			}
			if (pos_to_index.find(pos) == pos_to_index.end()) {
				pos_to_index[pos] = pos_id;
				index_to_pos[pos_id++] = pos;
			}
			w.pos = pos_to_index[pos];
			w.m.resize(w.len+1, 0);
			s.emplace_back(w);
			int l = util::find(9, *f.raw, y+1, tail);
			string pron;
			for (auto k = y+1; k < l; ++k) {
				char buf[5] = {0};
				io::i2c((*f.raw)[k], buf);
				pron += buf;
			}
			int m = util::find(9, *f.raw, l+1, tail);
			string label;
			for (auto k = l+1; k < m; ++k) {
				char buf[5] = {0};
				io::i2c((*f.raw)[k], buf);
				label += buf;
			}
			lb.emplace_back(label);
		}
		words.emplace_back(s);
		labels.emplace_back(lb);
	}
	return 0;
}

int chunking(vector<vector<word> >& words, vector<vector<string> >& labels, vector<nsentence>& corpus) {
	for (auto i = 0; i < (int)words.size(); ++i) {
		int chunk_head = 0;
		int chunk_len = 0;
		int chunk_k = 0;
		nsentence s;
		for (auto j = 0; j < (int)words[i].size(); ++j) {
			string& label = labels[i][j];
			if (label[0] == 'B') {
				string ne(label, 2, string::npos);
				if (label_to_index.find(ne) == label_to_index.end()) {
					label_to_index[ne] = label_id;
					index_to_label[label_id++] = ne;
				}
				chunk_head = j;
				chunk_len = 1;
				chunk_k = label_to_index[ne];
			} else if (label[0] == 'I') {
				++chunk_len;
			} else if (label[0] == 'O') {
				if (chunk_len > 0) {
					chunk c(words[i], chunk_head, chunk_len);
					c.k = chunk_k;
					c.type = chunktype2::get(c);
					s.c.emplace_back(c);
				}
				chunk_head = j;
				chunk_len = 1;
				chunk_k = 1;
			}
		}
		chunk c(words[i], chunk_head, chunk_len);
		c.k = chunk_k;
		c.type = chunktype2::get(c);
		s.c.emplace_back(c);
		s.n.resize(s.c.size()+1, 0);
		corpus.emplace_back(s);
	}
	return 0;
}

int stats(vector<nsentence>& corpus) {
	vector<double> chunk_len(label_id, 0);
	//vector<double> type_len(chartype::n, 0);
	vector<double> type_len(chunktype2::n, 0);
	int chunk_num = 0;
	double avg_chunk_len = 0;
	vector<vector<int> > chunk_len_dist(label_id, vector<int>()); 
	//vector<vector<int> > type_len_dist(chartype::n, vector<int>());
	vector<vector<int> > type_len_dist(chunktype2::n, vector<int>());
	vector<int> chunk_num_of_label(label_id, 0);
	//vector<int> chunk_num_of_type(chartype::n, 0);
	vector<int> chunk_num_of_type(chunktype2::n, 0);
	vector<int> chunk_num_of_type_change_for_label(label_id, 0);
	vector<int> chunk_num_of_character_type_change_for_label(label_id, 0);
	tvec_dist cdist;
	tvec_dist wdist;
	tset_dist uniq_ctdist;
	tset_dist uniq_wtdist;
	vector<vector<int> > ctransition(chartype::n, vector<int>(chartype::n, 0));
	vector<vector<int> > wtransition(chartype::n, vector<int>(chartype::n, 0));
	
	for (auto i = 0; i < (int)corpus.size(); ++i) {
		nsentence& s = corpus[i];
		chunk_num += s.size();
		for (auto j = 0; j < (int)s.size(); ++j) {
			chunk& c = s.ch(j);
			avg_chunk_len += c.len;
			chunk_len[c.k] += c.len;
			type_len[c.type] += c.len;
			chunk_num_of_label[c.k]++;
			chunk_num_of_type[c.type]++;
			if ((int)chunk_len_dist[c.k].size() < c.len+1) {
				chunk_len_dist[c.k].resize(c.len+1, 0);
			}
			if ((int)type_len_dist[c.type].size() < c.len+1) {
				type_len_dist[c.type].resize(c.len+1, 0);
			}
			vector<int> wtypev;
			vector<int> ctypev;
			set<int> cutypes;
			set<int> wutypes;
			int change = 0;
			int ct_change = 0;
			type tp = wordtype::get(c.wd(0));
			type cp = chartype::get(c.wd(0)[0]);
			for (auto k = 0; k < c.len; ++k) {
				type u = wordtype::get(c.wd(k));
				wutypes.insert(u);
				if (tp != u)
					++change;
				tp = u;
				wtypev.emplace_back(u);
				word& w = c.wd(k);
				for (auto l = 0; l < w.len; ++l) {
					ctypev.emplace_back(chartype::get(w[l]));
					type cu = chartype::get(w[l]);
					cutypes.insert(cu);
					if (cp != cu)
						++ct_change;
					cp = cu;
				}
			}
			// word type suffix transition
			for (auto k = c.len-1; k > 0; --k) {
				type s = wordtype::get(c.wd(k));
				type t = wordtype::get(c.wd(k-1));
				wtransition[s][t]++;
			}
			// char type transition
			vector<type> ctypes;
			for (auto k = 0; k < c.len; ++k) {
				word& w = c.wd(k);
				for (auto m = 0; m < w.len; ++m) {
					ctypes.emplace_back(chartype::get(w[m]));
				}
			}
			for (auto k = ctypes.size()-1; k > 0; --k) {
				type s = ctypes[k];
				type t = ctypes[k-1];
				ctransition[s][t]++;
			}
			if (ctypev.size() == 0) {
				cout << "this chunk has no character:" << c << endl;
				cout << "sentence:";
				for (auto& ch : s.c) {
					cout << ch << endl;
				}
			}
			chunk_len_dist[c.k][c.len]++;
			type_len_dist[c.type][c.len]++;
			chunk_num_of_type_change_for_label[c.k] += change;
			chunk_num_of_character_type_change_for_label[c.k] += change;
			if (cdist.find(ctypev) == cdist.end())
				cdist[ctypev] = 0;
			if (wdist.find(wtypev) == wdist.end())
				wdist[wtypev] = 0;
			if (uniq_ctdist.find(cutypes) == uniq_ctdist.end())
				uniq_ctdist[cutypes] = 0;
			if (uniq_wtdist.find(wutypes) == uniq_wtdist.end())
				uniq_wtdist[wutypes] = 0;
			cdist[ctypev]++;
			wdist[wtypev]++;
			uniq_ctdist[cutypes]++;
			uniq_wtdist[wutypes]++;
		}
	}
	/*
	   for (auto& l : chunk_len)
	   l /= chunk_num;
	   for (auto& l : type_len)
	   l /= chunk_num;
	   */
	avg_chunk_len /= chunk_num;
	cout << "avg_chunk_length\t" << avg_chunk_len << endl;
	cout << "avg chunk length of each label" << endl;
	for (auto l = 2; l < label_id; ++l) {
		cout << index_to_label[l] << "\t" << chunk_len[l]/chunk_num_of_label[l] << endl;
	}
	cout << "avg chunk length of each chunk type" << endl;
	for (auto l = 0; l < (int)type_len.size(); ++l) {
		if (type_len[l])
			cout << l << "\t" << type_len[l]/chunk_num_of_type[l] << endl;
	}
	cout << "chunk len dist" << endl;
	for (auto l = 2; l < label_id; ++l) {
		cout << index_to_label[l] << "\t";
		for (auto i = 0; i < (int)chunk_len_dist[l].size(); ++i) {
			if (chunk_len_dist[l][i])
				cout << i << ":" << chunk_len_dist[l][i] << "\t";
		}
		cout << endl;
	}
	cout << "type len dist" << endl;
	//for (auto l = 0; l < chartype::n; ++l) {
	for (auto l = 0; l < chunktype2::n; ++l) {
		cout << "type=" << l << "\t";
		for (auto i = 0; i < (int)type_len_dist[l].size(); ++i) {
			if (type_len_dist[l][i])
				cout << i << ":" << type_len_dist[l][i] << "\t";
		}
		cout << endl;
	}
	cout << "num of type change" << endl;
	for (auto l = 2; l < label_id; ++l) {
		cout << index_to_label[l] << "\t" << (double)chunk_num_of_type_change_for_label[l]/chunk_num_of_label[l] << endl;
	}
	cout << "num of character type change" << endl;
	for (auto l = 2; l < label_id; ++l) {
		cout << index_to_label[l] << "\t" << (double)chunk_num_of_character_type_change_for_label[l]/chunk_num_of_label[l] << endl;
	}
	vector<pair<int, vector<int> > > ctmp;
	cout << "character type distributions" << endl;
	for (auto& i : cdist) {
		ctmp.emplace_back(make_pair(i.second, i.first));
		/*
		   cout << "ctype_vec:";
		   for (auto& j : i.first) {
		   cout << j << " ";
		   }
		   cout << "\t#:" << i.second << endl;
		   */
	}
	sort(ctmp.begin(), ctmp.end(),
			[](const auto& a, const auto& b) {
			return a.first > b.first;
			});
	for (auto& i : ctmp) {
		cout << "#" << i.first << "\t";
		for (auto& j : i.second) {
			cout << j << " ";
		}
		cout << endl;
	}

	vector<pair<int, vector<int> > > wtmp;
	cout << "word type distributions" << endl;
	for (auto& i: wdist) {
		wtmp.emplace_back(make_pair(i.second, i.first));
		/*
		   cout << "wtype_vec:";
		   for (auto& j : i.first) {
		   cout << j << " ";
		   }
		   cout << "\t#:" << i.second << endl;
		   */
	}
	sort(wtmp.begin(), wtmp.end(),
			[](const auto& a, const auto& b) {
			return a.first > b.first;
			});
	for (auto& i : wtmp) {
		cout << "#" << i.first << "\t";
		for (auto& j :i.second) {
			cout << j << " ";
		}
		cout << endl;
	}

	vector<pair<int, set<int> > > cutypes;
	for (auto& i : uniq_ctdist) {
		cutypes.emplace_back(make_pair(i.second, i.first));
	}
	sort(cutypes.begin(), cutypes.end(),
			[](const auto& a, const auto& b) {
			return a.first > b.first;
			});
	double cum_ctype_prob = 0;
	cout << "character uniq_type distributions" << endl;
	for (auto& i : cutypes) {
		cum_ctype_prob += (double)i.first/chunk_num;
		cout << "#" << i.first << "(" << cum_ctype_prob << ")" << "\t";
		for (auto& j: i.second) {
			cout << j << " ";
		}
		cout << endl;
	}

	vector<pair<int, set<int> > > wutypes;
	for (auto& i : uniq_wtdist) {
		wutypes.emplace_back(make_pair(i.second, i.first));
	}
	sort(wutypes.begin(), wutypes.end(),
			[](const auto& a, const auto& b) {
			return a.first > b.first;
			});
	double cum_wtype_prob = 0;
	cout << "word uniq_type distributions" << endl;
	for (auto& i : wutypes) {
		cum_wtype_prob += (double)i.first/chunk_num;
		cout << "#" << i.first << "(" << cum_wtype_prob << ")" << "\t";
		for (auto& j: i.second) {
			cout << j << " ";
		}
		cout << endl;
	}
	cout << "word type transition" << endl;
	for (auto i = 0; i < wtransition.size(); ++i) {
		for (auto j = 0; j < wtransition.size(); ++j) {
			if (wtransition[i][j] > 0)
				cout << i << "," << j << ":" << wtransition[i][j] << " ";
		}
		cout << endl;
	}
	cout << "character type transition" << endl;
	for (auto i = 0; i < ctransition.size(); ++i) {
		for (auto j = 0; j < ctransition.size(); ++j) {
			if (ctransition[i][j] > 0)
				cout << i << "," << j << ":" << ctransition[i][j] << " ";
		}
		cout << endl;
	}
	return 0;
}

int main(int argc, char **argv) {
	cio labeled_corpus(*(argv+1));
	vector<vector<word> > words;
	vector<vector<string> > labels;
	load_label(labeled_corpus, words, labels);
	vector<nsentence> ne_corpus;
	chunking(words, labels, ne_corpus);
	stats(ne_corpus);
	return 0;
}
