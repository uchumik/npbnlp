#ifndef NPBNLP_SUFFIX_DA_H
#define NPBNLP_SUFFIX_DA_H

#include"da2.h"

namespace npbnlp {
	/*
	template<>
		class code<unsigned int> {
			public:
				using code_serializer = void(*)(unsigned int&, FILE*);
				using code_deserializer = void(*)(unsigned int&, FILE*);
				code():_id(3), _t(3) {}
				virtual ~code(){}
				long operator[](unsigned int x) {
					if (x == 0)
						return _t+1;
					else if (x == _t)
						return 1;
					_id = std::max(_id, (long)x+1);
					return 1+x;
				}
				void set_terminal(unsigned int k) {
					_t = k;
				}
				long id() {
					return _id+1;
				}
				void save(FILE *fp, code_serializer f) {
					if (fwrite(&_id, sizeof(long), 1, fp) != 1)
						throw "failed to write _id";
					if (fwrite(&_t, sizeof(unsigned int), 1, fp) != 1)
						throw "failed to write _t";
				}
				void load(FILE *fp, code_deserializer f) {
					if (fread(&_id, sizeof(long), 1, fp) != 1)
						throw "failed to load _id";
					if (fread(&_t, sizeof(unsigned int), 1, fp) != 1)
						throw "failed to load _t";
				}
			protected:
				long _id;
				unsigned int _t;
		};
		*/
	template<class V>
		class sda : public da<unsigned int, V> {
			public:
				using rst = typename da<unsigned int, V>::rst;
				sda(unsigned int terminal):da<unsigned int, V>(terminal),_terminal(terminal) {
				}
				virtual ~sda() {
				}
				typename std::vector<V>::iterator begin() {
					return da<unsigned int,V>::_value.begin();
				}
				typename std::vector<V>::iterator end() {
					return da<unsigned int,V>::_value.end();
				}
				bool is_erased(long id) {
					if (id < 0 || id > std::numeric_limits<int32_t>::max())
						return false;
					return da<unsigned int,V>::_erased.count((int32_t)id) != 0;
				}
				rst cs_search(sentence& s, int i, int n) {
					rst r;
					long b = 0;
					int len = 0;
					for (auto j = i; j > i-n; --j) {
						long c = da<unsigned int,V>::_base[b] + 1;
						if (c >= 0 && c < (long)da<unsigned int,V>::_check.size() && da<unsigned int,V>::_check[c] == b && da<unsigned int,V>::_base[c] < 0)
							r.emplace_back(std::make_pair(len, -da<unsigned int,V>::_base[c]));
						b = da<unsigned int,V>::_traverse(b, s[j]);
						++len;
						if (b < 0)
							return r;
					}
					long c = da<unsigned int,V>::_base[b] + 1;
					if (c >= 0 && c < (long)da<unsigned int,V>::_check.size() && da<unsigned int,V>::_check[c] == b && da<unsigned int,V>::_base[c] < 0)
						r.emplace_back(std::make_pair(len,-da<unsigned int,V>::_base[c]));
					return r;
				}
				rst cs_search(word& w, int i, int n) {
					rst r;
					long b = 0;
					int len = 0;
					for (auto j = i; j > i-n; --j) {
						long c = da<unsigned int,V>::_base[b] + 1;
						if (c >= 0 && c < (long)da<unsigned int,V>::_check.size() && da<unsigned int,V>::_check[c] == b && da<unsigned int,V>::_base[c] < 0)
							r.emplace_back(std::make_pair(len, -da<unsigned int,V>::_base[c]));
						b = da<unsigned int,V>::_traverse(b, w[j]);
						++len;
						if (b < 0)
							return r;
					}
					long c = da<unsigned int,V>::_base[b] + 1;
					if (c >= 0 && c < (long)da<unsigned int,V>::_check.size() && da<unsigned int,V>::_check[c] == b && da<unsigned int,V>::_base[c] < 0)
						r.emplace_back(std::make_pair(len,-da<unsigned int,V>::_base[c]));
					return r;
				}
				rst cs_search(std::vector<int>& k, int i, int n) {
					rst r;
					long b = 0;
					int len = 0;
					for (auto j = i; j > i-n; --j) {
						long c = da<unsigned int,V>::_base[b] + 1;
						if (c >= 0 && c < (long)da<unsigned int,V>::_check.size() && da<unsigned int,V>::_check[c] == b && da<unsigned int,V>::_base[c] < 0)
							r.emplace_back(std::make_pair(len, -da<unsigned int,V>::_base[c]));
						int key = (j >= 0)? k[j]: 0; 
						b = da<unsigned int,V>::_traverse(b, key);
						++len;
						if (b < 0)
							return r;
					}
					long c = da<unsigned int,V>::_base[b] + 1;
					if (c >= 0 && c < (long)da<unsigned int,V>::_check.size() && da<unsigned int,V>::_check[c] == b && da<unsigned int,V>::_base[c] < 0)
						r.emplace_back(std::make_pair(len,-da<unsigned int,V>::_base[c]));
					return r;
				}
				long rexactmatch(sentence& s, int i, int n) {
					long b = 0;
					for (auto j = i; j > i-n; --j) {
						b = da<unsigned int,V>::_traverse(b, s[j]);
						if (b < 0)
							return -1;
					}
					b = da<unsigned int,V>::_traverse(b, _terminal); // terminal code
					if (b >= 0 && da<unsigned int,V>::_base[b] < 0)
						return -da<unsigned int,V>::_base[b];
					return -1;
				}
				long rexactmatch(word& w, int i, int n) {
					long b = 0;
					for (auto j = i; j > i-n; --j) {
						b = da<unsigned int,V>::_traverse(b, w[j]);
						if (b < 0)
							return -1;
					}
					b = da<unsigned int,V>::_traverse(b, _terminal);
					if (b >= 0 && da<unsigned int,V>::_base[b] < 0)
						return -da<unsigned int,V>::_base[b];
					return -1;
				}
				long rexactmatch(std::vector<int>& k, int i, int n) {
					long b = 0;
					for (auto j = i; j > i-n; --j) {
						int key = (j >= 0)? k[j]: 0; 
						b = da<unsigned int,V>::_traverse(b, key);
						if (b < 0)
							return -1;
					}
					b = da<unsigned int,V>::_traverse(b, _terminal);
					if (b >= 0 && da<unsigned int,V>::_base[b] < 0)
						return -da<unsigned int,V>::_base[b];
					return -1;
				}
				std::optional<V> getval(long id) {
					if (id >= (long)da<unsigned int,V>::_value.size() || id < 0 || da<unsigned int,V>::_erased.count((int32_t)id) != 0)
						return std::nullopt;
					return da<unsigned int,V>::_value[id];
				}
				V& val(long id) {
					if (id >= (long)da<unsigned int,V>::_value.size() || id < 0 || da<unsigned int,V>::_erased.count((int32_t)id) != 0) {
						std::cerr << "found invalid id:" << id << std::endl;
						throw "found invalid id at val(int id)";
					}
					return da<unsigned int,V>::_value[id];
				}
				void insert(sentence& s, int i, int n) {
					node<unsigned int> tree;
					make_subtree(tree, s, i, n);
					insert_subtree(tree, 0);
					//if (da<unsigned int,V>::_value.size() < da<unsigned int,V>::_nid) {
					while (da<unsigned int,V>::_value.size() < da<unsigned int,V>::_nid) {
						//da<unsigned int,V>::_value.resize(da<unsigned int,V>::_nid, V(n));
						da<unsigned int,V>::_value.emplace_back(V(n));
					}
				}
				void insert(word& w, int i, int n) {
					node<unsigned int> tree;
					make_subtree(tree, w, i, n);
					insert_subtree(tree, 0);
					//if (da<unsigned int,V>::_value.size() < da<unsigned int,V>::_nid) {
					while (da<unsigned int,V>::_value.size() < da<unsigned int,V>::_nid) {
						//da<unsigned int,V>::_value.resize(da<unsigned int,V>::_nid, V(n));
						da<unsigned int,V>::_value.emplace_back(V(n));
					}
				}
				void insert(std::vector<int>& k, int i, int n) {
					node<unsigned int> tree;
					make_subtree(tree, k, i, n);
					insert_subtree(tree, 0);
					while (da<unsigned int,V>::_value.size() < da<unsigned int,V>::_nid) {
						da<unsigned int,V>::_value.emplace_back(V(n));
					}
				}
				void erase(sentence& s, int i, int n) {
					long b = 0;
					for (auto j = i; j > i-n; --j) {
						b = da<unsigned int,V>::_traverse(b, s[j]);
						if (b < 0)
							return;
					}
					b = da<unsigned int,V>::_traverse(b, _terminal);
					if (b >= 0 && da<unsigned int,V>::_base[b] < 0 && da<unsigned int,V>::_check[b] >= 0) {
						da<unsigned int,V>::_erased.insert(-da<unsigned int,V>::_base[b]);
						da<unsigned int,V>::_value[-da<unsigned int,V>::_base[b]] = V(-1);
						auto p = da<unsigned int,V>::_check[b];
						da<unsigned int,V>::_link(b);
						while (p > 0) {
							std::vector<long> sib;
							da<unsigned int,V>::_get_sibling(p, sib);
							if (sib.empty()) {
								auto n = da<unsigned int,V>::_check[p];
								da<unsigned int,V>::_link(p);
								p = n;
							} else {
								break;
							}
						}
					}
				}
				void erase(word& w, int i, int n) {
					long b = 0;
					for (auto j = i; j > i-n; --j) {
						b = da<unsigned int,V>::_traverse(b, w[j]);
						if (b < 0)
							return;
					}
					b = da<unsigned int,V>::_traverse(b, _terminal);
					if (b >= 0 && da<unsigned int,V>::_base[b] < 0 && da<unsigned int,V>::_check[b] >= 0) {
						da<unsigned int,V>::_erased.insert(-da<unsigned int,V>::_base[b]);
						da<unsigned int,V>::_value[-da<unsigned int,V>::_base[b]] = V(-1);
						auto p = da<unsigned int,V>::_check[b];
						da<unsigned int,V>::_link(b);
						while (p > 0) {
							std::vector<long> sib;
							da<unsigned int,V>::_get_sibling(p, sib);
							if (sib.empty()) {
								auto n = da<unsigned int,V>::_check[p];
								da<unsigned int,V>::_link(p);
								p = n;
							} else {
								break;
							}
						}
					}
				}
				void erase(std::vector<int>& k, int i, int n) {
					long b = 0;
					for (auto j = i; j > i-n; --j) {
						int key = (j >= 0)? k[j]: 0; 
						b = da<unsigned int,V>::_traverse(b, key);
						if (b < 0)
							return;
					}
					b = da<unsigned int,V>::_traverse(b, _terminal);
					if (b >= 0 && da<unsigned int,V>::_base[b] < 0 && da<unsigned int,V>::_check[b] >= 0) {
						da<unsigned int,V>::_erased.insert(-da<unsigned int,V>::_base[b]);
						da<unsigned int,V>::_value[-da<unsigned int,V>::_base[b]] = V(-1);
						auto p = da<unsigned int,V>::_check[b];
						da<unsigned int,V>::_link(b);
						while (p > 0) {
							std::vector<long> sib;
							da<unsigned int,V>::_get_sibling(p, sib);
							if (sib.empty()) {
								auto n = da<unsigned int,V>::_check[p];
								da<unsigned int,V>::_link(p);
								p = n;
							} else {
								break;
							}
						}
					}
				}
				void insert_subtree(node<unsigned int>& t, long b) {
					for (auto& s : t.sibling) {
						if (da<unsigned int,V>::_add(b, s.id)) {
							da<unsigned int,V>::_modify(b, s.id);
						}
					}
					for (auto& s : t.sibling) {
						long n = da<unsigned int,V>::_base[b] + s.id;
						insert_subtree(s, n);
					}
				}
				void make_subtree(node<unsigned int>& t, sentence& s, int i, int n) {
					node<unsigned int> *p = &t;
					node<unsigned int> terminal(1);
					terminal.parent = p;
					p->sibling.emplace_back(terminal);
					for (auto j = i; j > i-n; --j) {
						unsigned int bytes[4];
						int count = encoded_edges(s[j], bytes);
						for (int z = 0; z < count; ++z) {
							node<unsigned int> x(bytes[z] + 2);
							x.parent = p;
							p->sibling.emplace_back(x);
							p = &p->sibling.back();
						}
						node<unsigned int> m(1);
						m.parent = p;
						p->sibling.emplace_back(m);
					}
				}
				void make_subtree(node<unsigned int>& t, word& w, int i, int n) {
					node<unsigned int> *p = &t;
					node<unsigned int> terminal(1);
					terminal.parent = p;
					p->sibling.emplace_back(terminal);
					for (auto j = i; j > i-n; --j) {
						unsigned int bytes[4];
						int count = encoded_edges(w[j], bytes);
						for (int z = 0; z < count; ++z) {
							node<unsigned int> x(bytes[z] + 2);
							x.parent = p;
							p->sibling.emplace_back(x);
							p = &p->sibling.back();
						}
						node<unsigned int> m(1);
						m.parent = p;
						p->sibling.emplace_back(m);
					}
				}
				void make_subtree(node<unsigned int>& t, std::vector<int>& k, int i, int n) {
					node<unsigned int> *p = &t;
					node<unsigned int> terminal(1);
					terminal.parent = p;
					p->sibling.emplace_back(terminal);
					for (auto j = i; j > i-n; --j) {
						int key = (j >= 0)? k[j]: 0;
						unsigned int bytes[4];
						int count = encoded_edges(key, bytes);
						for (int z = 0; z < count; ++z) {
							node<unsigned int> x(bytes[z] + 2);
							x.parent = p;
							p->sibling.emplace_back(x);
							p = &p->sibling.back();
						}
						node<unsigned int> m(1);
						m.parent = p;
						p->sibling.emplace_back(m);
					}
				}
			protected:
				template<class K>
				int encoded_edges(K key, unsigned int out[4]) {
					long code = da<unsigned int,V>::_c[key];
					if (code < 128) {
						out[0] = code;
						return 1;
					}
					if (code < 16384) {
						out[0] = 0x80 | (code >> 8);
						out[1] = code & 255;
						return 2;
					}
					if (code < 2097152) {
						out[0] = 0xc0 | (code >> 16);
						out[1] = (code >> 8) & 255;
						out[2] = code & 255;
						return 3;
					}
					out[0] = 0xe0 | (code >> 24);
					out[1] = (code >> 16) & 255;
					out[2] = (code >> 8) & 255;
					out[3] = code & 255;
					return 4;
				}
				unsigned int _terminal;
				//code<unsigned int> _c;
		};
}

#endif
