#ifndef NPBNLP_DA_DOUBLY_LINKED_LIST_H
#define NPBNLP_DA_DOUBLY_LINKED_LIST_H

#include<vector>
#include<unordered_map>
#include<optional>
#include<map>
#include<bits/stdc++.h>
#include<cstdio>
#include<algorithm>
#include<iostream>
#include<memory>
namespace npbnlp {
	template<class T>
		class node {
			public:
				node():parent(NULL) {
				}
				node(T i):id(i),parent(NULL) {
				}
				virtual ~node() {
				}
				std::vector<node<T> > sibling;
				T id;
				node* parent;
		};
	template<class T, class Hash = std::hash<T>, class Pred = std::equal_to<T> >
		class code {
			public:
				using code_serializer = void(*)(T&, FILE*);
				using code_deserializer = void(*)(T&, FILE*);
				code():_id(2) {
				}
				virtual ~code() {
				}
				long get(T x) const {
					const auto i = _c.find(x);
					if (i == _c.end())
						return -1;
					return i->second;
				}
				long operator[](T x) {
					const auto i = _c.find(x);
					if (i == _c.end()) {
						long id = _id;
						_c[x] = _id++;
						return id;
					} else {
						return i->second;
					}
				}
				long id() {
					return _id;
				}
				void set_terminal(T k) {
					if (_c.find(k) == _c.end() || _c[k] != 1)
						_c[k] = 1;
				}
				void save(FILE *fp, code_serializer f) {
					if (fwrite(&_id, sizeof(long),1, fp) != 1)
						throw "failed to write _id";
					int codes = _c.size();
					if (fwrite(&codes, sizeof(int), 1, fp) != 1)
						throw "failed to write code_size";
					for (auto& i : _c) {
						T k = i.first;
						f(k, fp);
						/*
						if (fwrite(&i.first, sizeof(T), 1, fp) != 1)
							throw "failed to write code";
							*/
						if (fwrite(&i.second, sizeof(long), 1, fp) != 1)
							throw "failed to write code id";
					}
				}
				void load(FILE *fp, code_deserializer f) {
					if (fread(&_id, sizeof(long), 1, fp) != 1)
						throw "failed to load _id";
					int codes = 0;
					if (fread(&codes, sizeof(int),1, fp) != 1)
						throw "failed to load code_size";
					for (auto i = 0; i < codes; ++i) {
						T k;
						long id;
						/*
						if (fread(&k, sizeof(T), 1, fp) != 1)
							throw "failed to load code";
							*/
						f(k, fp);
						if (fread(&id, sizeof(long), 1, fp) != 1)
							throw "failed to load code";
						_c[k] = id;
					}
				}
			protected:
				long _id;
				std::unordered_map<T, long, Hash, Pred> _c;
		};
	template<>
		class code<char> {
			public:
				using code_serializer = void(*)(char&, FILE*);
				using code_deserializer = void(*)(char&, FILE*);
				code() {
				};
				virtual ~code() {
				}
				long get(char x) const {
					long y = (unsigned char)x;
					return 1+y;
				}
				long operator[](char x) {
					long y = (unsigned char)x;
					return 1+y;
				}
				long id() {
					return 257;
				}
				void set_terminal(char k){}
				void save(FILE *fp, code_serializer f) {
				}
				
				void load(FILE *fp, code_deserializer f) {
				}
		};
	template<class T, class V>
		class da {
			public:
				using serializer = void(*)(V&, FILE*);
				using deserializer = void(*)(V&, FILE*);
				using code_serializer = void(*)(T&, FILE*);
				using code_deserializer = void(*)(T&, FILE*);
				using rst = std::vector<std::pair<int, long> >;
				da(T terminal):_nid(1),_terminal(terminal) {
					_base.resize(8192);
					_check.resize(8192);
					// init doubly linked list
					std::iota(_base.rbegin(), _base.rend(), -_base.size()+2);
					std::iota(_check.rbegin(), _check.rend(), -_check.size());
					_base[0] = 0; _base[1] = -_base.size();
					_check[0] = 0;
					_head = 1;
					_tail = _check.size()-1;
					_listsize = _base.size()-1;
					_c.set_terminal(terminal);
				}
				virtual ~da() {
				}
				int build(std::vector<std::vector<T> >& keys, std::vector<V>& values) {
					if (keys.size() != values.size())
						return 1; // error
					if (_make_code(keys))
						return 1;
					node<T> tree;
					_make_tree(keys, tree);
					_add_subtree(tree, 0);
					_value.resize(_nid);
					for (auto i = 0; i < (int)keys.size(); ++i) {
						long n = exactmatch(keys[i]);
						_value[n] = values[i];
					}
					return 0;
				}
				void insert(std::vector<T>& key, V& value) {
					long b = 0;
					for (auto& k : key) {
						if (!_add(b, k)) {
							b = _traverse(b, k);
							continue;
						}
						b = _modify(b, k);
					}
					long n = exactmatch(key);
					if (_value.size() < _nid) {
						_value.resize(_nid);
					}
					_value[n] = value;
				}
				void erase(std::vector<T>& key) {
					long b = 0;
					for (auto& k : key) {
						b = _traverse(b, k);
						if (b < 0)
							return; // not found
					}
					if (_base[b] < 0 && _check[b] >= 0) { // leaf
						_erased.emplace_back(-_base[b]);
						_value[-_base[b]] = V();
						auto p = _check[b]; // parent
						_link(b);
						while (p > 0) {
							std::vector<long> s;
							_get_sibling(p, s);
							if (s.empty()) {
								auto n = _check[p];
								_link(p);
								p = n;
							} else {
								break;
							}
						}
					}
				}
				std::optional<V> getval(long id) {
					if (id >= (long)_value.size() || id < 0 || std::find(_erased.begin(),_erased.end(),id) != _erased.end())
						return std::nullopt;
					return _value[id];
				}
				V& operator[](std::vector<T>& key) {
					long id = exactmatch(key);
					if (id < 0) { // not found
						V v;
						insert(key, v);
						id = exactmatch(key);
						return _value[id];
					} else {
						return _value[id];
					}
				}
				da::rst cp_search(std::vector<T>& query) {
					rst r;
					long b = 0;
					int len = 0;
					for (auto& i : query) {
						long c = _base[b] + 1;
						if (_check[c] == b && _base[c] < 0)
							r.emplace_back(std::make_pair(len, -_base[c]));
						b = _traverse(b, i);
						++len;
						if (b < 0)
							break;
					}
					return r;
				}
				long exactmatch(std::vector<T>& query) {
					long b = 0;
					for (auto& i : query) {
						b = _traverse(b, i);
						if (b < 0) {
							return -1;
						}
					}
					if (_base[b] < 0)
						return -_base[b];
					return
						-1;
				}
				int save(const char *file, serializer func1, code_serializer func2) {
					FILE *fp = NULL;
					if ((fp = fopen(file, "wb")) == NULL)
						return 1;
					if (fwrite(&_nid, sizeof(long), 1, fp) != 1)
						return 1;
					if (fwrite(&_head, sizeof(long), 1, fp) != 1)
						return 1;
					if (fwrite(&_tail, sizeof(long), 1, fp) != 1)
						return 1;
					if (fwrite(&_listsize, sizeof(long), 1, fp) != 1)
						return 1;
					long bsize = _base.size();
					if (fwrite(&bsize, sizeof(long), 1, fp) != 1)
						return 1;
					if (fwrite(&_base[0], sizeof(long), bsize, fp) != (size_t)bsize)
						return 1;
					long csize = _check.size();
					if (fwrite(&csize, sizeof(long), 1, fp) != 1)
						return 1;
					if (fwrite(&_check[0], sizeof(long), csize, fp) != (size_t)csize)
						return 1;
					long esize = _erased.size();
					if (fwrite(&esize, sizeof(long), 1, fp) != 1)
						return 1;
					if (fwrite(&_erased[0], sizeof(long), esize, fp) != (size_t)esize)
						return 1;
					long vsize = _value.size();
					if (fwrite(&vsize, sizeof(long), 1, fp) != 1)
						return 1;

					for (auto& i : _value) {
						// callback for serialization
						func1(i, fp);
					}
					_c.save(fp, func2);
					fclose(fp);
					return 0;
				}
				int load(const char *file, deserializer func1, code_deserializer func2) {
					FILE *fp = NULL;
					if ((fp = fopen(file, "rb")) == NULL)
						return 1;
					if (fread(&_nid, sizeof(long), 1, fp) != 1)
						return 1;
					if (fread(&_head, sizeof(long), 1, fp) != 1)
						return 1;
					if (fread(&_tail, sizeof(long), 1, fp) != 1)
						return 1;
					if (fread(&_listsize, sizeof(long), 1, fp) != 1)
						return 1;
					long bsize = 0;
					if (fread(&bsize, sizeof(long), 1, fp) != 1)
						return 1;
					_base.resize(bsize);
					if (fread(&_base[0], sizeof(long), bsize, fp) != (size_t)bsize)
						return 1;
					long csize = 0;
					if (fread(&csize, sizeof(long), 1, fp) != 1)
						return 1;
					_check.resize(csize);
					if (fread(&_check[0], sizeof(long), csize, fp) != (size_t)csize)
						return 1;
					long esize = _erased.size();
					if (fread(&esize, sizeof(long), 1, fp) != 1)
						return 1;
					_erased.resize(esize,0);
					if (fread(&_erased[0], sizeof(long), esize, fp) != (size_t)esize)
						return 1;
					long vsize = 0;
					if (fread(&vsize, sizeof(long), 1, fp) != 1)
						return 1;
					_value.resize(vsize);
					for (auto i = 0; i < vsize; ++i) {
						// callback for deserialization
						V v;
						func1(v, fp);
						_value[i] = v;
					}
					_c.load(fp, func2);
					fclose(fp);
					return 0;
				}
			protected:
				using code_t = code<T>;
				long _nid;
				code_t _c;
				std::vector<long> _base;
				std::vector<long> _check;
				std::vector<V> _value;
				std::vector<long> _erased;
				long _head;
				long _tail;
				long _listsize;
				T _terminal;

				long _traverse(long b, T c) {
					if (b >= (long)_base.size())
						return -1;
					long code = _c.get(c);
					if (code < 0)
						return -1;
					//long n = _base[b] + _c[c];
					long n = _base[b] + code;
					if (n >= (long)_base.size())
						return -1;
					if (_check[n] != b) {
						return -1;
					}
					return n;
				}

				long _check_acceptable(long c, long b, long m) {
					//long k = _base[b] + c + m;
					long k = c + m;
					if (!_is_empty(k))
						return 0;
					/*
					if (_check[k] >= 0 && _check[k] != b)
						return 0;
						*/
					return k;
				}

				long _check_acceptable(node<T>& n, long b, long m) {
					return _check_acceptable(_c[n.id], b, m);
				}

				void _get_sibling(long b, std::vector<long>& n) {
					auto id = _c.id();
					for (auto i = 1; i < id; ++i) {
						long k = _base[b] + i;
						if (k < _check.size() &&  _check[k] == b) {
							n.emplace_back(i);
						}
					}
				}

				long _xcheck(std::vector<long>& sibling, long b) {
					auto h = _head;
					long m = 0;
					while (1) {
						m = h - sibling[0];
						h = -_check[h];
						if (m < 0)
							continue;
						bool accept = true;
						for (auto& s : sibling) {
							if (!_check_acceptable(s, b, m)) {
								accept = false;
								break;
							}
						}
						if (!accept) {
							if (h == _tail)
								_extend();
							continue;
						}
						break;
					}
					return m;
				}

				long _xcheck(node<T>& subtree, long b) {
					auto h = _head;
					long m = 0;
					while (1) {
						m = h - _c[subtree.sibling[0].id];
						h = -_check[h];
						if (m < 0)
							continue;
						bool accept = true;
						for (auto& s : subtree.sibling) {
							if (!_check_acceptable(s, b, m)) {
								accept = false;
								break;
							}
						}
						if (!accept) {
							if (h == _tail)
								_extend();
							continue;
						}
						break;
					}
					return m;
				}

				bool check_list() {
					// forward check
					auto prev = 0;
					for (auto h = _head; h != _tail; h = -_check[h]) {
						if (_check[h] >= 0) {
							std::cout << "in forward check, found not empty node " << h << " base[" << h << "]:" << _base[h] << " check[" << h << "]:" << _check[h] << " head:" << _head << " tail:" << _tail << " size:" << _base.size() << std::endl;
							std::cout << "prev node:" << prev << " base[" << prev << "]:" << _base[prev] << " check[" << prev << "]:" << _check[prev] << std::endl;
							return false;
						} else {
							if (prev == h) {
								std::cout << "in forward, found loop. current:" << h << " prev:" << prev << " next:" << -_check[h] << std::endl;
							}
						}
						prev = h;
					}
					prev = 0;
					for (auto t = _tail; t != _head; t = -_base[t]) {
						if (_base[t] >= 0) {
							std::cout << "in backward check, found not empty node " << t << " base[" << t << "]:" << _base[t] << " check[" << t << "]:" << _check[t] << " head:" << _head << " tail:" << _tail << " size:" << _base.size() << std::endl;
							std::cout << "prev node:" << prev << " base[" << prev << "]:" << _base[prev] << " check[" << prev << "]:" << _check[prev] << std::endl;
							return false;
						}
						prev = t;
					}
					return true;
				}

				long _next_nid() {
					if (!_erased.empty()) {
						long n = _erased[_erased.size()-1];
						_erased.pop_back();
						return n;
					}
					return _nid++;
				}

				void _link(long b) {
					++_listsize;
					auto prev = -_base[_tail];
					auto next = -_check[_tail];
					_base[b] = -_tail;
					_check[b] = _check[_tail];
					_check[_tail] = -b;
					_tail = b;
				}

				void _delink(long b) {
					--_listsize;
					if ((double)_listsize/_base.size() < 0.1) {
						_extend();
					}
					auto prev = -_base[b];
					auto next = -_check[b];
					if (b == _head) {
						_head = next;
						_base[_head] = -prev;
						//_base[_head] = -prev;
					} else if (b == _tail) {
						_tail = prev;
						_check[_tail] = -next;
						//_check[_tail] = -next;
					} else {
						_base[next] = _base[b];
						_check[prev] = _check[b];
					}
					_base[b] = 0;
					_check[b] = 0;
				}

				int _add(long b, T k) {
					// found
					if (_traverse(b, k) > 0)
						return 0;
					long n = _base[b] + _c[k];
					if (_is_empty(n)) {
						_delink(n);
						_check[n] = b;
						if (_c[k] == 1) {
							_base[n] = -_next_nid();
						}
						return 0;
					}
					return 1;
				}

				void _extend() {
					auto old_head = _head;
					auto old_tail = _tail;
					auto offset = _base.size();
					_base.resize(2*_base.size());
					_check.resize(2*_check.size());
					std::iota(_base.rbegin(), _base.rend()-offset,-_base.size()+2);
					std::iota(_check.rbegin(), _check.rend()-offset,-_check.size());
					/*
					   _base[offset] = -_tail;
					   _tail = _check.size()-1;
					   _base[_head] = -_base.size();
					   */
					auto h = _head;
					auto t = _tail;
					_base[h] = -_check.size()+1;
					_check[t] = -_base.size();
					_check[_check.size()-1] = -h;
					_head = offset;
					_base[_head] = -_base.size();
					_listsize += offset;
				}

				bool _is_empty(long i) {
					while (i >= (long)_base.size()) {
						_extend();
					}

					return (_base[i] < 0 && _check[i] < 0);
				}

				void _modify_check(long f, long t) {
					// leaf node
					if (_base[f] < 0 && _check[f] >= 0) 
						return;
					auto id = _c.id();
					for (auto i = 1; i < id; ++i) {
						long n = _base[f] + i;
						if (n < _check.size() && _check[n] == f) {
							_check[n] = t;
						}
					}
				}

				long _modify(long b, T k) {
					long old = _base[b];
					std::vector<long> sibling;
					_get_sibling(b, sibling);
					sibling.emplace_back(_c[k]);
					_base[b] = _xcheck(sibling, b);
					sibling.pop_back();
					for (auto& c : sibling) {
						long n = _base[b] + c;
						long t = old + c;
						_delink(n);
						// add sibling
						_check[n] = b;
						_base[n] = _base[t];
						// fixed transition
						_modify_check(t, n);
						// remove old base and check
						_link(t); // add node t to list
					}
					_add(b, k);
					return _base[b] + _c[k];
				}


				int _add_subtree(node<T>& subtree, long b) {
					long m = _xcheck(subtree, b);
					_base[b] = m;
					for (auto& s : subtree.sibling) {
						long n = _base[b] + _c[s.id];
						_delink(n);
						_check[n] = b;
						if (s.id == _terminal && s.sibling.empty())
							_base[n] = -_next_nid();
					}
					for (auto& s : subtree.sibling) {
						long n = _base[b] + _c[s.id];
						if (_base[n] >= 0 && _check[n] == b)
							_add_subtree(s, n);
					}
					return 1;
				}

				// add node to key_tree in order of sorted keys
				void _add_node(std::vector<T>& key, node<T>& n, int depth) {
					if (depth == (int)key.size())
						return;
					T k = key[depth];
					if (n.sibling.empty() || n.sibling[n.sibling.size()-1].id != k) {
						node<T> m(k);
						m.parent = &n;
						n.sibling.emplace_back(m);
						_add_node(key, n.sibling[n.sibling.size()-1], depth+1);
					} else if (n.sibling[n.sibling.size()-1].id == k) {
						_add_node(key, n.sibling[n.sibling.size()-1], depth+1);
					}
				}

				// build tree using sorted keys
				int _make_tree(std::vector<std::vector<T> >& keys, node<T>& tree) {
					int depth = 0;
					for (auto& k : keys) {
						_add_node(k, tree, depth);
					}
					return 0;
				}

				int _make_code(std::vector<std::vector<T> >& keys) {
					std::map<T, long> count;
					for (auto& i : keys) {
						for (auto& j : i) {
							if (count.find(j) == count.end())
								count[j] = 1;
							else
								count[j]++;
						}
					}
					std::vector<std::pair<T, long> > d;
					for (const auto& i : count) {
						d.emplace_back(i);
					}
					std::sort(d.begin(), d.end(),
							[](const auto& a, const auto& b) {
							return a.second > b.second;
							});
					for (auto& i : d) {
						long id = _c[i.first];
					}
					return 0;
				}
		};
}

#endif
