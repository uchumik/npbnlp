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
#include<cstdint>
#include<sys/stat.h>
#include<unistd.h>
namespace npbnlp {
	int compress(const char *input, const char *output);
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
				int32_t get(T x) const {
					const auto i = _c.find(x);
					if (i == _c.end())
						return -1;
					return i->second;
				}
				int32_t operator[](T x) {
					const auto i = _c.find(x);
					if (i == _c.end()) {
						if (_id >= (1 << 29))
							throw "da symbol code exceeds 29 bits";
						int32_t id = _id;
						_c[x] = _id++;
						return id;
					} else {
						return i->second;
					}
				}
				int32_t id() {
					return _id;
				}
				void set_terminal(T k) {
					if (_c.find(k) == _c.end() || _c[k] != 1)
						_c[k] = 1;
				}
				void save(FILE *fp, code_serializer f) {
					if (fwrite(&_id, sizeof(int32_t),1, fp) != 1)
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
						if (fwrite(&i.second, sizeof(int32_t), 1, fp) != 1)
							throw "failed to write code id";
					}
				}
				void load(FILE *fp, code_deserializer f) {
					if (fread(&_id, sizeof(int32_t), 1, fp) != 1)
						throw "failed to load _id";
					int codes = 0;
					if (fread(&codes, sizeof(int),1, fp) != 1)
						throw "failed to load code_size";
					for (auto i = 0; i < codes; ++i) {
						T k;
						int32_t id;
						/*
						if (fread(&k, sizeof(T), 1, fp) != 1)
							throw "failed to load code";
							*/
						f(k, fp);
						if (fread(&id, sizeof(int32_t), 1, fp) != 1)
							throw "failed to load code";
						_c[k] = id;
					}
				}
			protected:
				int32_t _id;
				std::unordered_map<T, int32_t, Hash, Pred> _c;
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
					node<int32_t> tree;
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
						long code = _c.get(k);
						if (code < 0)
							code = _c[k];
						int32_t bytes[4];
						int nbytes = code == 1 ? 1 : _encode(static_cast<uint32_t>(code), bytes);
						if (code == 1)
							bytes[0] = 1;
						for (int i = 0; i < nbytes; ++i) {
							long e = code == 1 ? 1 : bytes[i] + 2;
							if (!_add(b, e)) {
								b = _base[b] + e;
								continue;
							}
							b = _modify(b, e);
						}
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
					const char magic[8] = {'D', 'A', '2', 'E', '2', 'I', '3', '2'};
					const int32_t version = 1;
					if (fwrite(magic, sizeof(magic), 1, fp) != 1)
						return 1;
					if (fwrite(&version, sizeof(version), 1, fp) != 1)
						return 1;
					if (fwrite(&_nid, sizeof(int32_t), 1, fp) != 1)
						return 1;
					if (fwrite(&_head, sizeof(int32_t), 1, fp) != 1)
						return 1;
					if (fwrite(&_tail, sizeof(int32_t), 1, fp) != 1)
						return 1;
					if (fwrite(&_listsize, sizeof(int32_t), 1, fp) != 1)
						return 1;
					int32_t bsize = _base.size();
					if (fwrite(&bsize, sizeof(int32_t), 1, fp) != 1)
						return 1;
					if (fwrite(&_base[0], sizeof(int32_t), bsize, fp) != (size_t)bsize)
						return 1;
					int32_t csize = _check.size();
					if (fwrite(&csize, sizeof(int32_t), 1, fp) != 1)
						return 1;
					if (fwrite(&_check[0], sizeof(int32_t), csize, fp) != (size_t)csize)
						return 1;
					int32_t esize = _erased.size();
					if (fwrite(&esize, sizeof(int32_t), 1, fp) != 1)
						return 1;
					if (esize > 0 && fwrite(&_erased[0], sizeof(int32_t), esize, fp) != (size_t)esize)
						return 1;
					int32_t vsize = _value.size();
					if (fwrite(&vsize, sizeof(int32_t), 1, fp) != 1)
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
					char magic[8];
					const char expected[8] = {'D', 'A', '2', 'E', '2', 'I', '3', '2'};
					if (fread(magic, sizeof(magic), 1, fp) != 1)
						return 1;
					if (std::memcmp(magic, expected, sizeof(magic)) != 0) {
						fclose(fp);
						throw "legacy da format: convert with da_convert";
					}
					int32_t version = 0;
					if (fread(&version, sizeof(version), 1, fp) != 1 || version != 1)
						return 1;
					if (fread(&_nid, sizeof(int32_t), 1, fp) != 1)
						return 1;
					if (fread(&_head, sizeof(int32_t), 1, fp) != 1)
						return 1;
					if (fread(&_tail, sizeof(int32_t), 1, fp) != 1)
						return 1;
					if (fread(&_listsize, sizeof(int32_t), 1, fp) != 1)
						return 1;
					int32_t bsize = 0;
					if (fread(&bsize, sizeof(int32_t), 1, fp) != 1)
						return 1;
					_base.resize(bsize);
					if (fread(&_base[0], sizeof(int32_t), bsize, fp) != (size_t)bsize)
						return 1;
					int32_t csize = 0;
					if (fread(&csize, sizeof(int32_t), 1, fp) != 1)
						return 1;
					_check.resize(csize);
					if (fread(&_check[0], sizeof(int32_t), csize, fp) != (size_t)csize)
						return 1;
					int32_t esize = _erased.size();
					if (fread(&esize, sizeof(int32_t), 1, fp) != 1)
						return 1;
					_erased.resize(esize,0);
					if (esize > 0 && fread(&_erased[0], sizeof(int32_t), esize, fp) != (size_t)esize)
						return 1;
					int32_t vsize = 0;
					if (fread(&vsize, sizeof(int32_t), 1, fp) != 1)
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
				friend int compress(const char *input, const char *output);
				using code_t = code<T>;
				int32_t _nid;
				code_t _c;
				std::vector<int32_t> _base;
				std::vector<int32_t> _check;
				std::vector<V> _value;
				std::vector<int32_t> _erased;
				int32_t _head;
				int32_t _tail;
				int32_t _listsize;
				T _terminal;

				long _traverse(long b, T c) {
					if (b < 0 || b >= (long)_base.size())
						return -1;
					long code = _c.get(c);
					if (code < 0)
						return -1;
					int32_t bytes[4];
					int nbytes = code == 1 ? 1 : _encode(static_cast<uint32_t>(code), bytes);
					if (code == 1)
						bytes[0] = 1;
					for (int i = 0; i < nbytes; ++i) {
						long edge = code == 1 ? 1 : bytes[i] + 2;
						long n = _base[b] + edge;
						if (n < 0 || n >= (long)_base.size() || _check[n] != b)
							return -1;
						b = n;
					}
					return b;
				}

				int _encode(uint32_t x, int32_t out[4]) {
					if (x < 128) {
						out[0] = x;
						return 1;
					}
					if (x < 16384) {
						out[0] = 0x80 | (x >> 8);
						out[1] = x & 255;
						return 2;
					}
					if (x < 2097152) {
						out[0] = 0xc0 | (x >> 16);
						out[1] = (x >> 8) & 255;
						out[2] = x & 255;
						return 3;
					}
					out[0] = 0xe0 | (x >> 24);
					out[1] = (x >> 16) & 255;
					out[2] = (x >> 8) & 255;
					out[3] = x & 255;
					return 4;
				}

				long _check_acceptable(long c, long b, long m) {
					if (m < 0)
						return 0;
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

				long _check_acceptable(node<int32_t>& n, long b, long m) {
					return _check_acceptable(n.id, b, m);
				}

				void _get_sibling(long b, std::vector<long>& n) {
					for (long i = 1; i <= 258; ++i) {
						long k = _base[b] + i;
						if (k < _check.size() &&  _check[k] == b) {
							n.emplace_back(i);
						}
					}
				}

				/*
				long _xcheck(std::vector<long>& sibling, long b) {
					auto h = _tail;
					long m = 0;
					while (1) {
						m = h - sibling[0];
						h = -_base[h];
						if (m < 0) {
							if (h == _head) {
								_extend();
								h = _tail;
							}
							continue;
						}
						bool accept = true;
						for (auto& s: sibling) {
							if (!_check_acceptable(s, b, m)) {
								accept = false;
								break;
							}
						}
						if (!accept) {
							if (h == _head) {
								_extend();
								h = _tail;
							}
							continue;
						}
						break;
					}
					return m;
				}
				*/

				long _xcheck(std::vector<long>& sibling, long b) {
					auto h = _head;
					long m = 0;
					while (1) {
						m = h - sibling[0];
						//auto c = h;
						h = -_check[h];
						/*
						if (m < 0) {
							if (c == _tail) {
								_extend();
								h = _head;
							}
							continue;
						}
						*/
						bool accept = true;
						for (auto& s : sibling) {
							if (!_check_acceptable(s, b, m)) {
								accept = false;
								break;
							}
						}
						if (!accept) {
							if (h == _tail) {
								_extend();
								h = _head;
							}
							continue;
						}
						break;
					}
					return m;
				}

				/*
				long _xcheck(node<T>& subtree, long b) {
					auto h = _tail;
					long m = 0;
					while (1) {
						m = h - _c[subtree.sibling[0].id];
						h = -_base[h];
						if (m < 0) {
							if (h == _head) {
								_extend();
								h = _tail;
							}
							continue;
						}
						bool accept = true;
						for (auto& s : subtree.sibling) {
							if (!_check_acceptable(s, b, m)) {
								accept = false;
								break;
							}
						}
						if (!accept) {
							if (h == _head) {
								_extend();
								h = _tail;
							}
							continue;
						}
						break;
					}
					return m;
				}
				*/

				long _xcheck(node<int32_t>& subtree, long b) {
					auto h = _head;
					long m = 0;
					while (1) {
						m = h - subtree.sibling[0].id;
						//auto c = h;
						h = -_check[h];
						/*
						if (m < 0) {
							if (c == _tail) {
								_extend();
								h = _head;
							}
							continue;
						}
						*/
						bool accept = true;
						for (auto& s : subtree.sibling) {
							if (!_check_acceptable(s, b, m)) {
								accept = false;
								break;
							}
						}
						if (!accept) {
							if (h == _tail) {
								_extend();
								h = _head;
							}
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
					if (_nid >= std::numeric_limits<int32_t>::max())
						throw "da exceeds int32 index";
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

				int _add(long b, long e) {
					// found
					long n = _base[b] + e;
					if (n >= 0 && n < (long)_base.size() && _check[n] == b)
						return 0;
					if (_is_empty(n)) {
						_delink(n);
						_check[n] = b;
						if (e == 1) {
							_base[n] = -_next_nid();
						}
						return 0;
					}
					return 1;
				}

				void _extend() {
					if (_base.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max()) / 2)
						throw "da exceeds int32 index";
					auto old_head = _head;
					auto old_tail = _tail;
					auto offset = _base.size();
					_base.resize(2*_base.size());
					_check.resize(2*_check.size());
					std::iota(_base.rbegin(), _base.rend()-offset,-_base.size()+2);
					std::iota(_check.rbegin(), _check.rend()-offset,-_check.size());
					// link unoccupied list to tail node
					/*
					_base[offset] = -_tail;
					_check[_tail] = -offset;
					_tail = _check.size()-1;
					_listsize += offset;
					*/

					// link unoccupied list to head node
					auto h = _head;
					auto t = _tail;
					_head = offset;
					_base[_head] = -_base.size();
					_base[h] = -_check.size()+1;
					_check[_check.size()-1] = -h;
					_check[_tail] = -_check.size();
					_listsize += offset;
				}

				bool _is_empty(long i) {
					if (i < 2 || i >= (long)_base.size())
						return false;
					return (_base[i] < 0 && _check[i] < 0);
				}
				/*
				bool _is_empty(long i) {
					//if (i >= (long)_base.size())
					//	return false;
					while (i >= (long)_base.size()) {
						_extend();
					}

					return (_base[i] < 0 && _check[i] < 0);
				}
				*/

				void _modify_check(long f, long t) {
					// leaf node
					if (_base[f] < 0 && _check[f] >= 0) 
						return;
					for (long i = 1; i <= 258; ++i) {
						long n = _base[f] + i;
						if (n < _check.size() && _check[n] == f) {
							_check[n] = t;
						}
					}
				}

				long _modify(long b, long e) {
					long old = _base[b];
					std::vector<long> sibling;
					_get_sibling(b, sibling);
					sibling.emplace_back(e);
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
					_add(b, e);
					return _base[b] + e;
				}


				int _add_subtree(node<int32_t>& subtree, long b) {
					long m = _xcheck(subtree, b);
					_base[b] = m;
					for (auto& s : subtree.sibling) {
						long n = _base[b] + s.id;
						_delink(n);
						_check[n] = b;
						if (s.id == 1)
							_base[n] = -_next_nid();
					}
					for (auto& s : subtree.sibling) {
						long n = _base[b] + s.id;
						if (_base[n] >= 0 && _check[n] == b)
							_add_subtree(s, n);
					}
					return 1;
				}

				void _compact_free_list() {
					long last_occupied = 1;
					for (long i = 2; i < (long)_check.size(); ++i) {
						if (_check[i] >= 0)
							last_occupied = i;
					}
					long new_size = std::max<long>(8192, last_occupied + 259);
					long old_size = static_cast<long>(_check.size());
					_base.resize(new_size);
					_check.resize(new_size);
					for (long i = old_size; i < new_size; ++i) {
						_base[i] = -1;
						_check[i] = -1;
					}
					std::vector<long> free_cells;
					for (long i = 1; i < new_size; ++i) {
						if (_check[i] < 0)
							free_cells.emplace_back(i);
					}
					for (size_t i = 0; i < free_cells.size(); ++i) {
						long current = free_cells[i];
						long previous = i == 0 ? new_size : free_cells[i - 1];
						long next = i + 1 == free_cells.size() ? new_size : free_cells[i + 1];
						_base[current] = -previous;
						_check[current] = -next;
					}
					_head = free_cells.front();
					_tail = free_cells.back();
					_listsize = free_cells.size();
				}


				std::vector<int32_t> _expand_key(const std::vector<T>& key) {
					std::vector<int32_t> transition;
					for (auto& symbol : key) {
						long code = _c.get(symbol);
						if (code == 1) {
							transition.emplace_back(1);
							continue;
						}
						if (code < 0)
							throw "missing DA symbol code";
						int32_t bytes[4];
						int length = _encode(static_cast<uint32_t>(code), bytes);
						for (int i = 0; i < length; ++i)
							transition.emplace_back(bytes[i] + 2);
					}
					return transition;
				}

				void _add_node(std::vector<int32_t>& key, node<int32_t>& n, int depth) {
					if (depth == (int)key.size())
						return;
					int32_t k = key[depth];
					if (n.sibling.empty() || n.sibling[n.sibling.size()-1].id != k) {
						node<int32_t> m(k);
						m.parent = &n;
						n.sibling.emplace_back(m);
						_add_node(key, n.sibling[n.sibling.size()-1], depth+1);
					} else if (n.sibling[n.sibling.size()-1].id == k) {
						_add_node(key, n.sibling[n.sibling.size()-1], depth+1);
					}
				}

				int _make_tree(std::vector<std::vector<T> >& keys, node<int32_t>& tree) {
					std::vector<std::vector<int32_t> > transitions;
					transitions.reserve(keys.size());
					for (auto& key : keys)
						transitions.emplace_back(_expand_key(key));
					std::sort(transitions.begin(), transitions.end());
					int depth = 0;
					for (auto& transition : transitions)
						_add_node(transition, tree, depth);
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
	inline void validate_da_arrays(const std::vector<int32_t>& base, const std::vector<int32_t>& check, int32_t nid) {
		if (base.size() < 2 || base.size() != check.size() || nid < 1)
			throw "invalid da arrays";
		for (size_t i = 0; i < check.size(); ++i) {
			if (check[i] >= static_cast<int32_t>(check.size()))
				throw "invalid da parent reference";
			if (check[i] >= 0 && base[i] < 0 && (base[i] == INT32_MIN || -base[i] >= nid))
				throw "invalid da value id";
		}
		std::function<void(int32_t)> visit = [&](int32_t parent) {
			if (parent < 0 || parent >= static_cast<int32_t>(base.size()) || base[parent] < 0 || base[parent] >= static_cast<int32_t>(base.size()))
				throw "invalid da base";
			for (int32_t edge = 1; edge <= 257; ++edge) {
				int64_t child = static_cast<int64_t>(base[parent]) + edge;
				if (child < 0 || child >= static_cast<int64_t>(check.size()) || check[child] != parent)
					continue;
				if (edge != 1)
					visit(static_cast<int32_t>(child));
			}
		};
		visit(0);
	}

	inline void verify_compacted_arrays(const std::vector<int32_t>& old_base, const std::vector<int32_t>& old_check, const std::vector<int32_t>& new_base, const std::vector<int32_t>& new_check, int32_t nid) {
		validate_da_arrays(old_base, old_check, nid);
		validate_da_arrays(new_base, new_check, nid);
		int64_t old_reachable = 0;
		int64_t new_reachable = 0;
		std::function<void(int32_t, int32_t)> compare = [&](int32_t old_parent, int32_t new_parent) {
			++old_reachable;
			std::vector<int32_t> old_edges;
			std::vector<int32_t> new_edges;
			for (int32_t edge = 1; edge <= 257; ++edge) {
				int64_t old_child = static_cast<int64_t>(old_base[old_parent]) + edge;
				if (old_child >= 0 && old_child < static_cast<int64_t>(old_check.size()) && old_check[old_child] == old_parent)
					old_edges.emplace_back(edge);
				int64_t new_child = static_cast<int64_t>(new_base[new_parent]) + edge;
				if (new_child >= 0 && new_child < static_cast<int64_t>(new_check.size()) && new_check[new_child] == new_parent)
					new_edges.emplace_back(edge);
			}
			if (old_edges != new_edges)
				throw "compressed da transition self-check failed";
			for (int32_t edge : old_edges) {
				int32_t old_child = old_base[old_parent] + edge;
				int32_t new_child = new_base[new_parent] + edge;
				if (edge == 1) {
					++old_reachable;
					if (old_base[old_child] != new_base[new_child])
						throw "compressed da value self-check failed";
				} else {
					compare(old_child, new_child);
				}
			}
		};
		std::function<void(int32_t)> count_new = [&](int32_t parent) {
			++new_reachable;
			for (int32_t edge = 1; edge <= 257; ++edge) {
				int64_t child = static_cast<int64_t>(new_base[parent]) + edge;
				if (child < 0 || child >= static_cast<int64_t>(new_check.size()) || new_check[child] != parent)
					continue;
				if (edge == 1)
					++new_reachable;
				else
					count_new(static_cast<int32_t>(child));
			}
		};
		compare(0, 0);
		count_new(0);
		if (old_reachable != new_reachable)
			throw "compressed da reachability self-check failed";
	}

	inline int compress(const char *input, const char *output) {
		if (std::string(input) == std::string(output))
			throw "input and output paths must differ";
		struct stat in_stat;
		struct stat out_stat;
		if (stat(input, &in_stat) != 0)
			throw "failed to stat input";
		if (stat(output, &out_stat) == 0 && in_stat.st_dev == out_stat.st_dev && in_stat.st_ino == out_stat.st_ino)
			throw "input and output paths must differ";
		FILE *source = fopen(input, "rb");
		if (source == nullptr)
			throw "failed to open input";
		FILE *target = nullptr;
		std::string temporary = std::string(output) + ".XXXXXX";
		bool created = false;
		try {
			const char magic[8] = {'D', 'A', '2', 'E', '2', 'I', '3', '2'};
			char input_magic[8];
			int32_t version = 0;
			if (fread(input_magic, 1, sizeof(input_magic), source) != sizeof(input_magic))
				throw "truncated da header";
			if (std::memcmp(input_magic, magic, sizeof(magic)) != 0)
				throw "legacy da format: convert with da_convert";
			if (fread(&version, sizeof(version), 1, source) != 1)
				throw "truncated da version";
			if (version != 1)
				throw "unsupported da format";
			int32_t nid = 0;
			int32_t head = 0;
			int32_t tail = 0;
			int32_t listsize = 0;
			if (fread(&nid, sizeof(nid), 1, source) != 1 || fread(&head, sizeof(head), 1, source) != 1 || fread(&tail, sizeof(tail), 1, source) != 1 || fread(&listsize, sizeof(listsize), 1, source) != 1)
				throw "truncated da header";
			int32_t bsize = 0;
			if (fread(&bsize, sizeof(bsize), 1, source) != 1)
				throw "truncated da base size";
			if (bsize < 2 || static_cast<uint64_t>(bsize) * sizeof(int32_t) > static_cast<uint64_t>(in_stat.st_size))
				throw "invalid da base size";
			std::vector<int32_t> base(bsize);
			if (fread(base.data(), sizeof(int32_t), bsize, source) != static_cast<size_t>(bsize))
				throw "truncated da base array";
			int32_t csize = 0;
			if (fread(&csize, sizeof(csize), 1, source) != 1)
				throw "truncated da check size";
			if (csize != bsize || static_cast<uint64_t>(csize) * sizeof(int32_t) > static_cast<uint64_t>(in_stat.st_size))
				throw "invalid da check size";
			std::vector<int32_t> check(csize);
			if (fread(check.data(), sizeof(int32_t), csize, source) != static_cast<size_t>(csize))
				throw "truncated da check array";
			da<int32_t, int32_t> compacted(0);

			std::function<void(int32_t, int32_t)> rebuild = [&](int32_t old_parent, int32_t new_parent) {
					std::vector<long> siblings;
				for (int32_t edge = 1; edge <= 257; ++edge) {
					int64_t child64 = static_cast<int64_t>(base[old_parent]) + edge;
					if (child64 < 0 || child64 >= csize || check[child64] != old_parent)
						continue;
					siblings.emplace_back(edge);
				}
				if (siblings.empty())
					return;
				compacted._base[new_parent] = compacted._xcheck(siblings, new_parent);
				for (int32_t edge : siblings) {
					int32_t old_child = base[old_parent] + edge;
					int32_t new_child = compacted._base[new_parent] + edge;
					compacted._delink(new_child);
					compacted._check[new_child] = new_parent;
					if (edge == 1) {
						if (base[old_child] >= 0)
							throw "invalid da terminal";
						compacted._base[new_child] = base[old_child];
					} else {
						compacted._base[new_child] = 0;
					}
				}
				for (int32_t edge : siblings) {
					if (edge != 1)
						rebuild(base[old_parent] + edge, compacted._base[new_parent] + edge);
				}
			};
			rebuild(0, 0);
			compacted._nid = nid;
			compacted._compact_free_list();
			verify_compacted_arrays(base, check, compacted._base, compacted._check, nid);
			std::vector<char> name(temporary.begin(), temporary.end());
			name.emplace_back('\0');
			int fd = mkstemp(name.data());
			if (fd < 0)
				throw "failed to open output";
			temporary = name.data();
			created = true;
			mode_t mask = umask(0);
			umask(mask);
			if (fchmod(fd, 0666 & ~mask) != 0) {
				close(fd);
				throw "failed to set output permissions";
			}
			target = fdopen(fd, "wb");
			if (target == nullptr) {
				close(fd);
				throw "failed to open output";
			}
			if (fwrite(magic, 1, sizeof(magic), target) != sizeof(magic) || fwrite(&version, sizeof(version), 1, target) != 1 || fwrite(&compacted._nid, sizeof(int32_t), 1, target) != 1 || fwrite(&compacted._head, sizeof(int32_t), 1, target) != 1 || fwrite(&compacted._tail, sizeof(int32_t), 1, target) != 1 || fwrite(&compacted._listsize, sizeof(int32_t), 1, target) != 1)
				throw "failed to write compressed DA";
			int32_t compacted_size = compacted._base.size();
			if (fwrite(&compacted_size, sizeof(compacted_size), 1, target) != 1 || fwrite(compacted._base.data(), sizeof(int32_t), compacted._base.size(), target) != compacted._base.size() || fwrite(&compacted_size, sizeof(compacted_size), 1, target) != 1 || fwrite(compacted._check.data(), sizeof(int32_t), compacted._check.size(), target) != compacted._check.size())
				throw "failed to write compressed DA";
			std::vector<char> buffer(1024 * 1024);
			size_t read_size = 0;
			while ((read_size = fread(buffer.data(), 1, buffer.size(), source)) > 0) {
				if (fwrite(buffer.data(), 1, read_size, target) != read_size)
					throw "failed to write compressed DA";
			}
			if (ferror(source) || fflush(target) != 0)
				throw "failed to copy DA values";
			if (fclose(target) != 0)
				throw "failed to close compressed DA";
			target = nullptr;
			fclose(source);
			source = nullptr;
			if (rename(temporary.c_str(), output) != 0)
				throw "failed to rename compressed DA";
		} catch (...) {
			if (target != nullptr)
				fclose(target);
			if (source != nullptr)
				fclose(source);
			if (created)
				remove(temporary.c_str());
			throw;
		}
		return 0;
	}
}

#endif
