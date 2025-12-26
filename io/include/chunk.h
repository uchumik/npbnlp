#ifndef NPBNLP_CHUNK_H
#define NPBNLP_CHUNK_H

#include"io.h"
#include"nio.h"
#include<unordered_map>
#include<iostream>
#include<mutex>

namespace npbnlp {
	class chunk {
		public:
			chunk();
			chunk(std::vector<word>& d);
			chunk(std::vector<word>& d, int head, int len);
			chunk(const chunk& c);
			chunk(chunk&& c);
			chunk& operator=(const chunk& c);
			chunk& operator=(chunk&& c) noexcept;
			virtual ~chunk();
			void save(FILE *fp);
			void load(FILE *fp, std::vector<word>& d);
			int operator[](int i) const;
			word& wd(int i) const;
			int k;
			int head;
			int len;
			int id;
			int type;
			std::vector<int> n;
			friend std::ostream& operator<<(std::ostream& os, const chunk& c) {
				for (auto i = 0; i < c.len; ++i) {
					const word& w = c.wd(i);
					for (auto j = 0; j < w.len; ++j) {
						char buf[5] = {0};
						io::i2c(w[j], buf);
						os << buf;
					}
					if (i < c.len-1)
						os << " ";
				}
				os << ":" << c.type;
				if (c.id > 0)
					os << ":" << c.id;
				if (c.k > 0)
					os << ":" << c.k;
				return os;
			}
		protected:
			std::vector<word> *_doc;
		private:
	};
	struct chash {
		size_t operator() (const chunk& c) const {
			size_t seed = c.len;
			for (int i = 0; i < c.len; ++i) {
				auto x = c[i];
				x = ((x >> 16) ^ x) * 0x45d9f3b;
				x = ((x >> 16) ^ x) * 0x45d9f3b;
				x = (x >> 16) ^ x;
				seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			}
			return seed;
		}
		/*
		size_t operator() (const chunk& c) const {
			size_t id = 19780211;
			for (int i = 0; i < c.len; ++i)
				id = id*37*c[i];
				//id = id*37*c.wd(i).pos;
			return id;
		}
		*/
	};
	struct ccmp {
		bool operator() (const chunk& a, const chunk& b) const {
			if (a.len != b.len)
				return false;
			int i = 0;
			for (; a[i] == b[i] && i < a.len; ++i);
			//return true;
			//for (; a.wd(i).pos == b.wd(i).pos && i < a.len; ++i);
			return (i == a.len);
		}
	};
	class cid {
		using cdic = std::unordered_map<chunk, int, chash, ccmp>;
		public:
			cid(const cid& c) = delete;
			cid& operator=(const cid& c) = delete;
			static std::shared_ptr<cid> create();
			int operator[](chunk& c);
			int index(chunk& c);
			void remove(chunk& c);
			void save(const char *file);
			bool load(const char *file);
		private:
			static std::shared_ptr<cid> _idx;
			static std::mutex _mutex;
			static std::shared_ptr<std::vector<word> > _word;
			static std::shared_ptr<std::vector<unsigned int> > _letter;
			static std::shared_ptr<std::vector<word> > _load_word;
			static std::shared_ptr<std::vector<unsigned int> > _load_letter;
			int _id;
			cdic _index;
			std::vector<int> _misn;
			void _store(FILE *fp);
		protected:
			cid(int id);
	};
}
#endif
