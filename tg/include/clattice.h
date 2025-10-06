#ifndef NPBNLP_CLATTICE_H
#define NPBNLP_CLATTICE_H

#include"nio.h"
#include"chartype.h"
#include"chunk.h"
#include<vector>
namespace npbnlp {
	enum chsize {
		C_MISC = 4,
		C_ARABIC = 5,
		C_LATIN = 5,
		C_THAI = 5,
		C_GREEK = 5,
		C_MYANMAR = 5,
		C_HEBREW = 5,
		C_DIGIT = 5,
		C_HANGUL = 5,
		C_HIRAGANA = 2,
		C_KATAKANA = 10,
		C_HIRA_KATA = 2,//5,
		C_HIRA_HANJI = 2,//5,
		C_KATA_HANJI = 7,
		C_HIRA_KATA_HANJI = 2,//7,
		C_HANJI = 7,
		C_KATA_OR_HIRA = 1,//5,
		C_SYNBOL = 5,
		C_PUNC = 1
	};
	class clattice {
		public:
			clattice(nio& f, int i);
			virtual ~clattice();
			chunk& ch(int i, int len);
			chunk* cp(int i, int len);
			int size(int i);
			std::vector<int>::iterator begin(int i, int j);
			std::vector<int>::iterator end(int i, int j);
			std::vector<std::vector<chunk> > c;
			std::vector<std::vector<std::vector<int> > > k;
			std::vector<std::vector<double> > prior;
			
		private:
			int _chsize(type& t, type& u);
	};
}

#endif
