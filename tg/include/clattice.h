#ifndef NPBNLP_CLATTICE_H
#define NPBNLP_CLATTICE_H

#include"nio.h"
#include"chartype.h"
#include"chunk.h"
#include<vector>
namespace npbnlp {
	class clattice2 {
		public:
			clattice2(nio& f, int i, std::vector<int>& max_chunksize);
			virtual ~clattice2();
			chunk& ch(int i, int len);
			chunk* cp(int i, int len);
			int size(int i);
			std::vector<int>::iterator begin(int i, int j);
			std::vector<int>::iterator end(int i, int j);
			std::vector<std::vector<chunk> > c;
			std::vector<std::vector<std::vector<int> > > k;
			std::vector<std::vector<double> > prior;
			std::vector<std::vector<std::vector<double> > > emit; // emit[t][len-1][class]: root context emission cache
			// B-obs context factor: lctx[startpos][class] left factor, rctx[endpos][class] right factor
			std::vector<std::vector<double> > lctx;
			std::vector<std::vector<double> > rctx;
		private:

	};
}

#endif
