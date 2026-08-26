#ifndef NPBNLP_VLATTICE_H
#define NPBNLP_VLATTICE_H

#include "io.h"
#include "sentence.h"
#include <vector>

namespace npbnlp {
	class vlattice {
		public:
			vlattice(sentence& s);
			vlattice(io& f, int i);
			virtual ~vlattice();
			word& wd(int i);
			int size(int i);
			void slice(int i, double u);
			double u(int i);
			std::vector<int>::iterator begin(int i);
			std::vector<int>::iterator end(int i);
			int order(int i);
			void set_order(int i, int n);
			std::vector<double> mu;
			std::vector<std::vector<int> > k;
			std::vector<int> n;
			// State whose log probability produced mu[i]. _build_lattice has to
			// keep exactly this state alive; _k can grow underneath it while the
			// lattice is being built, so recomputing min(_k, pos) there would
			// check a different state than the threshold came from.
			std::vector<int> cur;
			sentence s;
	};
}

#endif
