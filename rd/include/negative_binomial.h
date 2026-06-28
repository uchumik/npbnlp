#ifndef NPBNLP_NBIN_H
#define NPBNLP_NBIN_H

#include<random>
#include<unordered_map>
namespace npbnlp {
	struct ivhash {
		size_t operator() (const std::pair<int, int>& x) const {
			return x.second*100+x.first;
		}
	};
	struct ivcmp {
		bool operator() (const std::pair<int, int>& a, const std::pair<int, int>& b) const {
			return (a.first == b.first && a.second == b.second);
		}
	};
	using cmb_lookup = std::unordered_map<std::pair<int,int>, int, ivhash, ivcmp>;
	class negative_binomial {
		public:
			negative_binomial();
			virtual ~negative_binomial();
			double density(double p, int x, int y);
			double cdf(double p, int x, int y);
		private:
			int combination(int x, int y);
	};
}
#endif
