#ifndef NPBNLP_PC_CACHE_H
#define NPBNLP_PC_CACHE_H

#include<memory>
#include<shared_mutex>
#include<unordered_map>
#include<vector>
namespace npbnlp {
	class context;
	class lm;
	class pc_cache {
		public:
			pc_cache(context *root);
			int sample(const context *h, lm *m);
		private:
			struct dist {
				std::vector<double> p;
				double c;
			};
			std::vector<int> _k;
			std::unordered_map<const context*, std::shared_ptr<dist> > _dist;
			std::shared_mutex _mutex;
			std::shared_ptr<dist> _make(const context *h, lm *m) const;
	};
}
#endif
