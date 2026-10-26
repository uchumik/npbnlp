#include"pc_cache.h"
#include"context.h"
#include"lm.h"
#include"math.h"
#include"rd.h"
#include<mutex>

using namespace std;
namespace npbnlp {
	pc_cache::pc_cache(context *root):_k(root->keys()) {
	}

	shared_ptr<pc_cache::dist> pc_cache::_make(const context *h, lm *m) const {
		vector<double> t;
		for (int k : _k)
			t.push_back(m->lp(k, h));
		double z = 0;
		for (int i = 0; i < (int)t.size(); ++i)
			z = math::lse(z, t[i], (i == 0));
		shared_ptr<dist> d = make_shared<dist>();
		d->p.resize(t.size(), 0);
		d->c = 0;
		for (int i = 0; i < (int)t.size(); ++i) {
			d->p[i] = std::exp(t[i]-z);
			d->c += d->p[i];
		}
		return d;
	}

	int pc_cache::sample(const context *h, lm *m) {
		shared_ptr<dist> d;
		{
			shared_lock<shared_mutex> lock(_mutex);
			auto it = _dist.find(h);
			if (it != _dist.end())
				d = it->second;
		}
		if (!d) {
			d = _make(h, m);
			unique_lock<shared_mutex> lock(_mutex);
			auto it = _dist.emplace(h, d).first;
			d = it->second;
		}
		return _k[rd::draw(d->c, d->p)];
	}
}
