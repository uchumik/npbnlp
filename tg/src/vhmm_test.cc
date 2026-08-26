#include "vhmm.h"
#include "util.h"
#include "vhdp_context.h"
#include <iostream>

using namespace std;
using namespace npbnlp;

class test_vhmm: public vhmm {
	public:
		test_vhmm(int wn=1):vhmm(3,1,4,3) {
			set_word_ngram(wn);
		}
		void check() {
			check_pos(static_cast<vhdp_context*>(_pos->h()), true);
			for (auto& x : *_word) {
				if (x->h()->c()!=0 || x->h()->t()!=0)
					throw "vhmm emission round-trip failed";
			}
		}
		void check_pos(vhdp_context *c, bool=false) {
			if (c->c()!=0 || c->t()!=0 || c->stop()!=c->a() || c->pass()!=c->b()) {
				cerr << "counts c=" << c->c() << " t=" << c->t()
				     << " stop=" << c->stop() << " pass=" << c->pass() << endl;
				throw "vhmm transition round-trip failed";
			}
			for (auto& x:c->child())
				check_pos(static_cast<vhdp_context*>(x.second.get()));
		}
};

int main() {
	try {
		io f("/tmp/wo046_vhmm_test.txt");
		vector<sentence> c;
		util::store_sentences(f,c);
		// Both emission orders: at 1 the customer always sits at the root, at 2 and
		// 3 it sits at a drawn depth that remove() has to find again from the table.
		for (int wn : {1, 2, 3}) {
			test_vhmm m(wn);
			for (int i=0;i<(int)c.size();++i) m.init(c[i],i);
			// One sweep in the shape vtg uses: take a sentence out, resample it,
			// seat it again.  add() then runs against depths drawn under a state
			// assignment that init() never saw.
			for (int i=0;i<(int)c.size();++i) {
				m.remove(c[i],i);
				m.cache_max();
				c[i]=m.sample(c[i]);
				m.add(c[i],i);
			}
			for (int i=0;i<(int)c.size();++i) m.remove(c[i],i);
			m.check();
			cout<<"vhmm add/remove round-trip (word_ngram "<<wn<<"): PASS"<<endl;
		}
	} catch (const char *e) {
		cerr << e << endl;
		return 1;
	}
	return 0;
}
