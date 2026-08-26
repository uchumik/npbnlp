#include "vhdp.h"
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

using namespace npbnlp;
using namespace std;

static bool close_enough(double x, double y) { return fabs(x-y) < 1e-12; }

static void check_round_trip(vhdp_context *c) {
	if (c->c()!=0 || c->t()!=0 || c->stop()!=c->a() || c->pass()!=c->b())
		throw "round-trip context invariant failed";
	for (int k=0; k<=2000; ++k)
		if (c->stick_stop(k)!=0 || c->stick_n(k)!=0)
			throw "round-trip stick invariant failed";
	for (auto& p : c->child())
		check_round_trip(static_cast<vhdp_context*>(p.second.get()));
}

static void check_max(vhdp& model, vhdp_context *root, vhdp_context *c, int k_max) {
	for (int k=0; k<=k_max; ++k)
		for (int order=c->n(); order<model.n(); ++order)
			if (model.lp(k,c)>model.max_lp(k,order)+1e-12) throw "max upper bound failed";
	for (auto& p : c->child())
		check_max(model,root,static_cast<vhdp_context*>(p.second.get()),k_max);
	for (int k=k_max+1; k<=k_max+10; ++k)
		for (int order=0; order<model.n(); ++order)
			if (!close_enough(model.max_lp(k,order),model.lp(k,root)))
				throw "max_lp out-of-range fallback failed";
}

static void check_state_zero(vhdp& model, vhdp_context *c) {
	if (!(model.pr(0, c) > 0.) || !(model.lp(0, c) > -100.))
		throw "state 0 probability failed";
	for (auto& p : c->child())
		check_state_zero(model, static_cast<vhdp_context*>(p.second.get()));
}

static double probability_sum(vhdp& model, context *c, int K) {
	double sum=0.;
	for (int k=0; k<=K; ++k) sum += model.pr(k, c);
	return sum;
}

int main() {
	try {
		vhdp model(4);
		vector<pair<int, context*> > added;
		mt19937 r(17);
		for (int i=0; i<1000; ++i) {
			int order=1+(r()%4), state=r()%80;
			context *c=model.h();
			for (int j=1; j<order; ++j) c=static_cast<vhdp_context*>(c)->make(1+(r()%20));
			model.add(state,c); added.push_back({state,c});
		}
		for (auto i=added.rbegin(); i!=added.rend(); ++i) model.remove(i->first,i->second);
		check_round_trip(static_cast<vhdp_context*>(model.h()));
		cout << "1 add/remove round-trip (all nodes and sticks): PASS" << endl;

		// 8: order stop/pass counts are per selected context, not per CRP table.
		vhdp order_model(4);
		vhdp_context *order_root=static_cast<vhdp_context*>(order_model.h());
		vhdp_context *order_d1=static_cast<vhdp_context*>(order_root->make(11));
		vhdp_context *order_d2=static_cast<vhdp_context*>(order_d1->make(22));
		vector<pair<int, context*> > order_added;
		for (int i=0; i<7; ++i) { order_model.add(i%3, order_root); order_added.push_back({i%3,order_root}); }
		for (int i=0; i<5; ++i) { order_model.add(i%3, order_d1); order_added.push_back({i%3,order_d1}); }
		for (int i=0; i<3; ++i) { order_model.add(i%3, order_d2); order_added.push_back({i%3,order_d2}); }
		cout << "8 mixed-order counts root(stop/pass)=" << order_root->stop() << "/" << order_root->pass()
			 << " depth1=" << order_d1->stop() << "/" << order_d1->pass()
			 << " depth2=" << order_d2->stop() << "/" << order_d2->pass() << endl;
		// Assert the semantics, not arithmetic: every touched node must accumulate
		// stop, every ancestor must accumulate pass, and the CRP walking up has to
		// count there too (the prototype runs _context_add() at each node it
		// reaches).  Hard-coded totals here just encode whichever rule was in force.
		if (order_root->stop() <= order_root->a() || order_d1->stop() <= order_d1->a() || order_d2->stop() <= order_d2->a())
			throw "mixed-order stop counts did not accumulate";
		if (order_root->pass() <= order_root->b() || order_d1->pass() <= order_d1->b())
			throw "mixed-order pass counts did not accumulate";
		if (order_root->stop() <= 1+7 || order_d1->stop() <= 1+5)
			throw "CRP propagation did not reach the ancestors' order counts";
		for (auto i=order_added.rbegin(); i!=order_added.rend(); ++i) order_model.remove(i->first,i->second);
		check_round_trip(order_root);
		cout << "8 mixed-order stop/pass and add/remove reset: PASS" << endl;

		context *root=model.h();
		cout << setprecision(17);
		for (int K : {0,10,100,2000}) {
			double sum=probability_sum(model, root, K);
			cout << "2 probability normalization K=" << K << " sum=" << sum << endl;
			if (K==2000 && sum<=.99) throw "normalization failed";
		}
		cout << "2 probability normalization: PASS" << endl;

		double previous=1.;
		for (int k=1;k<=2000;++k) {
			// pr(k+1) materializes the definition's pr_pass(k) cache entry.
			model.pr(k+1,root);
			double p=static_cast<vhdp_context*>(root)->cache_pass(k);
			if (p == -DBL_MAX) throw "pr_pass was not materialized";
			if (p>previous+1e-12) throw "pr_pass monotonicity failed";
			previous=p;
		}
		cout << "3 pr_pass monotonicity: PASS" << endl;

		const int k_max=100;
		model.cache_max(k_max);
		check_max(model,static_cast<vhdp_context*>(root),static_cast<vhdp_context*>(root),k_max);
		cout << "4 max_lp upper bound for all deeper orders: PASS" << endl;

		vector<unsigned int> shallow_tokens={1,32,4,32,2};
		sentence shallow_sentence(shallow_tokens,0,shallow_tokens.size());
		for (int i=0; i<shallow_sentence.size(); ++i) shallow_sentence.wd(i).pos=i+1;
		shallow_sentence.wd(1).pos=20001;
		vector<unsigned int> deep_tokens={1,32,3,32,2,32,4};
		sentence deep_sentence(deep_tokens,0,deep_tokens.size());
		for (int i=0; i<deep_sentence.size(); ++i) deep_sentence.wd(i).pos=i+1;
		deep_sentence.wd(1).pos=20003;
		deep_sentence.wd(2).pos=20002;
		vhdp_context *shallow=static_cast<vhdp_context*>(static_cast<vhdp_context*>(root)->make(20002));
		vhdp_context *deeper=static_cast<vhdp_context*>(shallow->make(20003));
		vhdp_context *shallow_other=static_cast<vhdp_context*>(static_cast<vhdp_context*>(root)->make(20001));
		vector<tuple<sentence*,int,int,vhdp_context*> > find_exist_cases={{&shallow_sentence,2,4,shallow_other},{&deep_sentence,3,4,deeper}};
		for (auto q : find_exist_cases) {
			context *c=model.find_exist(*get<0>(q),get<1>(q),get<2>(q));
			if (!c || c->n()>=get<2>(q)) throw "find_exist did not return a shallow context";
			for (int k=0; k<=k_max; ++k)
				if (model.max_lp(k,get<2>(q)-1)+1e-12<model.lp(k,c)) throw "find_exist shallow upper bound failed";
		}
		if (model.find_exist(shallow_sentence,2,4)!=shallow_other || model.find_exist(deep_sentence,3,4)!=deeper)
			throw "find_exist shallow context selection failed";
		cout << "9 find_exist shallow-node deep-order bound: PASS" << endl;

		const char *file="vhdp_test.model"; model.save(file); vhdp loaded; loaded.load(file);
		for(int k=0;k<=100;++k) if(!close_enough(model.lp(k,root),loaded.lp(k,loaded.h()))) throw "save/load failed";
		remove(file); cout << "5 save/load round-trip: PASS" << endl;

		if (!(model.pr(0, root)>0.) || !(model.lp(0, root)>-100.)) throw "empty state 0 failed";
		vector<pair<int, context*> > state_zero_added;
		for (int i=0; i<100; ++i) {
			int state=i%12;
			context *c=root;
			if (i%3) c=static_cast<vhdp_context*>(root)->make(1+i%5);
			model.add(state, c);
			state_zero_added.push_back({state,c});
		}
		check_state_zero(model, static_cast<vhdp_context*>(root));
		for (auto i=state_zero_added.rbegin(); i!=state_zero_added.rend(); ++i)
			model.remove(i->first, i->second);
		check_round_trip(static_cast<vhdp_context*>(root));
		cout << "6 state 0 remains live and round-trip is clean: PASS" << endl;

		vhdp nonempty(4);
		vector<pair<int, context*> > nonempty_added;
		for (int i=0; i<1000; ++i) {
			int state=i%40;
			context *c=nonempty.h();
			if (i%2) c=static_cast<vhdp_context*>(nonempty.h())->make(1+i%8);
			nonempty.add(state, c);
			nonempty_added.push_back({state,c});
		}
		double root_sum=probability_sum(nonempty, nonempty.h(), 2000);
		context *depth1=static_cast<vhdp_context*>(nonempty.h())->find(2);
		if (!depth1) throw "depth1 context was not created";
		double depth1_sum=probability_sum(nonempty, depth1, 2000);
		cout << "7 non-empty root normalization K=2000 sum=" << root_sum << endl;
		cout << "7 non-empty depth1 normalization K=2000 sum=" << depth1_sum << endl;
		if (root_sum<=.99 || depth1_sum<=.99) throw "non-empty normalization failed";
		cout << "7 non-empty probability normalization: PASS" << endl;
	} catch (const char *e) { cerr << e << endl; return 1; }
	return 0;
}
