#ifndef NPBNLP_NPHSMM_H
#define NPBNLP_NPHSMM_H

#include"nio.h"
#include"hpyp.h"
#include"vpyp.h"
#include"clattice.h"
#include"vtable.h"
#include<functional>
#include<memory>
#include<vector>
namespace npbnlp {
	class nphsmm {
		public:
			nphsmm();
			nphsmm(int n, int m, int l, int k);
			virtual ~nphsmm();
			virtual nsentence sample(nio& f, int i);
			virtual nsentence sample(nio& f, int i, nsentence *cur);
			virtual nsentence parse(nio& f, int i);
			virtual void add(nsentence& s);
			virtual void remove(nsentence& s);
			virtual void init(nsentence& s);
			virtual void set(int v, int k);
			virtual int n();
			virtual int m();
			virtual int l();
			virtual void slice(double a, double b);
			virtual void estimate(int iter);
			virtual void poisson_correction(int n = 100);
			virtual void save(const char *file);
			virtual void load(const char *file);
		protected:
			int _n;
			int _m;
			int _l;
			int _k;
			int _v;
			int _K;
			double _a;
			double _b;
			std::shared_ptr<hpyp> _class;
			std::shared_ptr<std::vector<std::shared_ptr<hpyp> > > _chunk;
			std::shared_ptr<std::vector<std::shared_ptr<hpyp> > > _word;
			std::shared_ptr<std::vector<std::shared_ptr<vpyp> > > _letter;
			std::mutex _mutex;
			void _forward(clattice& l, int i, const context *c, const context *z, chunk& ch, int k, chunk& prev, int q, vt& a, vt& b, int n, bool unk, bool not_exsit);
			void _backward(clattice& l, int i, const context *c, const context *z, chunk& ch, int k, chunk& prev, int q, double& lpr, vt& b, int n, bool unk, bool not_exist);
			nsentence _minfer(nio& f, int i, bool best, nsentence *cur);
			void _mfill(clattice& l, vt& dp, vt& am, vt& bos, vt& trm);
			void _mchain(clattice& l, int pos, int d, const context *c, bool unk, chunk& ch, int p, vt& as, vt& dpn, vt& an, vt& trm);
			void _mcls(int e, std::vector<int>& rc, vt& as, vt& dpn, vt& an, vt& trm, int p, double base);
			double _mtr(int p, std::vector<int>& rc, vt& trm);
			void _mtable(clattice& l, int pos, int d, int e, const context *c, bool unk, chunk& ch, vt& as, vt& trm, std::vector<int>& cl, std::vector<int>& cr, std::vector<double>& tbl, std::vector<std::vector<int> >& lpath, std::vector<std::vector<int> >& rpath);
			void _slice(clattice& l, nsentence *cur);
			void _resize();
			void _shrink();
		private:
	};
}

#endif
