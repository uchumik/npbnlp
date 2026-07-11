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
			virtual nsentence parse(nio& f, int i);
			virtual void add(nsentence& s);
			virtual void remove(nsentence& s);
			virtual void init(nsentence& s);
			virtual void set(int v, int k);
			virtual void set_k(int k);
			virtual int n();
			virtual int m();
			virtual int l();
			virtual void slice(double a, double b);
			virtual void set_original(bool f);
			// pos-pattern base measure (A-2): G0(c|z) = pos-seq HPYP x lexical fill-in
			virtual void set_posbase(bool f);
			virtual void set_lex(std::function<double(word&, int)> f, int posv = 0);
			// context-distribution factor (B-obs): per-class left/right distance
			// -indexed word distributions score each chunk by its surrounding words.
			// Vocabulary is acquired from data by the HPYPs (no external V needed).
			virtual void set_ctx(int j);
			// class-normalized (softmax) gate for the context factor: instead of
			// adding log(psi_z/psi_bg) per chunk (which inflates chunk-existence mass),
			// subtract logsumexp over surviving classes so the factor only redistributes
			// probability across classes and is invariant to the number of chunks.
			virtual void set_ctxgate(bool f);
			// per-word tokenizer-class channel: chunk latent class z emits each
			// word's tokenizer class (word.pos) via a multinomial theta_{z,p}, added
			// on top of (not replacing) the word LM. Per-word => invariant to chunk
			// count, so it sharpens class assignment without biasing merges.
			virtual void set_wclass(bool f);
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
			bool _original;
			std::shared_ptr<hpyp> _class;
			std::shared_ptr<std::vector<std::shared_ptr<hpyp> > > _chunk;
			std::shared_ptr<std::vector<std::shared_ptr<hpyp> > > _word;
			std::shared_ptr<std::vector<std::shared_ptr<vpyp> > > _letter;
			std::shared_ptr<std::vector<double> > _prior;   // aggregate boundary rate per chunk_type (for _clength)
			std::shared_ptr<std::vector<int> > _length;
			std::shared_ptr<std::vector<int> > _num;
			std::shared_ptr<std::vector<int> > _change;
			std::shared_ptr<std::vector<int> > _clength;
			std::shared_ptr<std::vector<double> > _cprior;
			// per-(chunk_type, word_type) Bernoulli boundary process (geometric-per-position
			// duration): _bp[t*WT+u] = P(boundary after a word of type u inside a chunk of type t).
			// index via _bpi(t,u); WT = chartype::n. Replaces the single-parameter geometric.
			std::shared_ptr<std::vector<double> > _bp;
			std::shared_ptr<std::vector<int> > _bcount;     // boundary events per (chunk_type, word_type)
			std::shared_ptr<std::vector<int> > _ccount;     // continue events per (chunk_type, word_type)
			// pos-pattern base measure (A-2): per-class HPYP over tokenizer pos ids
			bool _posbase;
			int _posv; // pos vocabulary (= tokenizer class count) for the pos-seq LM base
			std::shared_ptr<std::vector<std::shared_ptr<hpyp> > > _posseq;
			std::function<double(word&, int)> _lex;
			// context-distribution factor (B-obs): _ctxj = window radius (0 = off);
			// _lctx[k]/_rctx[k] are per-class HPYP(2) whose context symbol is the
			// distance e (1.._ctxj), giving distance->class->uniform backoff.
			int _ctxj;
			std::shared_ptr<std::vector<std::shared_ptr<hpyp> > > _lctx;
			std::shared_ptr<std::vector<std::shared_ptr<hpyp> > > _rctx;
			// class-independent background context LMs; the factor is a likelihood
			// ratio log(psi_z/psi_bg) so it is 0-centred and does not bias chunk count.
			std::shared_ptr<hpyp> _lbg;
			std::shared_ptr<hpyp> _rbg;
			// when true, the context factor is class-normalized (softmax gate) rather
			// than added raw; keeps bpdur boundary discipline while sharpening class.
			bool _ctxgate;
			// per-word tokenizer-class channel: _wclass on/off, _wc[z*(_posv+1)+p]
			// = count of tokenizer class p emitted by chunk latent class z. Sized
			// lazily to (_K+1)*(_posv+1) once _posv (= tokenizer class count) is known.
			bool _wclass;
			std::shared_ptr<std::vector<int> > _wc;
			std::mutex _mutex;
			// logsumexp of the raw context factor over surviving classes at segment
			// (t,j) starting at `start`; used as the normalizer for the softmax gate.
			double _ctx_norm(clattice2& l, int t, int j, int start);
			int _wci(int z, int p) const { return z*(_posv+1)+p; }
			void _wc_ensure();                         // lazy-size _wc from _posv
			double _wclass_lp(int z, chunk& ch);       // Sum_j log P(word_j.pos | z)
			void _wclass_count(chunk& ch, int d);      // add (d=+1) / remove (d=-1) theta counts
			void _forward(clattice2& l, int i, const context *c, const context *z, double& prior, chunk& ch, int k, chunk& prev, int q, vt& a, vt& b, int n, bool unk, bool not_exsit);
			void _backward(clattice2& l, int i, const context *c, const context *z, chunk& ch, int k, chunk& prev, int q, double& lpr, vt& b, int n, bool unk, bool not_exist);
			nsentence _minfer(nio& f, int i, bool best);
			void _mfill(clattice2& l, vt& dp, vt& am, vt& bos, vt& trm);
			void _mchain(clattice2& l, int pos, int d, const context *c, bool unk, chunk& ch, int p, double lnp, vt& as, vt& dpn, vt& an, vt& trm);
			void _mcls(int e, std::vector<int>& rc, vt& as, vt& dpn, vt& an, vt& trm, int p, double base);
			double _mtr(int p, std::vector<int>& rc, vt& trm);
			void _mtable(clattice2& l, int pos, int d, int e, const context *c, bool unk, chunk& ch, vt& as, vt& trm, std::vector<int>& cl, std::vector<int>& cr, std::vector<double>& tbl, std::vector<std::vector<int> >& lpath, std::vector<std::vector<int> >& rpath);
			void _slice(clattice2& l);
			void _length_prior(clattice2& l);
			void _resize();
			void _shrink();
			double _posseq_lp(int k, chunk& c);
			void _posseq_add(int k, chunk& c);
			void _posseq_remove(int k, chunk& c);
			void _install_cbase();
			void _context_factor(clattice2& l);
			void _ctx_seat(nsentence& s, bool add);
		private:
	};
}

#endif
