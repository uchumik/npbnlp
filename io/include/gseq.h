#ifndef NPBNLP_GSEQ_H
#define NPBNLP_GSEQ_H

#include <string>
#include <vector>

namespace npbnlp {
	class fvector {
	public:
		fvector();
		fvector(const char *str);
		fvector(const fvector& v);
		fvector& operator=(const fvector& v);
		virtual ~fvector();
		double& operator[](int i);
		double operator[](int i) const;
		int size() const;
		void push_back(double x);
		int pos;
		int n;
		std::vector<double> v;
	private:
		void _parse(const char *str);
	};

	class gsentence {
	public:
		gsentence();
		virtual ~gsentence();
		fvector& vec(int i);
		int size() const;
		void add(const fvector& x);
		std::vector<fvector> v;
	private:
		fvector _bos;
		fvector _eos;
		friend class gio;
	};

	class gio {
	public:
		gio();
		gio(const char *file);
		virtual ~gio();
		void load(const char *file);
		int size() const;
		int dim() const;
		gsentence& operator[](int i);
		std::vector<gsentence> seq;
	private:
		int _dim;
	};
}

#endif
