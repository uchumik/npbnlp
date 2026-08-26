#include "gseq.h"
#include <cstdio>
#include <fstream>
#include <sstream>

namespace npbnlp {
	fvector::fvector():pos(0),n(1) {}
	fvector::fvector(const char *str):pos(0),n(1) { _parse(str); }
	fvector::fvector(const fvector& x):pos(x.pos),n(x.n),v(x.v) {}
	fvector& fvector::operator=(const fvector& x) {
		if (this != &x) { pos = x.pos; n = x.n; v = x.v; }
		return *this;
	}
	fvector::~fvector() {}
	double& fvector::operator[](int i) {
		if (i < 0) throw "negative fvector index";
		if (i >= (int)v.size()) v.resize(i + 1, 0.0);
		return v[i];
	}
	double fvector::operator[](int i) const { return v[i]; }
	int fvector::size() const { return (int)v.size(); }
	void fvector::push_back(double x) { v.push_back(x); }
	void fvector::_parse(const char *str) {
		std::istringstream in(str);
		std::string first;
		if (!(in >> first)) return;
		size_t colon = first.find(':');
		if (colon != std::string::npos) {
			pos = std::stoi(first.substr(0, colon));
			if (colon + 1 < first.size()) v.push_back(std::stod(first.substr(colon + 1)));
		} else v.push_back(std::stod(first));
		double x;
		while (in >> x) v.push_back(x);
	}

	gsentence::gsentence() {}
	gsentence::~gsentence() {}
	fvector& gsentence::vec(int i) {
		if (i < 0) return _bos;
		if (i >= (int)v.size()) return _eos;
		return v[i];
	}
	int gsentence::size() const { return (int)v.size(); }
	void gsentence::add(const fvector& x) {
		if (v.empty()) {
			_bos.v.assign(x.size(), 0.0);
			_eos.v.assign(x.size(), 0.0);
		}
		v.push_back(x);
	}

	gio::gio():_dim(0) {}
	gio::gio(const char *file):_dim(0) { load(file); }
	gio::~gio() {}
	void gio::load(const char *file) {
		std::ifstream in(file);
		if (!in) throw "failed to open gio file";
		seq.clear();
		_dim = 0;
		gsentence current;
		std::string line;
		while (std::getline(in, line)) {
			std::istringstream trim(line);
			std::string first;
			if (!(trim >> first)) {
				if (current.size() > 0) { seq.push_back(current); current = gsentence(); }
				continue;
			}
			if (first[0] == '#') continue;
			fvector x(line.c_str());
			if (_dim == 0) _dim = x.size();
			if (x.size() != _dim) throw "dimension mismatch in gio::load";
			current.add(x);
		}
		if (current.size() > 0) seq.push_back(current);
		for (auto& s : seq) {
			s._bos.pos = 0; s._bos.n = 1; s._bos.v.assign(_dim, 0.0);
			s._eos.pos = 0; s._eos.n = 1; s._eos.v.assign(_dim, 0.0);
		}
	}
	int gio::size() const { return (int)seq.size(); }
	int gio::dim() const { return _dim; }
	gsentence& gio::operator[](int i) { return seq[i]; }
}
