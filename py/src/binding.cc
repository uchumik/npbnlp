// Phase E: inference-only pybind11 bindings for npbnlp.
//
// Each class mirrors the --parse code path of the corresponding CLI binary
// as closely as possible (constructors load models eagerly; parse() copies
// the input string in, computes into plain C++ structures, and converts to
// python objects only after the GIL is reacquired).
#include<pybind11/pybind11.h>
#include<pybind11/stl.h>

#include<memory>
#include<sstream>
#include<string>
#include<utility>
#include<vector>

#include"io.h"
#include"cio.h"
#include"nio.h"
#include"word.h"
#include"sentence.h"
#include"nsentence.h"
#include"chunk.h"
#include"tree.h"

#include"npylm.h"
#include"phsmm.h"
#include"nphsmm.h"
#include"ipcfg.h"
#include"usbd.h"

namespace py = pybind11;
using namespace npbnlp;

// ---------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------

// surface string of a word, same character loop as word::operator<< (io/include/word.h)
static std::string word_surface(word& w) {
	std::string s;
	for (auto i = 0; i < w.len; ++i) {
		char buf[5] = {0};
		io::i2c(w[i], buf);
		s += buf;
	}
	return s;
}

// surface string of a chunk, same character loop as chunk::operator<< (io/include/chunk.h)
static std::string chunk_surface(chunk& c) {
	std::string s;
	for (auto i = 0; i < c.len; ++i) {
		word& w = c.wd(i);
		s += word_surface(w);
	}
	return s;
}

// s-expression string for a parse tree, mirrors dump_node() in pa/src/pa.cc
static void tree_node_str(tree& t, int i, std::string& out) {
	node& c = t[i];
	if (c.i != c.j) {
		int left = t.s.size()*c.i+c.b-c.i*(1.+c.i)/2;
		int right = t.s.size()*(c.b+1)+c.j-(1.+c.b)*(2+c.b)/2;
		out += "(";
		out += std::to_string(c.k);
		out += " ";
		tree_node_str(t, left, out);
		tree_node_str(t, right, out);
		out += ")";
	} else if (c.k > 0 && c.i == c.j) {
		out += "(";
		out += std::to_string(c.k);
		out += " ";
		out += word_surface(t.wd(c.i));
		out += ")";
	}
}

// ---------------------------------------------------------------------
// Ws - word segmentation (NPYLM), mirrors ma/src/tokenize.cc parse()
// ---------------------------------------------------------------------
class Ws {
	public:
		Ws(const std::string& model, const std::string& dic) : _lm(2, 20) {
			std::shared_ptr<wid> d = wid::create();
			if (!d->load(dic.c_str()))
				throw "Ws: failed to load dic";
			_lm.load(model.c_str());
		}
		std::vector<std::vector<std::string> > parse(const std::string& text) {
			std::istringstream iss(text);
			io f(iss);
			std::vector<std::vector<std::string> > out;
			out.reserve(f.head.size() > 0 ? f.head.size()-1 : 0);
			for (auto i = 0; i < (int)f.head.size()-1; ++i) {
				sentence s = _lm.parse(f, i);
				std::vector<std::string> words;
				words.reserve(s.size());
				for (auto j = 0; j < s.size(); ++j)
					words.emplace_back(word_surface(s.wd(j)));
				out.emplace_back(std::move(words));
			}
			return out;
		}
	private:
		npylm _lm;
};

// ---------------------------------------------------------------------
// Ma - morphological analysis (PHSMM), mirrors ma/src/ma.cc parse()
// ---------------------------------------------------------------------
class Ma {
	public:
		Ma(const std::string& model, const std::string& dic) : _lm(1, 20, 2, 10) {
			std::shared_ptr<wid> d = wid::create();
			if (!d->load(dic.c_str()))
				throw "Ma: failed to load dic";
			_lm.load(model.c_str());
		}
		std::vector<std::vector<std::pair<std::string, int> > > parse(const std::string& text) {
			std::istringstream iss(text);
			io f(iss);
			std::vector<std::vector<std::pair<std::string, int> > > out;
			out.reserve(f.head.size() > 0 ? f.head.size()-1 : 0);
			for (auto i = 0; i < (int)f.head.size()-1; ++i) {
				sentence s = _lm.parse(f, i);
				std::vector<std::pair<std::string, int> > words;
				words.reserve(s.size());
				for (auto j = 0; j < s.size(); ++j) {
					word& w = s.wd(j);
					words.emplace_back(word_surface(w), w.pos);
				}
				out.emplace_back(std::move(words));
			}
			return out;
		}
	private:
		phsmm _lm;
};

// ---------------------------------------------------------------------
// Ne - NER (NPHSMM), mirrors tg/src/ne.cc tokenize()+parse() sequence
// (see tg/src/ne.cc: tokenize(), parse(nio&), main()'s --parse branch)
// ---------------------------------------------------------------------
class Ne {
	public:
		Ne(const std::string& model, const std::string& cdic,
		   const std::string& tokenizer, const std::string& wdic)
			: _cdic(cdic), _wdic(wdic) {
			_toklm = std::make_shared<phsmm>();
			_toklm->load(tokenizer.c_str());
			std::shared_ptr<cid> cd = cid::create();
			if (!cd->load(_cdic.c_str()))
				throw "Ne: failed to load cdic";
			_lm.set_lex([this](word& w, int p) { return _toklm->lexlp(w, p); }, _toklm->k());
			_lm.load(model.c_str());
			_lm.slice(1, 5); // matches ne.cc statics a=1, b=5
		}
		std::vector<std::vector<std::pair<std::string, int> > > parse(const std::string& text) {
			// mirrors tg/src/ne.cc:tokenize(io&, vector<sentence>&), minus the
			// wdic->save() at the end (inference must not mutate the dic file)
			std::shared_ptr<wid> d = wid::create();
			if (!d->load(_wdic.c_str()))
				throw "Ne: failed to load wdic";
			std::shared_ptr<cid> cd = cid::create();
			if (!cd->load(_cdic.c_str()))
				throw "Ne: failed to load cdic";
			std::istringstream iss(text);
			io g(iss);
			std::vector<sentence> ws(g.head.size() > 0 ? g.head.size()-1 : 0);
			for (auto i = 0; i < (int)g.head.size()-1; ++i)
				ws[i] = _toklm->parse(g, i);
			for (auto& s : ws) {
				for (auto j = 0; j < s.size(); ++j) {
					word& w = s.wd(j);
					if (w.id == 1)
						w.id = d->index(w);
				}
			}
			nio f(ws);
			std::vector<std::vector<std::pair<std::string, int> > > out;
			out.reserve(f.head.size() > 0 ? f.head.size()-1 : 0);
			for (auto i = 0; i < (int)f.head.size()-1; ++i) {
				nsentence s = _lm.parse(f, i);
				std::vector<std::pair<std::string, int> > chunks;
				chunks.reserve(s.size());
				for (auto j = 0; j < s.size(); ++j) {
					chunk& c = s.ch(j);
					chunks.emplace_back(chunk_surface(c), c.k);
				}
				out.emplace_back(std::move(chunks));
			}
			return out;
		}
	private:
		std::string _cdic;
		std::string _wdic;
		std::shared_ptr<phsmm> _toklm;
		nphsmm _lm;
};

// ---------------------------------------------------------------------
// Usbd - sentence boundary detection, mirrors sb/src/usbd_main.cc parse()
// ---------------------------------------------------------------------
class Usbd {
	public:
		Usbd(const std::string& model, const std::string& dic, const std::string& mode = "letter")
			: _dic(dic) {
			if (mode == "word")
				_type = sequence_type::word;
			else if (mode == "letter")
				_type = sequence_type::letter;
			else
				throw "Usbd: unsupported mode (expected 'letter' or 'word')";
			if (!_bd.create(5, _type))
				throw "Usbd: failed to create detector";
			_bd.load(model.c_str());
			if (_type == sequence_type::word) {
				std::shared_ptr<wid> d = wid::create();
				if (!d->load(_dic.c_str()))
					throw "Usbd: failed to load dic";
			}
		}
		std::vector<std::string> parse(const std::string& text) {
			std::istringstream iss(text);
			cio f(iss);
			std::vector<std::string> out;
			for (auto i = 0; i < (int)f.chunk->size(); ++i) {
				io& doc = (*f.chunk)[i];
				std::vector<int> c;
				_bd.parse(doc, c);
				if (_type == sequence_type::word)
					_dump_word(doc, c, out);
				else
					_dump_letter(doc, c, out);
			}
			return out;
		}
	private:
		// mirrors dump_letter() in sb/src/usbd_main.cc
		static void _dump_letter(io& f, std::vector<int>& head, std::vector<std::string>& out) {
			for (auto i = 0; i < (int)head.size()-1; ++i) {
				int h = head[i];
				int t = head[i+1];
				std::string s;
				for (auto j = h; j < t; ++j) {
					char buf[5] = {0};
					io::i2c((*f.raw)[j], buf);
					s += buf;
				}
				out.emplace_back(std::move(s));
			}
		}
		// mirrors dump_word() in sb/src/usbd_main.cc
		static void _dump_word(io& f, std::vector<int>& head, std::vector<std::string>& out) {
			sentence s;
			for (auto i = 0; i < (int)f.head.size()-1; ++i) {
				int h = f.head[i];
				int t = f.head[i+1];
				sentence ss;
				ss.init_without_indexing(*f.raw, h, t);
				s.cat(ss);
			}
			int id = 0;
			for (auto i = 0; i < (int)head.size()-1; ++i) {
				int t = head[i+1];
				std::string line;
				for (; id < s.size() && s.wd(id).head < t; ++id) {
					word& w = s.wd(id);
					line += word_surface(w);
					if (id < s.size()-1 && s.wd(id+1).head < t)
						line += " ";
				}
				out.emplace_back(std::move(line));
			}
		}
		std::string _dic;
		sequence_type _type;
		bd_wrap _bd;
};

// ---------------------------------------------------------------------
// Pa - grammar induction (IPCFG), mirrors pa/src/pa.cc parse()
// ---------------------------------------------------------------------
class Pa {
	public:
		Pa(const std::string& model, const std::string& dic) : _g(20) {
			std::shared_ptr<wid> d = wid::create();
			if (!d->load(dic.c_str()))
				throw "Pa: failed to load dic";
			_g.load(model.c_str());
			_g.set(50000, 100); // matches pa.cc statics vocab=50000, K=100
		}
		std::vector<std::string> parse(const std::string& text) {
			std::istringstream iss(text);
			io f(iss);
			std::vector<std::string> out;
			out.reserve(f.head.size() > 0 ? f.head.size()-1 : 0);
			for (auto i = 0; i < (int)f.head.size()-1; ++i) {
				tree t = _g.parse(f, i);
				std::string s;
				tree_node_str(t, t.s.size()-1, s);
				out.emplace_back(std::move(s));
			}
			return out;
		}
	private:
		ipcfg _g;
};

PYBIND11_MODULE(npbnlp, m) {
	m.doc() = "npbnlp inference-only python bindings (Phase E)";

	py::register_exception_translator([](std::exception_ptr p) {
		try {
			if (p) std::rethrow_exception(p);
		} catch (const char *ex) {
			PyErr_SetString(PyExc_RuntimeError, ex);
		}
	});

	py::class_<Ws>(m, "Ws")
		.def(py::init<const std::string&, const std::string&>(),
			py::arg("model"), py::arg("dic"))
		.def("parse", &Ws::parse, py::arg("text"),
			py::call_guard<py::gil_scoped_release>());

	py::class_<Ma>(m, "Ma")
		.def(py::init<const std::string&, const std::string&>(),
			py::arg("model"), py::arg("dic"))
		.def("parse", &Ma::parse, py::arg("text"),
			py::call_guard<py::gil_scoped_release>());

	py::class_<Ne>(m, "Ne")
		.def(py::init<const std::string&, const std::string&, const std::string&, const std::string&>(),
			py::arg("model"), py::arg("cdic"), py::arg("tokenizer"), py::arg("wdic"))
		.def("parse", &Ne::parse, py::arg("text"),
			py::call_guard<py::gil_scoped_release>());

	py::class_<Usbd>(m, "Usbd")
		.def(py::init<const std::string&, const std::string&, const std::string&>(),
			py::arg("model"), py::arg("dic"), py::arg("mode") = "letter")
		.def("parse", &Usbd::parse, py::arg("text"),
			py::call_guard<py::gil_scoped_release>());

	py::class_<Pa>(m, "Pa")
		.def(py::init<const std::string&, const std::string&>(),
			py::arg("model"), py::arg("dic"))
		.def("parse", &Pa::parse, py::arg("text"),
			py::call_guard<py::gil_scoped_release>());
}
