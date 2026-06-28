#include"usbd.h"
#include"usbd_lstm.h"
#include"rd.h"
#include<random>
#include<tuple>
#include<string>

#define BUFSIZE 256
#define SHIFT 3
#define BATCH 1
#define LAYERS 3
#define HIDDEN 1024
#define EMBSIZE 128
#define DIRECTIONS 2
#define OUTPUTS 1

using namespace std;
using namespace npbnlp;

/*
blstm::blstm(uint64_t layers, uint64_t hidden, uint64_t inputs, uint64_t embed_size, uint64_t output_size):lstm{nullptr}, rlstm{nullptr}, linear{nullptr}, embed{inputs, embed_size} {
	register_module("embed", embed);
	lstm = register_module("lstm",
			//torch::nn::LSTM(torch::nn::LSTMOptions(inputs, hidden).num_layers(layers))
			torch::nn::LSTM(torch::nn::LSTMOptions(embed_size, hidden).num_layers(layers))
			);
	rlstm = register_module("rlstm",
			//torch::nn::LSTM(torch::nn::LSTMOptions(inputs, hidden).num_layers(layers))
			torch::nn::LSTM(torch::nn::LSTMOptions(embed_size, hidden).num_layers(layers))
			);
	//linear = register_module("linear", torch::nn::Linear(hidden * DIRECTIONS, OUTPUTS));
	linear = register_module("linear", torch::nn::Linear(hidden * DIRECTIONS, output_size));
}

blstm::~blstm() {
}

torch::Tensor blstm::forward(torch::Tensor x) {
	auto e = embed->forward(x);
	auto lstm1 = lstm->forward(e);
	auto lstm2 = rlstm->forward(torch::flip(e, 0));
	auto cat = torch::empty({DIRECTIONS, BATCH, x.size(0), HIDDEN});
	cat[0] = std::get<0>(lstm1).view({BATCH, x.size(0), HIDDEN});
	cat[1] = torch::flip(std::get<0>(lstm2).view({BATCH, x.size(0), HIDDEN}), 1);
	//auto out = torch::sigmoid(linear->forward(cat.view({BATCH, x.size(0), HIDDEN * DIRECTIONS})));
	auto out = torch::nn::functional::log_softmax(linear->forward(cat.view({BATCH, x.size(0), HIDDEN * DIRECTIONS})), 1);
	return out;
}
*/
/*
torch::Tensor blstm::forward(torch::Tensor x) {
	auto lstm1 = lstm->forward(x.view({x.size(0), BATCH, -1}));
	auto lstm2 = rlstm->forward(torch::flip(x, 0).view({x.size(0), BATCH, -1}));
	auto cat = torch::empty({DIRECTIONS, BATCH, x.size(0), HIDDEN});
	cat[0] = std::get<0>(lstm1).view({BATCH, x.size(0), HIDDEN});
	cat[1] = torch::flip(std::get<0>(lstm2).view({BATCH, x.size(0), HIDDEN}), 1);
	auto out = torch::sigmoid(linear->forward(cat.view({BATCH, x.size(0), HIDDEN * DIRECTIONS})));
	return out;
}
*/

/*
void blstm::save(const char *model) {
	string prefix(model);
	string mlstm1(prefix);
	string mlstm2(prefix);
	string mlinear(prefix);
	string membed(prefix);
	mlstm1 += "lstm1";
	mlstm2 += "lstm2";
	mlinear += "linear";
	membed += "embed";
	torch::save(lstm, mlstm1.c_str());
	torch::save(rlstm, mlstm2.c_str());
	torch::save(linear, mlinear.c_str());
	torch::save(embed, membed.c_str());
}

void blstm::load(const char *model) {
	string prefix(model);
	string mlstm1(prefix);
	string mlstm2(prefix);
	string mlinear(prefix);
	string membed(prefix);
	mlstm1 += "lstm1";
	mlstm2 += "lstm2";
	mlinear += "linear";
	membed += "embed";
	torch::load(lstm, mlstm1.c_str());
	torch::load(rlstm, mlstm2.c_str());
	torch::load(linear, mlinear.c_str());
	torch::load(embed, membed.c_str());
}
*/

//blstm::blstm(uint64_t layers, uint64_t hidden_size, uint64_t vocab_size, uint64_t embed_size, uint64_t output_size): embed(vocab_size, embed_size), lstm(torch::nn::LSTMOptions(embed_size, hidden_size).num_layers(layers)), rlstm(torch::nn::LSTMOptions(embed_size, hidden_size).num_layers(layers)), linear(hidden_size*2, vocab_size), hidden_size(hidden_size), layers(layers)  {
blstm::blstm(uint64_t layers, uint64_t hidden_size, uint64_t vocab_size, uint64_t embed_size, uint64_t output_size): embed(vocab_size, embed_size), lstm(torch::nn::LSTMOptions(embed_size, hidden_size).num_layers(layers)), rlstm(torch::nn::LSTMOptions(embed_size, hidden_size).num_layers(layers)), linear(hidden_size, vocab_size), hidden_size(hidden_size), layers(layers)  {
	register_module("embed", embed);
	register_module("lstm", lstm);
	//register_module("rlstm", rlstm);
	register_module("linear", linear);
}

blstm::~blstm() {
}

torch::Tensor blstm::forward(torch::Tensor x) {
	tuple<torch::Tensor, torch::Tensor> state1;
	tuple<torch::Tensor, torch::Tensor> state2;
	torch::Tensor output1;
	torch::Tensor output2;
	torch::Tensor h = torch::zeros({layers, BATCH, hidden_size});
	torch::Tensor c = torch::zeros({layers, BATCH, hidden_size});
	//torch::Tensor rh = torch::zeros({layers, BATCH, hidden_size});
	//torch::Tensor rc = torch::zeros({layers, BATCH, hidden_size});

	tie(output1, state1) = lstm->forward(embed->forward(x), make_tuple(h, c));
	/*
	tie(output2, state2) = rlstm->forward(embed->forward(torch::flip(x,0)), make_tuple(rh, rc));
	auto cat = torch::empty({2, BATCH, x.size(0), hidden_size});
	cat[0] = output1.view({BATCH, x.size(0), hidden_size});
	cat[1] = torch::flip(output2.view({BATCH, x.size(0), hidden_size}), 1);
	auto out = linear->forward(cat.view({BATCH, x.size(0), hidden_size*2}));

	*/
	auto out = linear->forward(output1.view({BATCH, x.size(0), hidden_size}));
	out = torch::nn::functional::log_softmax(out, 1);
	//cout << out << endl;
	return out;
}

void blstm::save(const char *model) {
	string prefix(model);
	string mparam(prefix);
	string mlstm1(prefix);
	string mlstm2(prefix);
	string mlinear(prefix);
	string membed(prefix);
	mlstm1 += "lstm1";
	mlstm2 += "lstm2";
	mlinear += "linear";
	membed += "embed";
	torch::save(lstm, mlstm1.c_str());
	torch::save(rlstm, mlstm2.c_str());
	torch::save(linear, mlinear.c_str());
	torch::save(embed, membed.c_str());
	FILE *fp = NULL;
	if ((fp = fopen(mparam.c_str(), "wb")) == NULL)
		throw "failed to open mparam in blstm::save";
	if (fwrite(&hidden_size, sizeof(uint64_t), 1, fp) != 1)
		throw "failed to write hidden_size in blstm::save";
	fclose(fp);
}

void blstm::load(const char *model) {
	string prefix(model);
	string mparam(prefix);
	string mlstm1(prefix);
	string mlstm2(prefix);
	string mlinear(prefix);
	string membed(prefix);
	mlstm1 += "lstm1";
	mlstm2 += "lstm2";
	mlinear += "linear";
	membed += "embed";
	torch::load(lstm, mlstm1.c_str());
	torch::load(rlstm, mlstm2.c_str());
	torch::load(linear, mlinear.c_str());
	torch::load(embed, membed.c_str());
	FILE *fp = NULL;
	if ((fp = fopen(mparam.c_str(), "rb")) == NULL)
		throw "failed to open mparam in blstm::load";
	if (fread(&hidden_size, sizeof(uint64_t), 1, fp) != 1)
		throw "failed to write hidden_size in blstm::load";
	fclose(fp);
}

usbd_lstm::usbd_lstm():_lm(nullptr), _char(0) {
}

usbd_lstm::~usbd_lstm() {
}

usbd_lstm::usbd_lstm(const usbd_lstm& d) {
	_lm = d._lm;
	_char = d._char;
}

usbd_lstm& usbd_lstm::operator=(const usbd_lstm& d) {
	_lm = d._lm;
	_char = d._char;
	return *this;
}

void usbd_lstm::init(cio& c, int n) {
	int id = 0;
	_index_to_char.resize(id+1);
	_char_to_index[0] = id;
	_index_to_char[id++] = 0;
	for (auto k = 0; k < (int)c.chunk->size(); ++k) {
		io& f = (*c.chunk)[k];
		int size = f.head.size() - 1;
		for (auto j = 0; j < size; ++j) {
			int head = f.head[j];
			int tail = f.head[j+1];
			for (auto i = head; i < tail; ++i) {
				if (_char_to_index.find((*f.raw)[i]) == _char_to_index.end()) {
					_index_to_char.resize(id+1, 0);
					int c = (*f.raw)[i];
					_char_to_index[c] = id;
					_index_to_char[id++] = c;
				}
				/*
				if (_char < 1+(*f.raw)[i]) {
					_char = 1+(*f.raw)[i];
				}
				*/
			}
		}
	}
	_char = id;
	//_lm = shared_ptr<blstm>(new blstm(LAYERS, HIDDEN, _char));
	_lm = shared_ptr<blstm>(new blstm(LAYERS, HIDDEN, _char, EMBSIZE, _char));
	for (auto k = 0; k < (int)c.chunk->size(); ++k) {
		io& f = (*c.chunk)[k];
		pretrain(f, 1);
	}
}

void usbd_lstm::pretrain(io& f, int iter) {
	int size = f.head.size() - 1;
	torch::optim::Adam optimizer(_lm->parameters(), torch::optim::AdamOptions(0.002));
	for (auto i = 0; i < iter; ++i) {
		int rd[size] = {0};
		rd::shuffle(rd, size);
		for (auto j = 0; j < size; ++j) {
			int head = f.head[rd[j]];
			int tail = f.head[rd[j]+1];
			word w(*f.raw, head, tail-head);
			torch::Tensor input = _input(w);
			torch::Tensor target = _target(w);
			torch::Tensor output = _lm->forward(input);

			//auto loss = torch::mse_loss(output.view({w.len, OUTPUTS}), target);
			//auto loss = torch::nn::functional::nll_loss(output.view({w.len, OUTPUTS}), target);
			//auto loss = torch::nll_loss(output.view({w.len, OUTPUTS}), target);
			auto loss = torch::nn::functional::nll_loss(output.reshape({-1, output.size(2)}), target.reshape(-1));
			optimizer.zero_grad();
			loss.backward();
			torch::nn::utils::clip_grad_norm_(_lm->parameters(), 0.5);
			optimizer.step();
		}
	}
}

void usbd_lstm::add(io& d, vector<int>& head) {
	int size = head.size() - 1;
	torch::optim::Adam optimizer(_lm->parameters(), torch::optim::AdamOptions(0.002));
	int rd[size] = {0};
	rd::shuffle(rd, size);
	for (auto i = 0; i < size; ++i) {
		int h = head[rd[i]];
		int t = head[rd[i]+1];
		word w(*d.raw, h, t-h);
		torch::Tensor input = _input(w);
		torch::Tensor target = _target(w);
		torch::Tensor output = _lm->forward(input);
		//cerr << "output:" << output.view({w.len, OUTPUTS}) << endl;
		//cerr << "target:" << target << endl;
		//auto loss = torch::mse_loss(output.view({w.len, OUTPUTS}), target);
		//auto loss = torch::nn::functional::nll_loss(output.view({w.len, OUTPUTS}), target);
		//auto loss = torch::nll_loss(output.view({w.len, OUTPUTS}), target);
		auto loss = torch::nn::functional::nll_loss(output.reshape({-1, output.size(2)}), target.reshape(-1));
		optimizer.zero_grad();
		loss.backward();
		torch::nn::utils::clip_grad_norm_(_lm->parameters(), 0.5);
		optimizer.step();
	}
}

void usbd_lstm::sample(io& d, vector<int>& b) {
	if (d.raw->size() > BUFSIZE) {
		auto i = d.head[0];
		while (i < (int)d.raw->size()) {
			io g;
			g.head.clear();
			g.raw = d.raw;
			g.head.emplace_back(i);
			for (auto j = 0; j < (int)d.head.size() && d.head[j] < i+BUFSIZE; ++j) {
				if (d.head[j] > i)
					g.head.emplace_back(d.head[j]);
			}
			if (g.head[g.head.size()-1] != d.head[d.head.size()-1]) {
				g.head.emplace_back(min((int)d.raw->size(), i+BUFSIZE));
			}
			vector<int> c;
			_sample(g, c);
			if (i == 0) {
				for (auto& j : c) {
					b.emplace_back(j);
				}
			} else {
				if (!b.empty())
					b.pop_back();
				for (auto k = 1; k < (int)c.size(); ++k) {
					b.emplace_back(c[k]);
				}
			}
			if (c.size() > 2) {
				i = c[c.size()-2];
			} else {
				i = min((int)d.raw->size(), i+BUFSIZE-SHIFT);
			}
		}
	} else {
		_sample(d, b);
	}
}

void usbd_lstm::parse(io& d, vector<int>& b) {
	if (d.raw->size() > BUFSIZE) {
		auto i = d.head[0];
		while (i < (int)d.raw->size()) {
			io g;
			g.head.clear();
			g.raw = d.raw;
			g.head.emplace_back(i);
			for (auto j = 0; j < (int)d.head.size() && d.head[j] < i+BUFSIZE; ++j) {
				if (d.head[j] > i)
					g.head.emplace_back(d.head[j]);
			}
			if (g.head[g.head.size()-1] != d.head[d.head.size()-1]) {
				g.head.emplace_back(min((int)d.raw->size(), i+BUFSIZE));
			}
			vector<int> c;
			_parse(g, c);
			if (i == 0) {
				for (auto& j : c) {
					b.emplace_back(j);
				}
			} else {
				if (!b.empty())
					b.pop_back();
				for (auto k = 1; k < (int)c.size(); ++k) {
					b.emplace_back(c[k]);
				}
			}
			if (c.size() > 2) {
				i = c[c.size()-2];
			} else {
				i = min((int)d.raw->size(), i+BUFSIZE/2);
			}
		}
	} else {
		_parse(d, b);
	}
}

void usbd_lstm::_sample(io& d, vector<int>& b) {
	vt dp;
	int size = d.head.size()-1;
	int head = d.head[0];
	int tail = d.head[size];
	int len = tail-head;
	if (len <= 1) {
		b.emplace_back(head);
		b.emplace_back(tail);
		return;
	}

	unordered_map<int, double> alpha;
	for (int t = 0; t < len; ++t) {
		double a = 0;
		for (int l = 0; l < t; ++l) {
			a = math::lse(a, dp[t-1][l].v, (l==0));
		}
		alpha[t-1] = a;
		for (int j = 0; j <= t; ++j) {
			word w(*d.raw, head+t-j, j+1);
			torch::Tensor input = _input(w);
			torch::Tensor output = _lm->forward(input);
			double lp = 0;
			for (auto i = 0; i < w.len; ++i) 
				lp += output[0][i][_char_to_index[w[i]]].item<double>();
			//double lp = output[0][j+1][0].item<double>();
			//cout << output << endl;
			//cout << "eos:" << lp << endl;
			dp[t][j].v = lp+alpha[t-j-1];
		}
	}
	b.emplace_back(tail);
	int t = len-1;
	while (t >= 0) {
		vector<double> table;
		for (int k = 0; k <= t; ++k) {
			table.emplace_back(dp[t][k].v);
		}
		int l = 1+rd::ln_draw(table);
		t -= l;
		if (t > 0)
			b.emplace_back(head+t+1);
	}
	b.emplace_back(head);
	reverse(b.begin(), b.end());
}

void usbd_lstm::_parse(io& d, vector<int>& b) {
	vt dp;
	int size = d.head.size()-1;
	int head = d.head[0];
	int tail = d.head[size];
	int len = tail-head;
	if (len <= 1) {
		b.emplace_back(head);
		b.emplace_back(tail);
		return;
	}

	unordered_map<int, double> alpha;
	for (int t = 0; t < len; ++t) {
		double a = 0;
		for (int l = 0; l < t; ++l) {
			a = math::lse(a, dp[t-1][l].v, (l==0));
		}
		alpha[t-1] = a;
		for (int j = 0; j <= t; ++j) {
			word w(*d.raw, head+t-j, j+1);
			torch::Tensor input = _input(w);
			torch::Tensor output = _lm->forward(input);
			double lp = output[0][j+1][0].item<double>();
			dp[t][j].v = lp+alpha[t-j-1];
		}
	}
	b.emplace_back(tail);
	int t = len-1;
	while (t >= 0) {
		vector<double> table;
		for (int k = 0; k <= t; ++k) {
			table.emplace_back(dp[t][k].v);
		}
		int l = 1+rd::best(table);
		t -= l;
		if (t > 0)
			b.emplace_back(head+t+1);
	}
	b.emplace_back(head);
	reverse(b.begin(), b.end());
}

/*
   void usbd_lstm::_sample(io& d, vector<int>& b) {
   vt dp;
   int size = d.head.size() - 1;
   int head = d.head[0];
   int tail = d.head[size];
   int len = tail-head;
   if (len <= 1) {
   b.emplace_back(head);
   b.emplace_back(tail);
   return;
   }
   word w(*d.raw, head, len);
   torch::Tensor input = _input(w);
   torch::Tensor output = _lm->forward(input);
   for (int t = 0; t < len; ++t) {
   double p = output[0][t][0].item<double>();
   double lp = log(max(p, 0.0001));
   dp[t][0].v = lp;
   for (int l = t-1; l >= 0; --l) {
   double q = output[0][l][0].item<double>();
   lp += log(max(1.-q, 0.0001));
   dp[t][t-l].v = lp;
   }
   }
   b.emplace_back(tail);
   int t = len - 1;
   while (t >= 0) {
   vector<double> table;
   for (int k = 0; k <= t; ++k) {
   table.emplace_back(dp[t][k].v);
   }
   int l = 1+rd::ln_draw(table);
   t -= l;
   if (t > 0)
   b.emplace_back(head+t+1);
   }
   b.emplace_back(head);
   reverse(b.begin(), b.end());
   }
   */

/*
   void usbd_lstm::sample(io& d, vector<int>& b) {
   int size = d.head.size() - 1;
   int head = d.head[0];
   int tail = d.head[size];
   int len = tail-head;
   word w(*d.raw, head, len);
//cout << w << " length:" << len << endl;
torch::Tensor input = _input(w);
torch::Tensor output = _lm->forward(input);
//cout << output.view({w.len, OUTPUTS}) << endl;
//cout << output.size(0) << endl;
//auto output_acc = output.accessor<float, 2>();
b.emplace_back(head);
bernoulli_distribution ber;
shared_ptr<generator> g = generator::create();
for (auto i = 0; i < len-1; ++i) {
//cout << output[0][i][0].item().toDouble() << endl;
bernoulli_distribution::param_type mu(output[0][i][0].item<double>());
if (ber((*g)(), mu) > 0) {
b.emplace_back(head+i+1);
}
}
b.emplace_back(tail);
//for (auto s : b) {
//	cout << s << endl;
//}
}
*/

/*
   void usbd_lstm::parse(io& d, vector<int>& b) {
   int size = d.head.size() - 1;
   int head = d.head[0];
   int tail = d.head[size];
   int len = tail-head;
   word w(*d.raw, head, len);
   torch::Tensor input = _input(w);
   torch::Tensor output = _lm->forward(input);
//auto output_acc = output.accessor<float, 2>();
b.emplace_back(head);
for (auto i = 0; i < len-1; ++i) {
if (output[0][i][0].item<double>() >= 0.5) {
b.emplace_back(head+i+1);
}
}
b.emplace_back(tail);
}
*/

void usbd_lstm::eval(cio& target, cio& correct, vector<double>& c) {
	if (target.chunk->size() != correct.chunk->size())
		return;
	if (!c.empty())
		c.clear();
	int letters = 0;
	int correct_seg = 0;
	int target_seg = 0;
	int tp = 0;
	int fp = 0;
	int fn = 0;
	for (auto i = 0; i < (int)target.chunk->size(); ++i) {
		correct_seg += (*correct.chunk)[i].head.size();
		target_seg += (*target.chunk)[i].head.size();
		letters += (*target.chunk)[i].raw->size();
		int j = 0;
		int k = 0;
		while (1) {
			if ((*correct.chunk)[i].head[j] == (*target.chunk)[i].head[k]) {
				++tp, ++j, ++k;
			} else if ((*correct.chunk)[i].head[j] < (*target.chunk)[i].head[k]) {
				++j; // correct boundary missing in target -> false negative
				++fn;
			} else {
				++k; // spurious boundary in target -> false positive
				++fp;
			}
			if (j >= (int)(*correct.chunk)[i].head.size() && k < (int)(*target.chunk)[i].head.size()) {
				int last = (*correct.chunk)[i].head[(*correct.chunk)[i].head.size()-1];
				for (; k < (int)(*target.chunk)[i].head.size(); ++k) {
					if ((*target.chunk)[i].head[k] == last)
						++tp;
					else
						++fp;
				}
			} else if (j < (int)(*correct.chunk)[i].head.size() && k >= (int)(*target.chunk)[i].head.size()) {
				int last = (*target.chunk)[i].head[(*target.chunk)[i].head.size()-1];
				for (; j < (int)(*correct.chunk)[i].head.size(); ++j) {
					if ((*correct.chunk)[i].head[j] == last)
						++tp;
					else
						++fn;
				}
			}
			if (j >= (int)(*correct.chunk)[i].head.size() && k >= (int)(*target.chunk)[i].head.size())
				break;
		}
	}
	int tn = letters-correct_seg;
	double prec = (double)tp/((double)tp+fp);
	double rec = (double)tp/((double)tp+fn);
	double f1 = prec*rec*2./(prec+rec);
	double acc = (double)(tp+tn)/(double)(tp+fp+tn+fn);
	c.emplace_back(prec);
	c.emplace_back(rec);
	c.emplace_back(f1);
	c.emplace_back(acc);
}

void usbd_lstm::save(const char *file) {
	return;
}

void usbd_lstm::load(const char *file) {
	return;
}

void usbd_lstm::remove(io& d, vector<int>& head) {
	return;
}

void usbd_lstm::set_punc(unsigned int c) {
	return;
}

void usbd_lstm::set_prior(double g, double c, double p) {
	return;
}

void usbd_lstm::set_general_prior(double p) {
	return;
}

void usbd_lstm::set_cr_prior(double p) {
	return;
}

void usbd_lstm::set_punc_prior(double p) {
	return;
}

void usbd_lstm::set_hyper_general(double a, double b) {
	return;
}

void usbd_lstm::set_hyper_cr(double a, double b) {
	return;
}

void usbd_lstm::set_hyper_punc(double a, double b) {
	return;
}

void usbd_lstm::estimate_lm_hyper(int iter) {
	return;
}

void usbd_lstm::estimate_prior(cio& corpus, vector<vector<int> >& boundaries) {
	return;
}

torch::Tensor usbd_lstm::_input(word& w) {
	vector<int64_t> v;
	//for (auto i = 0; i < w.len; ++i) {
	for (auto i = -1; i < w.len; ++i) {
		//v.emplace_back(w[i]);
		v.emplace_back(_char_to_index[w[i]]);
	}
	return torch::from_blob(v.data(), {v.size(), 1},
			torch::TensorOptions().dtype(torch::kInt64)).clone();
	/*
	   torch::Tensor input = torch::empty({w.len, _char});
	//input.to_sparse();
	auto input_acc = input.accessor<float, 2>();
	for (auto i = 0; i < w.len; ++i) {
	input_acc[i][w[i]] = 1.;
	}
	return input;
	*/
	//return input.to_sparse();
}

torch::Tensor usbd_lstm::_target(word& w) {
	vector<int64_t> v;
	for (auto i = 0; i < w.len+1; ++i) {
		//v.emplace_back(w[i]);
		v.emplace_back(_char_to_index[w[i]]);
	}
	return torch::from_blob(v.data(), {v.size(), 1}, torch::TensorOptions().dtype(torch::kInt64)).clone();
}
/*
   torch::Tensor usbd_lstm::_target(word& w) {
   torch::Tensor target = torch::empty({w.len, OUTPUTS});
   auto target_acc = target.accessor<float, 2>();
   for (auto i = 0; i < w.len; ++i) {
   if (i < w.len-1)
   target_acc[i][0] = 0.;
   else
   target_acc[i][0] = 1.;
   }
   return target;
   }
   */
