#include"seed.h"

using namespace std;
using namespace npbnlp;

shared_ptr<seed> seed::_sd;
mutex seed::_mutex;
shared_ptr<seed> seed::create() {
	lock_guard<mutex> lock(_mutex);
	if (_sd == nullptr)
		_sd = shared_ptr<seed>(new seed);
	return _sd;
}

seed::seed() {
	_is_set = false;
}

seed::~seed() {
}

unsigned int seed::operator()() {
	if (_is_set)
		return _generator();
	return _seed();
}

void seed::set(unsigned int s) {
	_generator.seed(s);
	_is_set = true;
}

bool seed::is_set() const {
	return _is_set;
}
