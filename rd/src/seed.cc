#include"seed.h"

using namespace std;
using namespace npbnlp;

shared_ptr<seed> seed::_sd;
mutex seed::_mutex;
bool seed::_fixed = false;
unsigned int seed::_value = 0;

void seed::set(unsigned int value) {
	lock_guard<mutex> lock(_mutex);
	if (_sd != nullptr)
		throw "seed::set must precede random generator creation";
	_fixed = true;
	_value = value;
}
shared_ptr<seed> seed::create() {
	lock_guard<mutex> lock(_mutex);
	if (_sd == nullptr)
		_sd = shared_ptr<seed>(new seed);
	return _sd;
}

seed::seed():_pseudo(_value) {
}

seed::~seed() {
}

unsigned int seed::operator()() {
	return _fixed ? _pseudo() : _seed();
}
