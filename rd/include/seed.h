#ifndef NPBNLP_SEED_H
#define NPBNLP_SEED_H

#include<random>
#include<memory>
#include<mutex>
namespace npbnlp {
	class seed {
		public:
			static std::shared_ptr<seed> create();
			// Must be called before any random distribution is created.  This is
			// primarily for reproducible sampler diagnostics.
			static void set(unsigned int value);
			seed(seed&&) = delete;
			seed(const seed&) = delete;
			seed& operator=(const seed&) = delete;
			seed& operator=(seed&&) = delete;
			unsigned int operator()();
			virtual ~seed();
		private:
			seed();
			static std::shared_ptr<seed> _sd;
			static std::mutex _mutex;
			static bool _fixed;
			static unsigned int _value;
			std::random_device _seed;
			std::mt19937 _pseudo;

	};
}

#endif
