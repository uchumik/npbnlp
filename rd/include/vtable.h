#ifndef NPBNLP_VTABLE_H
#define NPBNLP_VTABLE_H

#include<mutex>
#include<memory>
#include<unordered_map>
namespace npbnlp {
	class vt;
	using vtable = std::unordered_map<int, std::shared_ptr<vt> >;
	class vt {
		public:
			vt();
			vt(int n, int id, vt *p);
			virtual ~vt();
			vt& operator[](int i);
			vt* get_parent();
			double v;
			int n;
			int id;
			bool is_init();
			void set(bool f);
			vtable::iterator begin();
			vtable::iterator end();
		protected:
			vt *_parent;
			bool _init;
			std::shared_ptr<vtable> _t;
			std::mutex _mutex;

	};
}

#endif
