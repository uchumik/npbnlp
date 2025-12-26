#ifndef NPBNLP_CHUNKTYPE_H
#define NPBNLP_CHUNKTYPE_H

#include"wordtype.h"

namespace npbnlp {
	enum chtype {
		CH_HIRAGANA,	// chartype:0
		CH_KATAKANA,	// chartype:1
		CH_HANJI,	// chartype:3
		CH_LATIN,	// chartype:13
		CH_DIGIT,	// chartype:16
		CH_PUNC,	// chartype:17
		CH_SYNBOL,	// chartype:18
		CH_HIRA_KATA,	// chartype:0, 1
		CH_HIRA_HANJI,	// chartype:0, 3 wordtype:5
		CH_HIRA_DIGIT,	// chartype:0, 16
		CH_HIRA_PUNC,	// chartype:0, 17
		CH_KATA_HANJI,	// chartype:1, 3 wordtype:6
		CH_KATA_LATIN,	// chartype:1, 13
		CH_KATA_DIGIT,	// chartype:1, 16
		CH_KATA_PUNC,	// chartype:1, 17
		CH_HANJI_LATIN,	// chartype:3, 13
		CH_HANJI_DIGIT,	// chartype:3, 16
		CH_HANJI_PUNC,	// chartype:3, 17
		CH_LATIN_DIGIT,	// chartype:13, 16
		CH_LATIN_PUNC,	// chartype:13, 17
		CH_LATIN_SYNBOL,	// chartype:13, 18
		CH_DIGIT_PUNC,	// chartype:16, 17
		CH_HIRA_KATA_HANJI,	// chartype:0, 1, 3
		CH_HIRA_HANJI_DIGIT,	// chartype:0, 3, 16
		CH_HIRA_HANJI_PUNC,	// chartype:0, 3, 17
		CH_KATA_HANJI_DIGIT,	// chartype:1, 3, 16
		CH_KATA_HANJI_PUNC,	// chartype:1, 3, 17
		CH_KATA_LATIN_PUNC,	// chartype:1, 13, 17
		CH_KATA_DIGIT_PUNC,	// chartype:1, 16, 17
		CH_HANJI_DIGIT_PUNC,	// chartype:3, 16, 17
		CH_LATIN_DIGIT_PUNC,	// chartype:13, 16, 17
		CH_MISC,
		CH_NONE = -1
	};
	class chunktype2 {
		public:
			const static int n = 32;
			constexpr static int chunk_transition[19][19] = {
				/*0*/ {1,1,0,1,1,1,1,0,1,0,0,0,0,1,0,0,1,1,1},
				/*1*/{1,1,0,1,1,1,1,1,1,0,0,0,0,1,0,0,1,1,1},
				/*2*/{},
				/*3*/{1,1,0,1,1,1,1,1,1,0,0,0,0,1,0,0,1,1,1},
				/*4*/{1,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1},
				/*5*/{1,1,0,1,0,1,1,0,1,0,0,0,0,1,0,0,1,1,1},
				/*6*/{1,1,0,1,0,1,0,0,1,0,0,0,0,1,0,0,1,1,1},
				/*7*/{0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0},
				/*8*/{1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1},
				/*9*/{},
				/*10*/{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
				/*11*/{},
				/*12*/{},
				/*13*/{1,1,0,1,0,1,0,0,1,0,0,0,0,1,0,0,1,1,1},
				/*14*/{},
				/*15*/{},
				/*16*/{1,1,0,1,0,1,1,0,1,0,0,0,0,1,0,0,1,1,1},
				/*17*/{1,1,0,1,1,1,1,0,1,0,0,0,0,1,0,0,1,1,1},
				/*18*/{1,1,0,1,0,1,0,0,1,0,0,0,0,1,0,0,1,1,1}

			};
			static chtype get(chunk& c) {
				chtype t = start(c.wd(c.len-1));
				for (auto i = c.len-1; i > 0; --i) {
					type cur = wordtype::get(c.wd(i));
					type prev = wordtype::get(c.wd(i-1));
					t = transition(t, prev, cur);
				}
				return t;
			}
			static chtype start(word& w) {
				type t = wordtype::get(w);
				return start(t);
			}
			static chtype start(type& t) {
				//type t = wordtype::get(w);
				switch (t) {
					case U_HIRAGANA:
						return CH_HIRAGANA;
						break;
					case U_KATAKANA:
						return CH_KATAKANA;
						break;
					case U_HANJI:
						return CH_HANJI;
						break;
					case U_LATIN:
						return CH_LATIN;
						break;
					case U_DIGIT:
						return CH_DIGIT;
						break;
					case U_PUNC:
						return CH_PUNC;
						break;
					case U_SYNBOL:
						return CH_SYNBOL;
						break;
					case U_KATA_OR_HIRA:
						return CH_HIRAGANA;
						break;
					case U_HIRA_KATA:
						return CH_HIRA_KATA;
						break;
					case U_HIRA_HANJI:
						return CH_HIRA_HANJI;
						break;
					case U_KATA_HANJI:
						return CH_KATA_HANJI;
						break;
					case U_HIRA_KATA_HANJI:
						return CH_HIRA_KATA_HANJI;
						break;
					case U_MISC:
						return CH_MISC;
						break;
					default:
						return CH_SYNBOL;
				}
			}
			static chtype transition(chtype& ct, type& prev, type& cur) {
				if (!chunk_transition[cur][prev])
					return CH_NONE;
				switch (ct) {
					case CH_HIRAGANA:
						if (prev == U_HIRAGANA) {
							return CH_HIRAGANA;
						} else if (prev == U_KATAKANA || prev == U_HIRA_KATA) {
							return CH_HIRA_KATA;
						} else if (prev == U_HANJI || prev == U_HIRA_HANJI) {
							return CH_HIRA_HANJI;
						} else if (prev == U_DIGIT) {
							return CH_HIRA_DIGIT;
						} else if (prev == U_PUNC) {
							return CH_HIRA_PUNC;
						} else if (prev == U_HIRA_KATA_HANJI) {
							return CH_HIRA_KATA_HANJI;
						}else {
							return CH_MISC;
						}
						break;
					case CH_KATAKANA:
						if (prev == U_HIRAGANA || prev == U_HIRA_KATA) {
							return CH_HIRA_KATA;
						} else if (prev == U_KATAKANA) {
							return CH_KATAKANA;
						} else if (prev == U_HANJI || prev == U_KATA_HANJI) {
							return CH_KATA_HANJI;
						} else if (prev == U_LATIN) {
							return CH_KATA_LATIN;
						} else if (prev == U_DIGIT) {
							return CH_KATA_DIGIT;
						} else if (prev == U_PUNC) {
							return CH_KATA_PUNC;
						} else if (prev == U_HIRA_KATA_HANJI) {
							return CH_HIRA_KATA_HANJI;
						} else {
							return CH_MISC;
						}
						break;
					case CH_HANJI:
						if (prev == U_HIRAGANA || prev == U_HIRA_HANJI) {
							return CH_HIRA_HANJI;
						} else if (prev == U_KATAKANA) {
							return CH_KATA_HANJI;
						} else if (prev == U_HANJI) {
							return CH_HANJI;
						} else if (prev == U_LATIN) {
							return CH_HANJI_LATIN;
						} else if (prev == U_DIGIT) {
							return CH_HANJI_DIGIT;
						} else if (prev == U_PUNC) {
							return CH_HANJI_PUNC;
						} else if (prev == U_HIRA_KATA_HANJI) {
							return CH_HIRA_KATA_HANJI;
						} else {
							return CH_MISC;
						}
						break;
					case CH_LATIN:
						if (prev == U_KATAKANA) {
							return CH_KATA_LATIN;
						} else if (prev == U_HANJI) {
							return CH_HANJI_LATIN;
						} else if (prev == U_LATIN) {
							return CH_LATIN;
						} else if (prev == U_DIGIT) {
							return CH_LATIN_DIGIT;
						} else if (prev == U_PUNC) {
							return CH_LATIN_PUNC;
						} else if (prev == U_SYNBOL) {
							return CH_LATIN_SYNBOL;
						} else {
							return CH_MISC;
						}
						break;
					case CH_DIGIT:
						if (prev == U_HIRAGANA) {
							return CH_HIRA_DIGIT;
						} else if (prev == U_KATAKANA) {
							return CH_KATA_DIGIT;
						} else if (prev == U_HANJI) {
							return CH_HANJI_DIGIT;
						} else if (prev == U_LATIN) {
							return CH_LATIN_DIGIT;
						} else if (prev == U_DIGIT) {
							return CH_DIGIT;
						} else if (prev == U_PUNC) {
							return CH_DIGIT_PUNC;
						} else if (prev == U_HIRA_HANJI) {
							return CH_HIRA_HANJI_DIGIT;
						} else if (prev == U_KATA_HANJI) {
							return CH_KATA_HANJI_DIGIT;
						} else {
							return CH_MISC;
						}
						break;
					case CH_PUNC:
						if (prev == U_HIRAGANA) {
							return CH_HIRA_PUNC;
						} else if (prev == U_KATAKANA) {
							return CH_KATA_PUNC;
						} else if (prev == U_HANJI) {
							return CH_HANJI_PUNC;
						} else if (prev == U_LATIN) {
							return CH_LATIN_PUNC;
						} else if (prev == U_DIGIT) {
							return CH_DIGIT_PUNC;
						} else if (prev == U_PUNC) {
							return CH_PUNC;
						} else if (prev == U_HIRA_HANJI) {
							return CH_HIRA_HANJI_PUNC;
						} else if (prev == U_KATA_HANJI) {
							return CH_KATA_HANJI_PUNC;
						} else {
							return CH_MISC;
						}
						break;
					case CH_SYNBOL:
						if (prev == U_LATIN) {
							return CH_LATIN_SYNBOL;
						} else if (prev == U_SYNBOL) {
							return CH_SYNBOL;
						} else {
							return CH_MISC;
						}
						break;
					case CH_HIRA_KATA:
						if (prev == U_HIRAGANA || prev == U_KATAKANA || prev == U_HIRA_KATA) {
							return CH_HIRA_KATA;
						} else if (prev == U_HANJI) {
							return CH_HIRA_KATA_HANJI;
						} else {
							return CH_MISC;
						}
						break;
					case CH_HIRA_HANJI:
						if (prev == U_HIRAGANA || prev == U_HANJI || prev == U_HIRA_HANJI) {
							return CH_HIRA_HANJI;
						} else if (prev == U_KATAKANA || prev == U_KATA_HANJI) {
							return CH_HIRA_KATA_HANJI;
						} else if (prev == U_DIGIT) {
							return CH_HIRA_HANJI_DIGIT;
						} else if (prev == U_PUNC) {
							return CH_HIRA_HANJI_PUNC;
						} else {
							return CH_MISC;
						}
						break;
					case CH_HIRA_DIGIT:
						if (prev == U_HIRAGANA || prev == U_DIGIT) {
							return CH_HIRA_DIGIT;
						} else if (prev == U_HANJI) {
							return CH_HIRA_HANJI_DIGIT;
						} else {
							return CH_MISC;
						}
						break;
					case CH_HIRA_PUNC:
						if (prev == U_HIRAGANA || prev == U_PUNC) {
							return CH_HIRA_PUNC;
						} else if (prev == U_HANJI) {
							return CH_HIRA_HANJI_PUNC;
						} else {
							return CH_MISC;
						}
						break;
					case CH_KATA_HANJI:
						if (prev == U_KATAKANA || prev == U_HANJI || prev == U_KATA_HANJI) {
							return CH_KATA_HANJI;
						} else if (prev == U_DIGIT) {
							return CH_KATA_HANJI_DIGIT;
						} else if (prev == U_PUNC) {
							return CH_KATA_HANJI_PUNC;
						} else {
							return CH_MISC;
						}
						break;
					case CH_KATA_LATIN:
						if (prev == U_KATAKANA || prev == U_LATIN) {
							return CH_KATA_LATIN;
						} else if (prev == U_PUNC) {
							return CH_KATA_LATIN_PUNC;
						} else {
							return CH_MISC;
						}
						break;
					case CH_KATA_DIGIT:
						if (prev == U_KATAKANA || prev == U_DIGIT) {
							return CH_KATA_DIGIT;
						} else if (prev == U_PUNC) {
							return CH_KATA_DIGIT_PUNC;
						} else {
							return CH_MISC;
						}
						break;
					case CH_KATA_PUNC:
						if (prev == U_KATAKANA || prev == U_PUNC) {
							return CH_KATA_PUNC;
						} else if (prev == U_HANJI) {
							return CH_KATA_HANJI_PUNC;
						} else if (prev == U_LATIN) {
							return CH_KATA_LATIN_PUNC;
						} else if (prev == U_DIGIT) {
							return CH_KATA_DIGIT_PUNC;
						} else {
							return CH_MISC;
						}
						break;
					case CH_HANJI_LATIN:
						if (prev == U_HANJI || prev == U_LATIN) {
							return CH_HANJI_LATIN;
						} else {
							return CH_MISC;
						}
						break;
					case CH_HANJI_DIGIT:
						if (prev == U_HANJI || prev == U_DIGIT) {
							return CH_HANJI_DIGIT;
						} else if (prev == U_PUNC) { 
							return CH_HANJI_DIGIT_PUNC;
						} else if (prev == U_HIRAGANA) {
							return CH_HIRA_HANJI_DIGIT;
						} else {
							return CH_MISC;
						}
						break;
					case CH_HANJI_PUNC:
						if (prev == U_HANJI || prev == U_PUNC) {
							return CH_HANJI_PUNC;
						} else if (prev == U_DIGIT) {
							return CH_HANJI_DIGIT_PUNC;
						} else if (prev == U_HIRAGANA) {
							return CH_HIRA_HANJI_PUNC;
						} else {
							return CH_MISC;
						}
						break;
					case CH_LATIN_DIGIT:
						if (prev == U_LATIN || prev == U_DIGIT) {
							return CH_LATIN_DIGIT;
						} else if (prev == U_PUNC) {
							return CH_LATIN_DIGIT_PUNC;
						} else {
							return CH_MISC;
						}
						break;
					case CH_LATIN_PUNC:
						if (prev == U_LATIN || prev == U_PUNC) {
							return CH_LATIN_PUNC;
						} else if (prev == U_KATAKANA) {
							return CH_KATA_LATIN_PUNC;
						} else if (prev == U_DIGIT) {
							return CH_LATIN_DIGIT_PUNC;
						} else {
							return CH_MISC;
						}
						break;
					case CH_LATIN_SYNBOL:
						if (prev == U_LATIN || prev == U_SYNBOL) {
							return CH_LATIN_SYNBOL;
						} else {
							return CH_MISC;
						}
						break;
					case CH_DIGIT_PUNC:
						if (prev == U_DIGIT || prev == U_PUNC) {
							return CH_DIGIT_PUNC;
						} else if (prev == U_KATAKANA) {
							return CH_KATA_DIGIT_PUNC;
						} else if (prev == U_HANJI) {
							return CH_HANJI_DIGIT_PUNC;
						} else {
							return CH_MISC;
						}
						break;
					case CH_HIRA_KATA_HANJI:
						if (prev == U_HIRAGANA || prev == U_KATAKANA || prev == U_HANJI || prev == U_HIRA_KATA_HANJI) {
							return CH_HIRA_KATA_HANJI;
						} else {
							return CH_MISC;
						}
						break;
					case CH_HIRA_HANJI_DIGIT:
						if (prev == U_HIRAGANA || prev == U_HANJI || prev == U_DIGIT) {
							return CH_HIRA_HANJI_DIGIT;
						} else {
							return CH_MISC;
						}
						break;
					case CH_HIRA_HANJI_PUNC:
						if (prev == U_HIRAGANA || prev == U_HANJI || prev == U_PUNC) {
							return CH_HIRA_HANJI_PUNC;
						} else {
							return CH_MISC;
						}
						break;
					case CH_KATA_HANJI_DIGIT:
						if (prev == U_KATAKANA || prev == U_HANJI || prev == U_DIGIT) {
							return CH_KATA_HANJI_DIGIT;
						} else {
							return CH_MISC;
						}
						break;
					case CH_KATA_HANJI_PUNC:
						if (prev == U_KATAKANA || prev == U_HANJI || prev == U_PUNC) {
							return CH_KATA_HANJI_PUNC;
						} else {
							return CH_MISC;
						}
						break;
					case CH_KATA_LATIN_PUNC:
						if (prev == U_KATAKANA || prev == U_LATIN || prev == U_PUNC) {
							return CH_KATA_LATIN_PUNC;
						} else {
							return CH_MISC;
						}
						break;
					case CH_KATA_DIGIT_PUNC:
						if (prev == U_KATAKANA || prev == U_DIGIT || prev == U_PUNC) {
							return CH_KATA_DIGIT_PUNC;
						} else {
							return CH_MISC;
						}
						break;
					case CH_HANJI_DIGIT_PUNC:
						if (prev == U_HANJI || prev == U_DIGIT || prev == U_PUNC) {
							return CH_HANJI_DIGIT_PUNC;
						} else {
							return CH_MISC;
						}
						break;
					case CH_LATIN_DIGIT_PUNC:
						if (prev == U_LATIN || prev == U_DIGIT || prev == U_PUNC) {
							return CH_LATIN_DIGIT_PUNC;
						} else {
							return CH_MISC;
						}
						break;
					case CH_MISC:
						return CH_MISC;
						//return CH_NONE;
						break;
					default:
						return CH_NONE;
				}
			}
	};
	class chunktype {
		public:
			static type get(chunk& c) {
				type t = wordtype::get(c.wd(0));
				for (auto i = 1; i < c.len; ++i) {
					type u = wordtype::get(c.wd(i));
					chunktype::change(t, u);
				}
				return t;
			}
			static void change(type& t, type& u) {
				switch (u) {
					case U_HIRAGANA:
						if (t == U_KATAKANA)
							t = U_HIRA_KATA;
						else if (t == U_HANJI)
							t = U_HIRA_HANJI;
						else if (t == U_KATA_HANJI)
							t = U_HIRA_KATA_HANJI;
						else if (t == U_KATA_OR_HIRA)
							t = U_HIRAGANA;
						else if (t == U_PUNC)
							break;
						else if (t == U_HIRA_HANJI || t == U_HIRA_KATA || t == U_HIRA_KATA_HANJI) {
						} else if (u != t)
							t = U_MISC;
						break;
					case U_KATAKANA:
						if (t == U_HIRAGANA)
							t = U_HIRA_KATA;
						else if (t == U_HANJI)
							t = U_KATA_HANJI;
						else if (t == U_HIRA_HANJI)
							t = U_HIRA_KATA_HANJI;
						else if (t == U_KATA_OR_HIRA)
							t = U_KATAKANA;
						else if (t == U_PUNC)
							break;
						else if (t == U_HIRA_KATA || t == U_KATA_HANJI || t == U_HIRA_KATA_HANJI) {
						} else if (u != t)
							t = U_MISC;
						break;
					case U_HANJI:
						if (t == U_HIRAGANA)
							t = U_HIRA_HANJI;
						else if (t == U_KATAKANA)
							t = U_KATA_HANJI;
						else if (t == U_HIRA_KATA || t == U_KATA_OR_HIRA)
							t = U_HIRA_KATA_HANJI;
						else if (t == U_PUNC)
							break;
						else if (t == U_HIRA_HANJI || t == U_KATA_HANJI || t == U_HIRA_KATA_HANJI) {
						} else if (u != t)
							t = U_MISC;
						break;
					case U_KATA_OR_HIRA:
						if (t != U_HIRAGANA && t != U_KATAKANA && t != U_HIRA_KATA && t != U_KATA_HANJI && t != U_HIRA_HANJI && t != U_HIRA_KATA_HANJI)
							t = U_MISC;
						else if (t == U_PUNC)
							break;
						else if (u != t)
							t = U_MISC;
						break;
					case U_HIRA_KATA:
						if (t == U_HIRAGANA || t == U_KATAKANA)
							t = u;
						else if (t == U_HANJI)
							t = U_HIRA_KATA_HANJI;
						else if (t == U_PUNC)
							break;
						else if (t != u)
							t = U_MISC;
						break;
					case U_HIRA_HANJI:
						if (t == U_HIRAGANA || t == U_HANJI)
							t = u;
						else if (t == U_KATAKANA)
							t = U_HIRA_KATA_HANJI;
						else if (t == U_PUNC)
							break;
						else if (t != u)
							t = U_MISC;
						break;
					case U_KATA_HANJI:
						if (t == U_KATAKANA || t == U_HANJI)
							t = u;
						else if (t == U_HIRAGANA)
							t = U_HIRA_KATA_HANJI;
						else if (t == U_PUNC)
							break;
						else if (t != u)
							t = U_MISC;
						break;
					case U_HIRA_KATA_HANJI:
						if (t == U_HIRAGANA || t == U_KATAKANA || t == U_HANJI)
							t = u;
						else if (t == U_PUNC)
							break;
						else if (t != u)
							t = U_MISC;
						break;
					case U_LATIN:
						if (t == U_KATAKANA)
							break;
						else if (t == U_PUNC)
							t = U_LATIN;
						else if (t != u)
							t = U_MISC;
						break;
					case U_SYNBOL:
						if (t == U_LATIN || t == U_DIGIT)
							break;
						else if (t != u)
							t = U_MISC;
						break;
					case U_PUNC:
						if (t == U_LATIN || t == U_DIGIT || t == U_KATAKANA)
							break;
						else if (t != U_MISC)
							t = U_PUNC;
						break;
					default:
						if (t != U_PUNC && t != u) {
							t = U_MISC;
						}
				}
			}
	};
}
#endif
