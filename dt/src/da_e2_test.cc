#include "da2.h"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <unistd.h>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

using namespace npbnlp;

std::string output_dir;

template<class T, class V>
struct test_da : da<T, V> {
	using da<T, V>::_base;
	using da<T, V>::_check;
	using da<T, V>::_c;
	using da<T, V>::_nid;
	using da<T, V>::da;
};

std::string output_path(const char *name) {
	return output_dir + "/" + name;
}

template<class V>
void write_value(V& value, FILE *fp) {
	fwrite(&value, sizeof(V), 1, fp);
}

template<class V>
void read_value(V& value, FILE *fp) {
	if (fread(&value, sizeof(V), 1, fp) != 1)
		throw "failed to read value";
}

template<class T>
void write_key(T& value, FILE *fp) {
	fwrite(&value, sizeof(T), 1, fp);
}

template<class T>
void read_key(T& value, FILE *fp) {
	if (fread(&value, sizeof(T), 1, fp) != 1)
		throw "failed to read key";
}

template<class T>
std::vector<std::vector<T> > make_keys(uint32_t seed, int count, int alphabet) {
	std::mt19937 random(seed);
	std::uniform_int_distribution<int> symbol(1, alphabet);
	std::uniform_int_distribution<int> length(4, 12);
	std::set<std::vector<T> > unique;
	while ((int)unique.size() < count) {
		std::vector<T> key;
		key.emplace_back(static_cast<T>(1));
		key.emplace_back(static_cast<T>(2));
		int n = length(random);
		for (int i = 0; i < n; ++i)
			key.emplace_back(static_cast<T>(symbol(random)));
		key.emplace_back(static_cast<T>(0));
		unique.emplace(key);
	}
	return std::vector<std::vector<T> >(unique.begin(), unique.end());
}

bool detects_invalid_arrays(const std::vector<int32_t>& base, const std::vector<int32_t>& check, int32_t nid) {
	try {
		validate_da_arrays(base, check, nid);
	} catch (const char *) {
		return true;
	}
	return false;
}

bool detects_bad_rebuild(const std::vector<int32_t>& base, const std::vector<int32_t>& check, const std::vector<int32_t>& bad_base, const std::vector<int32_t>& bad_check, int32_t nid) {
	try {
		verify_compacted_arrays(base, check, bad_base, bad_check, nid);
	} catch (const char *) {
		return true;
	}
	return false;
}

int verify_failure_injection(const test_da<unsigned int, int>& trie) {
	const auto& base = trie._base;
	const auto& check = trie._check;
	int32_t nid = trie._nid;
	std::vector<int32_t> changed_base = base;
	std::vector<int32_t> changed_check = check;
	int32_t terminal = -1;
	int32_t parent = -1;
	for (int32_t p = 0; p < static_cast<int32_t>(base.size()) && terminal < 0; ++p) {
		if (base[p] < 0)
			continue;
		for (int32_t edge = 1; edge <= 257; ++edge) {
			int64_t child = static_cast<int64_t>(base[p]) + edge;
			if (child >= 0 && child < static_cast<int64_t>(check.size()) && check[child] == p) {
				if (edge == 1) {
					terminal = static_cast<int32_t>(child);
					parent = p;
					break;
				}
			}
		}
	}
	if (terminal < 0)
		return 1;
	changed_check[terminal] = -1;
	if (!detects_bad_rebuild(base, check, changed_base, changed_check, nid))
		return 2;
	changed_base = base;
	changed_check = check;
	int32_t extra = -1;
	for (int32_t edge = 1; edge <= 257; ++edge) {
		int64_t child = static_cast<int64_t>(base[0]) + edge;
		if (child >= 0 && child < static_cast<int64_t>(check.size()) && check[child] < 0) {
			extra = static_cast<int32_t>(child);
			break;
		}
	}
	if (extra < 0)
		return 3;
	changed_check[extra] = 0;
	changed_base[extra] = 0;
	if (!detects_bad_rebuild(base, check, changed_base, changed_check, nid))
		return 4;
	changed_base = base;
	changed_check = check;
	changed_base[terminal] = -(-base[terminal] == 1 ? 2 : 1);
	if (!detects_bad_rebuild(base, check, changed_base, changed_check, nid))
		return 5;
	changed_check = check;
	changed_check[terminal] = static_cast<int32_t>(check.size());
	if (!detects_invalid_arrays(base, changed_check, nid))
		return 6;
	changed_check = check;
	changed_base = base;
	changed_base[parent] = static_cast<int32_t>(base.size());
	if (!detects_invalid_arrays(changed_base, changed_check, nid))
		return 7;
	changed_base = base;
	changed_base[terminal] = -nid;
	if (!detects_invalid_arrays(changed_base, changed_check, nid))
		return 8;
	return 0;
}

template<class T>
int run_type(const char *path, int alphabet) {
	std::vector<std::vector<T> > keys = make_keys<T>(1234567, 10000, alphabet);
	std::vector<int> values;
	std::map<std::vector<T>, int> expected;
	for (int i = 0; i < (int)keys.size(); ++i) {
		values.emplace_back(i + 1);
		expected[keys[i]] = i + 1;
	}

	test_da<T, int> inserted(static_cast<T>(0));
	for (size_t i = 0; i < keys.size(); ++i)
		inserted.insert(keys[i], values[i]);
	const char *type_name = std::is_same<T, char>::value ? "char" : "unsigned_int";
	std::cerr << type_name << "\ttraining_cells\t" << inserted._base.size() << std::endl;
	int32_t training_used_cells = 0;
	for (int32_t check : inserted._check) {
		if (check >= 0)
			++training_used_cells;
	}
	std::cerr << type_name << "\ttraining_used_cells\t" << training_used_cells << std::endl;
	if constexpr (std::is_same<T, unsigned int>::value) {
		int failure = verify_failure_injection(inserted);
		if (failure != 0)
			return 15 + failure;
	}
	for (const auto& entry : expected) {
		std::vector<T> key = entry.first;
		if (inserted.exactmatch(key) <= 0)
			return 1;
	}
	for (size_t i = 0; i < keys.size(); i += 7) {
		inserted.erase(keys[i]);
		expected.erase(keys[i]);
	}
	for (const auto& key : keys) {
		int found = inserted.exactmatch(const_cast<std::vector<T>&>(key));
		auto expected_value = expected.find(key);
		if ((expected_value == expected.end()) != (found < 0))
			return 2;
		if (expected_value != expected.end() && inserted.getval(found).value() != expected_value->second)
			return 3;
	}
	std::vector<int> ids_before_save(keys.size(), -1);
	for (size_t i = 0; i < keys.size(); ++i)
		ids_before_save[i] = inserted.exactmatch(keys[i]);
	if (inserted.save(path, write_value<int>, write_key<T>) != 0)
		return 3;
	std::string compressed_path = std::string(path) + ".compressed";
	if (compress(path, compressed_path.c_str()) != 0)
		return 4;
	std::ifstream compacted_file(path, std::ios::binary);
	compacted_file.seekg(28);
	int32_t compacted_cells = 0;
	compacted_file.read(reinterpret_cast<char *>(&compacted_cells), sizeof(compacted_cells));
	compacted_file.seekg(32 + static_cast<std::streamoff>(compacted_cells) * sizeof(int32_t));
	int32_t check_cells = 0;
	compacted_file.read(reinterpret_cast<char *>(&check_cells), sizeof(check_cells));
	std::vector<int32_t> compacted_check(check_cells);
	compacted_file.read(reinterpret_cast<char *>(compacted_check.data()), check_cells * sizeof(int32_t));
	int32_t compacted_used_cells = 0;
	for (int32_t check : compacted_check) {
		if (check >= 0)
			++compacted_used_cells;
	}
	std::cerr << type_name << "\tcompacted_used_cells\t" << compacted_used_cells << std::endl;
	compacted_file.seekg(0, std::ios::end);
	std::cerr << type_name << "\tcompacted_cells\t" << compacted_cells << std::endl;
	std::cerr << type_name << "\tcompacted_bytes\t" << compacted_file.tellg() << std::endl;
	for (size_t i = 0; i < keys.size(); ++i) {
		if (inserted.exactmatch(keys[i]) != ids_before_save[i])
			return 5;
	}

	test_da<T, int> built(static_cast<T>(0));
	if (built.build(keys, values) != 0)
		return 5;
	if (std::is_same<T, unsigned int>::value && built._c.id() <= 16384)
		return 6;
	for (size_t i = 0; i < keys.size(); ++i) {
		if (built.exactmatch(keys[i]) <= 0)
			return 6;
	}
	if (built.save(path, write_value<int>, write_key<T>) != 0)
		return 7;
	test_da<T, int> loaded(static_cast<T>(0));
	if (loaded.load(compressed_path.c_str(), read_value<int>, read_key<T>) != 0)
		return 8;
	std::string compressed_again = compressed_path + ".again";
	if (compress(compressed_path.c_str(), compressed_again.c_str()) != 0)
		return 8;
	test_da<T, int> loaded_again(static_cast<T>(0));
	if (loaded_again.load(compressed_again.c_str(), read_value<int>, read_key<T>) != 0)
		return 8;
	for (size_t i = 0; i < keys.size(); ++i) {
		int id = loaded.exactmatch(keys[i]);
		auto expected_value = expected.find(keys[i]);
		if (expected_value == expected.end()) {
			if (id >= 0)
				return 9;
		} else if (id <= 0 || loaded.getval(id).value() != expected_value->second) {
			return 9;
		}
		if (loaded_again.exactmatch(keys[i]) != id)
			return 9;
	}
	for (size_t i = 0; i < 100; ++i) {
		std::vector<T> query = keys[i];
		auto actual = loaded.cp_search(query);
		std::vector<std::pair<int, long> > expected_prefixes;
		for (size_t j = 0; j < keys.size(); ++j) {
			if (expected.find(keys[j]) == expected.end())
				continue;
			if (keys[j].size() > query.size())
				continue;
			if (std::equal(keys[j].begin(), keys[j].end(), query.begin()))
				expected_prefixes.emplace_back(keys[j].size() - 1, loaded.exactmatch(const_cast<std::vector<T>&>(keys[j])));
		}
		std::sort(expected_prefixes.begin(), expected_prefixes.end());
		if (actual != expected_prefixes)
			return 10;
	}
	std::vector<T> added = keys.back();
	added.pop_back();
	added.emplace_back(static_cast<T>(123));
	added.emplace_back(static_cast<T>(0));
	int added_value = 100001;
	loaded.insert(added, added_value);
	int added_id = loaded.exactmatch(added);
	if (added_id <= 0 || loaded.getval(added_id).value() != added_value)
		return 11;
	return 0;
}

int main(int argc, char **argv) {
	char temporary_dir[] = "/tmp/da_e2_test.XXXXXX";
	bool temporary_output = argc == 1;
	if (temporary_output) {
		char *created = mkdtemp(temporary_dir);
		if (created == NULL)
			return 14;
		output_dir = created;
	} else {
		output_dir = argv[1];
	}
	std::filesystem::create_directories(output_dir);
	try {
	int char_result = run_type<char>(output_path("da_e2_char.idx").c_str(), 255);
	if (char_result != 0) {
		std::cerr << "char DA test failed: " << char_result << std::endl;
		return char_result;
	}
	int uint_result = run_type<unsigned int>(output_path("da_e2_uint.idx").c_str(), 17000);
	if (uint_result != 0) {
		std::cerr << "unsigned int DA test failed: " << uint_result << std::endl;
		return uint_result;
	}
	FILE *legacy = fopen(output_path("legacy.idx").c_str(), "wb");
	if (legacy == NULL)
		return 12;
	char legacy_header[8] = {};
	fwrite(legacy_header, sizeof(legacy_header), 1, legacy);
	fclose(legacy);
	da<char, int> rejected(static_cast<char>(0));
	bool legacy_rejected = false;
	try {
		rejected.load(output_path("legacy.idx").c_str(), read_value<int>, read_key<char>);
	} catch (const char *message) {
		legacy_rejected = std::string(message) == "legacy da format: convert with da_convert";
	}
	if (!legacy_rejected)
		return 13;
	} catch (const char *message) {
		std::cerr << message << std::endl;
		return 10;
	}
	if (temporary_output)
		std::filesystem::remove_all(output_dir);
	return 0;
}
