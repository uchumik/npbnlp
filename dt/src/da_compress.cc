#include "da2.h"
#include <chrono>
#include <cstdio>
#include <iostream>
#include <sys/stat.h>
#include <vector>

using namespace npbnlp;

struct array_stats {
	int32_t cells;
	int32_t used;
	int32_t reachable;
};

array_stats inspect(const char *path) {
	FILE *fp = fopen(path, "rb");
	if (fp == nullptr)
		throw "failed to open model for inspection";
	char magic[8];
	int32_t version = 0;
	int32_t header[4];
	int32_t bsize = 0;
	if (fread(magic, 1, sizeof(magic), fp) != sizeof(magic) || fread(&version, sizeof(version), 1, fp) != 1 || fread(header, sizeof(int32_t), 4, fp) != 4 || fread(&bsize, sizeof(bsize), 1, fp) != 1) {
		fclose(fp);
		throw "failed to read model header";
	}
	const char expected[8] = {'D', 'A', '2', 'E', '2', 'I', '3', '2'};
	if (std::memcmp(magic, expected, sizeof(expected)) != 0 || version != 1 || bsize < 2) {
		fclose(fp);
		throw "invalid model header";
	}
	std::vector<int32_t> base(bsize);
	int32_t csize = 0;
	if (fread(base.data(), sizeof(int32_t), bsize, fp) != static_cast<size_t>(bsize) || fread(&csize, sizeof(csize), 1, fp) != 1 || csize != bsize) {
		fclose(fp);
		throw "failed to read model base array";
	}
	std::vector<int32_t> check(csize);
	if (fread(check.data(), sizeof(int32_t), csize, fp) != static_cast<size_t>(csize)) {
		fclose(fp);
		throw "failed to read model check array";
	}
	if (fclose(fp) != 0)
		throw "failed to close inspected model";
	array_stats result = {bsize, 0, 0};
	for (int32_t i = 1; i < csize; ++i) {
		if (check[i] >= 0)
			++result.used;
	}
	std::vector<int32_t> stack(1, 0);
	std::vector<bool> seen(csize, false);
	seen[0] = true;
	while (!stack.empty()) {
		int32_t parent = stack.back();
		stack.pop_back();
		++result.reachable;
		for (int32_t edge = 1; edge <= 257; ++edge) {
			int64_t child = static_cast<int64_t>(base[parent]) + edge;
			if (child >= 0 && child < csize && check[child] == parent && !seen[child]) {
				seen[child] = true;
				stack.emplace_back(static_cast<int32_t>(child));
			}
		}
	}
	return result;
}

int main(int argc, char **argv) {
	if (argc != 3) {
		std::cerr << "usage: da_compress <input> <output>\n";
		return 2;
	}
	struct stat before;
	if (stat(argv[1], &before) != 0) {
		std::cerr << "da_compress: failed to stat input\n";
		return 1;
	}
	auto start = std::chrono::steady_clock::now();
	try {
		compress(argv[1], argv[2]);
	} catch (const char *message) {
		std::cerr << "da_compress: " << message << "\n";
		return 1;
	}
	struct stat after;
	if (stat(argv[2], &after) != 0) {
		std::cerr << "da_compress: failed to stat output\n";
		return 1;
	}
	array_stats input_stats;
	array_stats output_stats;
	try {
		input_stats = inspect(argv[1]);
		output_stats = inspect(argv[2]);
	} catch (const char *message) {
		std::cerr << "da_compress: " << message << "\n";
		return 1;
	}
	auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	std::cerr << "input_bytes\t" << before.st_size << "\n";
	std::cerr << "output_bytes\t" << after.st_size << "\n";
	std::cerr << "input_cells\t" << input_stats.cells << "\n";
	std::cerr << "output_cells\t" << output_stats.cells << "\n";
	std::cerr << "input_used_cells\t" << input_stats.used << "\n";
	std::cerr << "output_used_cells\t" << output_stats.used << "\n";
	std::cerr << "input_reachable_cells\t" << input_stats.reachable << "\n";
	std::cerr << "output_reachable_cells\t" << output_stats.reachable << "\n";
	std::cerr << "seconds\t" << elapsed << "\n";
	return 0;
}
