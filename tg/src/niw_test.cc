#include "niw.h"
#include "dense.h"

#include <cmath>
#include <iostream>
#include <random>

using namespace npbnlp;

static void check(bool condition, const char* message) {
	if (!condition) {
		throw message;
	}
}

static fvector vector_of(const std::vector<double>& values) {
	fvector result;
	result.v = values;
	return result;
}

static double relative_error(double actual, double expected) {
	return std::abs(actual - expected) / std::abs(expected);
}

static double max_relative_error(const std::vector<double>& actual,
	                               const std::vector<double>& expected) {
	double result = 0.0;
	for (size_t i = 0; i < actual.size(); ++i) {
		result = std::max(result, relative_error(actual[i], expected[i]));
	}
	return result;
}

static double max_abs_error(const std::vector<double>& actual,
	                          const std::vector<double>& expected) {
	double result = 0.0;
	for (size_t i = 0; i < actual.size(); ++i) {
		result = std::max(result, std::abs(actual[i] - expected[i]));
	}
	return result;
}

int main() {
	try {
		const int d = 3;
		const std::vector<double> mu = {1.0, -1.0, 0.5};
		const std::vector<double> covariance = {
			1.0, 0.7, 0.2,
			0.7, 2.0, 0.4,
			0.2, 0.4, 1.5
		};
		std::vector<double> lower;
		check(dense::chol(covariance, d, lower), "covariance is not positive definite");

		niw model(d);
		model.set_prior(mu, 0.1, d + 2.0, 1.0);
		model.resize(1);
		std::mt19937 random(7);
		std::normal_distribution<double> normal(0.0, 1.0);
		for (int n = 0; n < 10000; ++n) {
			std::vector<double> standard_normal(d, 0.0);
			for (int i = 0; i < d; ++i) {
				standard_normal[i] = normal(random);
			}
			std::vector<double> value(d, 0.0);
			for (int i = 0; i < d; ++i) {
				for (int j = 0; j <= i; ++j) {
					value[i] += lower[i * d + j] * standard_normal[j];
				}
				value[i] += mu[i];
			}
			model.add(0, vector_of(value));
		}
		std::vector<double> posterior_mean;
		model.posterior_sigma_mean(0, posterior_mean);
		std::cout << "1 posterior E[Sigma] vs true (all components):\n";
		for (int i = 0; i < d; ++i) {
			std::cout << "  " << posterior_mean[i * d] << " "
				<< posterior_mean[i * d + 1] << " "
				<< posterior_mean[i * d + 2] << "\n";
		}
		double covariance_error = max_relative_error(posterior_mean, covariance);
		std::cout << "  max relative error = " << covariance_error << "\n";
		check(covariance_error < 0.05, "posterior covariance error");
		std::cout << "1 covariance non-diagonal: PASS\n";

		niw student(1);
		student.set_prior({0.0}, 0.1, 3.0, 1.0);
		student.resize(1);
		fvector point = vector_of({0.3});
		double actual = student.lp(0, point);
		double degrees = 3.0;
		double scale = 11.0 / 3.0;
		double expected = std::lgamma((degrees + 1.0) / 2.0)
			- std::lgamma(degrees / 2.0)
			- 0.5 * std::log(degrees * M_PI)
			- 0.5 * std::log(scale)
			- (degrees + 1.0) / 2.0
				* std::log1p((0.09 / scale) / degrees);
		std::cout << "2 Student-t values: actual=" << actual
			<< " expected=" << expected
			<< " difference=" << std::abs(actual - expected) << "\n";
		check(std::abs(actual - expected) < 1e-10, "Student-t analytic mismatch");
		std::cout << "2 Student-t analytic: PASS\n";

		niw integral(2);
		integral.set_prior({0.0, 0.0}, 0.1, 4.0, 1.0);
		integral.resize(1);
		fvector grid_point;
		double integral_value = 0.0;
		for (int i = -1000; i <= 1000; ++i) {
			for (int j = -1000; j <= 1000; ++j) {
				grid_point.v = {i * 0.1, j * 0.1};
				integral_value += std::exp(integral.lp(0, grid_point)) * 0.01;
			}
		}
		std::cout << "3 numerical integral = " << integral_value << "\n";
		check(std::abs(integral_value - 1.0) < 1e-3, "Student-t normalization");
		std::cout << "3 normalization: PASS\n";

		niw symmetry(d);
		symmetry.set_prior(mu, 0.1, d + 2.0, 1.0);
		symmetry.resize(1);
		std::vector<fvector> observations;
		for (int i = 0; i < 100; ++i) {
			fvector value = vector_of({i * 0.1, -i * 0.03, i * 0.02});
			observations.push_back(value);
			symmetry.add(0, value);
		}
		for (auto it = observations.rbegin(); it != observations.rend(); ++it) {
			symmetry.remove(0, *it);
		}
		std::vector<double> sum;
		std::vector<double> cross_products;
		symmetry.sufficient_statistics(0, sum, cross_products);
		double sum_error = 0.0;
		for (double value : sum) {
			sum_error = std::max(sum_error, std::abs(value));
		}
		double cross_product_error = 0.0;
		for (double value : cross_products) {
			cross_product_error = std::max(cross_product_error, std::abs(value));
		}
		std::cout << "4 after reverse remove: count=" << symmetry.count(0)
			<< " max|s|=" << sum_error
			<< " max|S|=" << cross_product_error << "\n";
		check(symmetry.count(0) == 0, "remove count mismatch");
		check(sum_error < 1e-9 && cross_product_error < 1e-9,
			"remove sufficient statistics mismatch");
		std::cout << "4 add/remove symmetry: PASS\n";

		const std::vector<double> lambda0 = {
			2.0, 0.5, 0.2,
			0.5, 1.5, 0.3,
			0.2, 0.3, 1.2
		};
		niw sampler(d);
		sampler.set_prior({0.0, 0.0, 0.0}, 0.1, 8.0, lambda0);
		sampler.resize(1);
		sampler.set_sample_mode(true);
		std::vector<double> covariance_average(d * d, 0.0);
		for (int n = 0; n < 20000; ++n) {
			sampler.estimate();
			std::vector<double> sampled;
			sampler.sampled_sigma(0, sampled);
			for (int i = 0; i < d * d; ++i) {
				covariance_average[i] += sampled[i];
			}
		}
		for (double& value : covariance_average) {
			value /= 20000.0;
		}
		std::vector<double> sampler_expected = lambda0;
		for (double& value : sampler_expected) {
			value /= 8.0 - d - 1.0;
		}
		double sampler_error = max_relative_error(covariance_average, sampler_expected);
		std::cout << "5 Bartlett sample mean (all components):\n";
		for (int i = 0; i < d; ++i) {
			std::cout << "  " << covariance_average[i * d] << " "
				<< covariance_average[i * d + 1] << " "
				<< covariance_average[i * d + 2] << "\n";
		}
		std::cout << "  max relative error = " << sampler_error << "\n";
		check(sampler_error < 0.05, "Bartlett sample mean error");
		std::cout << "5 Bartlett sampler: PASS\n";

		niw consistency(d);
		consistency.set_prior(mu, 0.1, d + 2.0, covariance);
		consistency.resize(1);
		for (int n = 0; n < 5000; ++n) {
			consistency.add(0, vector_of({
				mu[0] + normal(random),
				mu[1] + normal(random),
				mu[2] + normal(random)
			}));
		}
		fvector comparison_point = vector_of({1.3, -0.7, 0.8});
		double marginalized = consistency.lp(0, comparison_point);
		consistency.set_sample_mode(true);
		consistency.estimate();
		double sampled_normal = consistency.lp(0, comparison_point);
		std::cout << "6 Student-t lp=" << marginalized
			<< " sampled normal lp=" << sampled_normal
			<< " absolute difference=" << std::abs(marginalized - sampled_normal) << "\n";
		check(std::abs(marginalized - sampled_normal) <= 0.5,
			"marginalized/sample consistency");
		std::cout << "6 marginalized/sample consistency: PASS\n";

		std::vector<double> matrix = {
			2.0, 0.3, 0.3,
			0.3, 1.5, 0.2,
			0.3, 0.2, 1.0
		};
		std::vector<double> matrix_lower;
		check(dense::chol(matrix, d, matrix_lower), "dense Cholesky");
		std::vector<double> vector = {1.0, 2.0, 3.0};
		std::vector<double> solved_lower;
		dense::solve_lower(matrix_lower, d, vector, solved_lower);
		std::vector<double> lower_reconstructed(d, 0.0);
		for (int i = 0; i < d; ++i) {
			for (int j = 0; j <= i; ++j) {
				lower_reconstructed[i] += matrix_lower[i * d + j] * solved_lower[j];
			}
		}
		std::vector<double> upper(d * d, 0.0);
		for (int i = 0; i < d; ++i) {
			for (int j = i; j < d; ++j) {
				upper[i * d + j] = matrix_lower[j * d + i];
			}
		}
		std::vector<double> solved_upper;
		dense::solve_upper(upper, d, vector, solved_upper);
		std::vector<double> upper_reconstructed(d, 0.0);
		for (int i = 0; i < d; ++i) {
			for (int j = i; j < d; ++j) {
				upper_reconstructed[i] += upper[i * d + j] * solved_upper[j];
			}
		}
		std::vector<double> factor = {
			1.2, 0.0, 0.0,
			0.4, 0.9, 0.0,
			-0.2, 0.3, 1.1
		};
		std::vector<double> product;
		dense::solve_rtri(matrix_lower, factor, d, product);
		std::vector<double> rtri_reconstructed(d * d, 0.0);
		for (int i = 0; i < d; ++i) {
			for (int j = 0; j < d; ++j) {
				for (int k = 0; k < d; ++k) {
					rtri_reconstructed[i * d + j] += product[i * d + k]
						* factor[j * d + k];
				}
			}
		}
		std::vector<double> xxt_result;
		dense::xxt(matrix_lower, d, xxt_result);
		std::vector<double> syr_result(d * d, 0.0);
		dense::syr(syr_result, d, vector, 1.0);
		std::vector<double> vector_outer(d * d, 0.0);
		for (int i = 0; i < d; ++i) {
			for (int j = 0; j < d; ++j) {
				vector_outer[i * d + j] = vector[i] * vector[j];
			}
		}
		double determinant = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
			- matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
			+ matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
		double logdet_error = std::abs(dense::logdet(matrix_lower, d) - std::log(determinant));
		double quad_expected = 0.0;
		for (double value : solved_lower) {
			quad_expected += value * value;
		}
		double quad_error = std::abs(dense::quad(matrix_lower, d, vector) - quad_expected);
		std::cout << "7 dense errors: lower=" << max_abs_error(lower_reconstructed, vector)
			<< " upper=" << max_abs_error(upper_reconstructed, vector)
			<< " solve_rtri=" << max_abs_error(rtri_reconstructed, matrix_lower)
			<< " xxt=" << max_abs_error(xxt_result, matrix)
			<< " syr=" << max_abs_error(syr_result, vector_outer)
			<< " logdet=" << logdet_error
			<< " quad=" << quad_error << "\n";
		check(max_abs_error(lower_reconstructed, vector) < 1e-10, "solve_lower");
		check(max_abs_error(upper_reconstructed, vector) < 1e-10, "solve_upper");
		check(max_abs_error(rtri_reconstructed, matrix_lower) < 1e-10, "solve_rtri");
		check(max_abs_error(xxt_result, matrix) < 1e-10, "xxt");
		check(max_abs_error(syr_result, vector_outer) < 1e-10, "syr");
		check(logdet_error < 1e-10, "logdet");
		check(quad_error < 1e-10, "quad");
		std::cout << "7 dense kernel: PASS\n";
		return 0;
	} catch (const char* error) {
		std::cerr << "FAIL: " << error << "\n";
		return 1;
	}
}
