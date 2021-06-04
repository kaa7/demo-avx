#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>
#define SIZE 16384
#define ITER 4096

using namespace std;

using namespace std::chrono;

void add_simple(int* a, int* b, int* c) {
	for (int i = 0; i < SIZE; ++i) {
		c[i] = a[i] + b[i];
	}
}

void add_avx(int* a, int* b, int* c) {
	__m256i res;

	for (int i = 0; i < SIZE; i += 8) {
		// load 8 ints from a[i] into an avx 256 bit register
		__m256i v_a = _mm256_load_si256((__m256i*)(a + i));

		// load 8 ints from b[i] into an avx 256 bit register
		__m256i v_b = _mm256_load_si256((__m256i*)(b + i));

		// add the two vectors
		res = _mm256_add_epi16(v_a, v_b);

		// store the value back
		_mm256_store_si256((__m256i*)(c + i), res);
	}
}

int main() {
	int* a = new int[SIZE];
	int* b = new int[SIZE];
	int* c = new int[SIZE];


	for (int i = 0; i < SIZE; ++i) {
		a[i] = i;
		b[i] = rand() % 100;
	}

	auto start = high_resolution_clock::now();

	for (int i = 0; i < ITER; ++i) {
		add_simple(a, b, c);
	}
	
	auto stop = high_resolution_clock::now();

	auto duration = duration_cast<microseconds>(stop - start);

	cout << "Simple execution time: " << duration.count() << " us\n";

	start = high_resolution_clock::now();

	for (int i = 0; i < ITER; ++i) {
		add_avx(a, b, c);
	}

	stop = high_resolution_clock::now();

	duration = duration_cast<microseconds>(stop - start);

	cout << "AVX execution time: " << duration.count() << " us\n";

	delete[] a;
	delete[] b;
	delete[] c;

	return 0;
}
