
// #include <bits/stdc++.h>

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>

using namespace std;

static inline double calculate(long long iterations, long long param1, long long param2) {
    double result = 1.0;
    for (long long i = 1; i <= iterations; ++i) {
        long long j1 = i * param1 - param2;
        result -= 1.0 / static_cast<double>(j1);
        long long j2 = i * param1 + param2;
        result += 1.0 / static_cast<double>(j2);
    }
    return result;
}

int main() {
    constexpr long long ITERATIONS = 200000000;
    constexpr long long PARAM1 = 4;
    constexpr long long PARAM2 = 1;

    auto start = chrono::high_resolution_clock::now();
    double result = calculate(ITERATIONS, PARAM1, PARAM2) * 4.0;
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed = end - start;
    printf("Result: %.12f\n", result);
    printf("Execution Time: %.6f seconds\n", elapsed.count());
    return 0;
}
