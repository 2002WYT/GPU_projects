#include "backend.hpp"
#include "cpu_verify.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

namespace {
void check(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
std::string quote(const std::string& s) {
    std::ostringstream o; o << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') o << '\\' << c;
        else if(c < 32) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else o << c;
    }
    o << '"'; return o.str();
}
uint64_t number(const std::string& s) {
    size_t n=0; if(s.empty() || s[0]=='-') throw std::runtime_error("invalid nonnegative integer");
    auto v=std::stoull(s,&n); if(n!=s.size()) throw std::runtime_error("invalid integer: "+s); return v;
}
}
int main(int argc, char** argv) {
    try {
        std::string ap,bp,b_mode="transpose"; uint64_t seed=20260910; int warmup=5,repeat=20;
        for(int i=1;i<argc;++i) {
            std::string key=argv[i], value; auto eq=key.find('=');
            if(key=="--help") { std::cout << "--a MATRIX [--b MATRIX] [--b-mode transpose|self|random] [--seed 20260910] [--warmup 5] [--repeat 20]\ntranspose uses A^T for real input and conjugate transpose A^H for complex input\n"; return 0; }
            if(eq!=std::string::npos) { value=key.substr(eq+1); key.resize(eq); }
            else { if(++i>=argc) throw std::runtime_error("missing argument for "+key); value=argv[i]; }
            if(key=="--a") ap=value; else if(key=="--b") bp=value; else if(key=="--b-mode") b_mode=value;
            else if(key=="--seed") seed=number(value);
            else if(key=="--warmup" || key=="--repeat") {
                auto n=number(value); if(n>INT32_MAX) throw std::runtime_error("iteration count too large");
                (key=="--warmup" ? warmup : repeat)=int(n);
            } else throw std::runtime_error("unknown argument: "+key);
        }
        if(ap.empty() || repeat<1) throw std::runtime_error("--a is required and --repeat must be positive");
        if(b_mode!="transpose" && b_mode!="self" && b_mode!="random") throw std::runtime_error("--b-mode must be transpose, self, or random");
        const auto a=spgemm::read_mtx(ap);
        const auto b=bp.empty()
            ?(b_mode=="self"?(a.rows==a.cols?a:throw std::runtime_error("A x A requires a square matrix"))
              :b_mode=="transpose"?spgemm::conjugate_transpose(a)
              :spgemm::generate_b(a,seed))
            :spgemm::read_mtx(bp);
        if(a.cols!=b.rows) throw std::runtime_error("incompatible matrix dimensions");
        check(cudaSetDevice(0));
        auto backend=spgemm::make_backend(a,b);
        auto run=[&] {
            check(cudaDeviceSynchronize()); auto t=std::chrono::steady_clock::now();
            backend->multiply(); check(cudaDeviceSynchronize());
            return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
        };
        double first=run(); for(int i=0;i<warmup;++i) run();
        std::vector<double> times; times.reserve(repeat); for(int i=0;i<repeat;++i) times.push_back(run());
        auto c=backend->download(); spgemm::validate_product(c);
        if(c.rows!=a.rows || c.cols!=b.cols) throw std::runtime_error("wrong output dimensions");
        auto sorted=times; std::sort(sorted.begin(),sorted.end());
        double median=(sorted[(repeat-1)/2]+sorted[repeat/2])/2;
        const char* value_type=(a.complex || b.complex)?"complex128":"float64";
        std::cout << std::setprecision(17) << "RESULT {\"backend\":" << quote(spgemm::backend_name())
          << ",\"value_type\":" << quote(value_type)
          << ",\"a_hash\":" << quote(spgemm::fingerprint(a)) << ",\"b_hash\":" << quote(spgemm::fingerprint(b))
          << ",\"rows\":" << c.rows << ",\"cols\":" << c.cols << ",\"nnz_a\":" << a.val.size()
          << ",\"nnz_b\":" << b.val.size() << ",\"nnz_c\":" << c.val.size() << ",\"b_mode\":" << quote(b_mode) << ",\"seed\":" << seed
          << ",\"warmup\":" << warmup << ",\"repeat\":" << repeat << ",\"first_ms\":" << first
          << ",\"median_ms\":" << median << ",\"mean_ms\":" << std::accumulate(times.begin(),times.end(),0.0)/repeat
          << ",\"min_ms\":" << sorted.front() << ",\"times_ms\":[";
        for(size_t i=0;i<times.size();++i) { if(i) std::cout << ','; std::cout << times[i]; }
        std::cout << "]}\n";
        // In-process verification against the CPU reference (no .csr dump). The
        // process exits 0 so the driver can parse the VERIFY line and classify
        // a failed verification as verify_failed rather than a crash.
        bool passed=false; double maxerr=0; int sampled=0;
        try { auto vr=spgemm::cpu_verify::verify(a,b,c); passed=vr.passed; maxerr=vr.max_error; sampled=vr.sample_rows; }
        catch(const std::exception&) { passed=false; }
        std::cout << "VERIFY {\"passed\":" << (passed?"true":"false")
                  << ",\"max_error\":" << maxerr << ",\"sample_rows\":" << sampled
                  << ",\"identity_trials\":3}\n";
        return 0;
    } catch(const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; }
}
