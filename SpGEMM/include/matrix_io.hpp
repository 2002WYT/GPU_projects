#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace spgemm {
struct HostCSR { int rows = 0, cols = 0; std::vector<int> row, col; std::vector<double> val; };
namespace detail {
inline void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
inline bool data_line(std::istream& in, std::string& line) {
    while (std::getline(in,line)) {
        const auto p = line.find_first_not_of(" \t\r\n");
        if (p != std::string::npos && line[p] != '%') return true;
    }
    return false;
}
inline void end_line(std::istringstream& line) { line >> std::ws; require(line.eof(), "unexpected extra Matrix Market field"); }
inline void validate(const HostCSR& a) {
    require(a.rows >= 0 && a.cols >= 0, "negative CSR dimensions");
    require(a.row.size() == static_cast<size_t>(a.rows)+1 && !a.row.empty() && a.row[0] == 0, "invalid CSR row offsets");
    require(a.col.size() == a.val.size() && a.val.size() <= static_cast<size_t>(INT32_MAX), "CSR nnz exceeds int32 or value count mismatch");
    require(a.row.back() == static_cast<int>(a.val.size()), "CSR terminal offset mismatch");
    for (int r=0;r<a.rows;++r) {
        require(a.row[r]>=0 && a.row[r]<=a.row[r+1] && a.row[r+1]<=a.row.back(), "invalid CSR row interval");
        for(int p=a.row[r];p<a.row[r+1];++p) {
            require(a.col[p]>=0 && a.col[p]<a.cols && (p==a.row[r] || a.col[p-1]<a.col[p]), "CSR columns must be sorted and unique");
            require(std::isfinite(a.val[p]), "nonfinite CSR value");
        }
    }
}
struct Entry { int r,c; double v; };
inline HostCSR canonical(int rows,int cols,std::vector<Entry>& entries) {
    std::stable_sort(entries.begin(),entries.end(),[](const Entry& x,const Entry& y) { return x.r<y.r || (x.r==y.r && x.c<y.c); });
    HostCSR a; a.rows=rows; a.cols=cols; a.row.assign(static_cast<size_t>(rows)+1,0);
    a.col.reserve(entries.size()); a.val.reserve(entries.size());
    int previous_row=-1,previous_col=-1;
    for(const auto& e:entries) {
        if(e.r==previous_row && e.c==previous_col) {
            a.val.back()+=e.v; require(std::isfinite(a.val.back()),"duplicate sum is nonfinite");
        } else {
            require(a.val.size()<static_cast<size_t>(INT32_MAX),"canonical nnz exceeds int32");
            a.col.push_back(e.c); a.val.push_back(e.v); ++a.row[static_cast<size_t>(e.r)+1];
            previous_row=e.r; previous_col=e.c;
        }
    }
    for(size_t r=1;r<a.row.size();++r) a.row[r]+=a.row[r-1];
    return a;
}
struct SplitMix64 {
    uint64_t state;
    uint64_t next() {
        uint64_t z=(state+=UINT64_C(0x9e3779b97f4a7c15));
        z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
        z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
        return z^(z>>31);
    }
    uint64_t bounded(uint64_t n) {
        const uint64_t threshold=(uint64_t(0)-n)%n;
        for(;;) { const uint64_t x=next(); if(x>=threshold) return x%n; }
    }
};
} // namespace detail

inline HostCSR read_mtx(std::string path) {
    std::ifstream in(path); in.imbue(std::locale::classic());
    detail::require(bool(in),"cannot open matrix: "+path);
    std::string line, magic,object,format,field,symmetry;
    detail::require(bool(std::getline(in,line)),"empty Matrix Market file");
    std::istringstream header(line); header >> magic >> object >> format >> field >> symmetry;
    detail::require(magic=="%%MatrixMarket" && object=="matrix" && format=="coordinate", "only Matrix Market coordinate matrices are supported");
    detail::require(field!="complex","complex Matrix Market values are unsupported: benchmark uses real FP64");
    detail::require(field=="real" || field=="integer" || field=="pattern","unsupported Matrix Market field");
    detail::require(symmetry=="general" || symmetry=="symmetric" || symmetry=="skew-symmetric","unsupported Matrix Market symmetry");
    detail::end_line(header);
    detail::require(detail::data_line(in,line),"missing matrix dimensions");
    std::istringstream dimensions(line); long long rows=-1,cols=-1,count=-1;
    detail::require(bool(dimensions>>rows>>cols>>count),"invalid matrix dimensions"); detail::end_line(dimensions);
    detail::require(rows>=0 && cols>=0 && count>=0 && rows<=INT32_MAX && cols<=INT32_MAX && count<=INT32_MAX,"matrix dimensions or nnz outside int32 range");
    detail::require(symmetry=="general" || rows==cols,"symmetric matrix must be square");
    detail::require((rows>0 && cols>0) || count==0,"zero dimension matrix has nonzero entries");
    std::vector<detail::Entry> entries; entries.reserve(static_cast<size_t>(count));
    for(long long i=0;i<count;++i) {
        detail::require(detail::data_line(in,line),"fewer entries than declared");
        std::istringstream record(line); record.imbue(std::locale::classic());
        long long r=0,c=0; double value=1;
        detail::require(bool(record>>r>>c),"invalid matrix coordinate");
        if(field=="integer") { long long integer=0; detail::require(bool(record>>integer),"invalid integer value"); value=static_cast<double>(integer); }
        else if(field=="real") detail::require(bool(record>>value),"invalid real value");
        detail::end_line(record);
        detail::require(r>=1 && r<=rows && c>=1 && c<=cols,"matrix coordinate out of bounds");
        detail::require(std::isfinite(value),"matrix value is nonfinite");
        detail::require(symmetry!="skew-symmetric" || r!=c || value==0,"skew-symmetric diagonal must be zero");
        const size_t added=(symmetry!="general" && r!=c)?2:1;
        detail::require(entries.size()<=static_cast<size_t>(INT32_MAX)-added,"expanded nnz exceeds int32");
        entries.push_back({static_cast<int>(r-1),static_cast<int>(c-1),value});
        if(added==2) entries.push_back({static_cast<int>(c-1),static_cast<int>(r-1),symmetry=="skew-symmetric"?-value:value});
    }
    detail::require(!detail::data_line(in,line),"more entries than declared");
    return detail::canonical(static_cast<int>(rows),static_cast<int>(cols),entries);
}

inline HostCSR generate_b(const HostCSR& a,uint64_t seed=20260910) {
    detail::validate(a);
    HostCSR b; b.rows=a.cols; b.cols=a.rows; b.row.assign(static_cast<size_t>(b.rows)+1,0);
    const uint64_t population=uint64_t(b.rows)*uint64_t(b.cols), count=a.val.size();
    detail::require(count<=population,"requested B nnz exceeds its coordinate capacity");
    detail::SplitMix64 rng{seed};
    std::vector<uint64_t> positions; positions.reserve(static_cast<size_t>(count));
    {
        // Floyd's sample is uniform without replacement, including near-full matrices.
        std::unordered_set<uint64_t> selected; selected.reserve(static_cast<size_t>(count));
        for(uint64_t j=population-count;j<population;++j) {
            const uint64_t candidate=rng.bounded(j+1);
            const uint64_t coordinate=selected.insert(candidate).second?candidate:j;
            if(coordinate!=candidate) selected.insert(coordinate);
            positions.push_back(coordinate);
        }
    }
    std::sort(positions.begin(),positions.end()); b.col.reserve(static_cast<size_t>(count)); b.val.reserve(static_cast<size_t>(count));
    for(uint64_t p:positions) {
        ++b.row[static_cast<size_t>(p/uint64_t(b.cols))+1]; b.col.push_back(static_cast<int>(p%uint64_t(b.cols)));
        const int raw=static_cast<int>(rng.bounded(2048));
        b.val.push_back(double(raw<1024?raw-1024:raw-1023)/1024.0);
    }
    for(size_t r=1;r<b.row.size();++r) b.row[r]+=b.row[r-1];
    return b;
}

// Transpose a CSR matrix via a counting-sort over A's column buckets. Because
// read_mtx/canonical yields rows with sorted, unique columns, iterating A's rows
// in order scatters into each transposed row with ascending, unique columns, so
// the result is canonical without an extra dedup/sort pass.
inline HostCSR transpose(const HostCSR& a) {
    detail::validate(a);
    HostCSR t; t.rows=a.cols; t.cols=a.rows;
    t.row.assign(static_cast<size_t>(t.rows)+1,0);
    for(int x:a.col) ++t.row[static_cast<size_t>(x)+1];
    for(size_t r=1;r<t.row.size();++r) t.row[r]+=t.row[r-1];
    t.col.resize(a.val.size()); t.val.resize(a.val.size());
    std::vector<int> cursor(t.row.begin(),t.row.end());
    for(int r=0;r<a.rows;++r)
        for(int p=a.row[r];p<a.row[r+1];++p) {
            const int pos=cursor[static_cast<size_t>(a.col[p])]++;
            t.col[pos]=r; t.val[pos]=a.val[p];
        }
    return t;
}

inline std::string fingerprint(const HostCSR& a) {
    detail::validate(a); static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559,"IEEE754 binary64 required");
    uint64_t hash=UINT64_C(14695981039346656037);
    const auto add=[&hash](uint64_t x,int bytes) { for(int i=0;i<bytes;++i) { hash^=x&255; hash*=UINT64_C(1099511628211); x>>=8; } };
    add(static_cast<uint32_t>(a.rows),4); add(static_cast<uint32_t>(a.cols),4); add(a.val.size(),8);
    for(int x:a.row) add(static_cast<uint32_t>(x),4);
    for(int x:a.col) add(static_cast<uint32_t>(x),4);
    for(double x:a.val) { uint64_t bits=0; std::memcpy(&bits,&x,8); add(bits,8); }
    std::ostringstream out; out.imbue(std::locale::classic()); out<<std::hex<<std::setw(16)<<std::setfill('0')<<hash; return out.str();
}

inline HostCSR cpu_product(const HostCSR& a,const HostCSR& b) {
    detail::validate(a); detail::validate(b); detail::require(a.cols==b.rows,"incompatible product dimensions");
    HostCSR c; c.rows=a.rows; c.cols=b.cols; c.row.push_back(0);
    std::unordered_map<int,double> sums; std::vector<int> columns;
    for(int r=0;r<a.rows;++r) {
        sums.clear(); columns.clear();
        for(int p=a.row[r];p<a.row[r+1];++p) for(int q=b.row[a.col[p]];q<b.row[a.col[p]+1];++q) sums[b.col[q]]+=a.val[p]*b.val[q];
        detail::require(sums.size()<=static_cast<size_t>(INT32_MAX)-c.val.size(),"product nnz exceeds int32");
        for(const auto& pair:sums) columns.push_back(pair.first);
        std::sort(columns.begin(),columns.end());
        for(int col:columns) { const double v=sums.at(col); detail::require(std::isfinite(v),"CPU product contains nonfinite value"); c.col.push_back(col); c.val.push_back(v); }
        c.row.push_back(static_cast<int>(c.val.size()));
    }
    return c;
}
} // namespace spgemm
