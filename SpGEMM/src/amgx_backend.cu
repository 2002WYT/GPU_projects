#include "backend.hpp"
#include <amgx_c.h>
#include <csr_multiply.h>
#include <matrix.h>
#include <cuda_runtime.h>

namespace spgemm {
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
using Config=amgx::TemplateConfig<AMGX_device,AMGX_vecDouble,AMGX_matDouble,AMGX_indInt>;
using Matrix=amgx::Matrix<Config>;
using Multiply=amgx::CSR_Multiply<Config>;
class Amgx final:public Backend {
    std::unique_ptr<Matrix> a,b,c;void* workspace=nullptr;bool initialized=false;
    static void upload(Matrix& d,const HostCSR& h){
        d.set_initialized(0);d.addProps(amgx::CSR);d.resize(h.rows,h.cols,int(h.val.size()),1,1,1);
        check(cudaMemcpy(d.row_offsets.raw(),h.row.data(),h.row.size()*sizeof(int),cudaMemcpyHostToDevice));
        if(!h.val.empty()){
            check(cudaMemcpy(d.col_indices.raw(),h.col.data(),h.col.size()*sizeof(int),cudaMemcpyHostToDevice));
            check(cudaMemcpy(d.values.raw(),h.val.data(),h.val.size()*sizeof(double),cudaMemcpyHostToDevice));
        }
        d.set_initialized(1);
    }
public:
    Amgx(const HostCSR& aa,const HostCSR& bb){
        if(aa.cols!=bb.rows)throw std::runtime_error("A.cols must equal B.rows");
        if(AMGX_initialize()!=AMGX_RC_OK)throw std::runtime_error("AMGX_initialize failed");initialized=true;
        a=std::make_unique<Matrix>();b=std::make_unique<Matrix>();c=std::make_unique<Matrix>();
        upload(*a,aa);upload(*b,bb);c->addProps(amgx::CSR);
    }
    ~Amgx() override {
        if(workspace)Multiply::csr_workspace_delete(workspace);
        c.reset();b.reset();a.reset();if(initialized)AMGX_finalize();
    }
    void multiply() override {
        // First-call workspace construction is timed; subsequent calls reuse
        // capacity only. AMGX reruns its full symbolic and numerical phases.
        if(!workspace)workspace=Multiply::csr_workspace_create();
        Multiply::csr_multiply(*a,*b,*c,workspace);
    }
    HostCSR download() override {
        HostCSR h;h.rows=c->get_num_rows();h.cols=c->get_num_cols();int nz=c->get_num_nz();
        h.row.resize(size_t(h.rows)+1);h.col.resize(nz);h.val.resize(nz);
        check(cudaMemcpy(h.row.data(),c->row_offsets.raw(),h.row.size()*sizeof(int),cudaMemcpyDeviceToHost));
        if(nz){check(cudaMemcpy(h.col.data(),c->col_indices.raw(),size_t(nz)*sizeof(int),cudaMemcpyDeviceToHost));check(cudaMemcpy(h.val.data(),c->values.raw(),size_t(nz)*sizeof(double),cudaMemcpyDeviceToHost));}
        return h;
    }
};
}
std::unique_ptr<Backend> make_backend(const HostCSR& a,const HostCSR& b){return std::make_unique<Amgx>(a,b);}
const char* backend_name(){return "amgx";}
}
