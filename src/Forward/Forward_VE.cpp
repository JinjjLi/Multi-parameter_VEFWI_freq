#include "Forward_VE.h"

Forward_VE::Forward_VE(Setup_par_VE* this_par){
   this_par_alias = this_par;
   D = std::vector<Eigen::SparseMatrix<std::complex<float>>>(this_par_alias->freq.size());
}

void Forward_VE::Get_D(){
    std::cout << "Starting FD method." << std::endl;
    bool flag_D = Get_data_anelastic(D, this_par_alias->freq, this_par_alias->fwave, this_par_alias->model_true, \
                                     this_par_alias->R, this_par_alias->omega0, this_par_alias->S, this_par_alias->PML_thick, \
                                     this_par_alias->nz, this_par_alias->nx, this_par_alias->dz);
    std::cout << "Ending FD method." << std::endl;
}

Forward_VE::~Forward_VE(){
    D.clear(); D.shrink_to_fit();
    std::cout << "Data object destroyed." << std::endl;
}
