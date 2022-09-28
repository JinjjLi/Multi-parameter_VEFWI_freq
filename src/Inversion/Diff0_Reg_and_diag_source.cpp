#include "../General/includefile.h"
#include "Diff0_Reg_and_diag_source.h"

bool Diff0_Reg_and_diag_source(Eigen::MatrixXf& model, float regfac, float stabregfac, \
                                                     int nz, int nx, \
                                                     Eigen::SparseMatrix<float> rH_big, \
                                                     Eigen::SparseMatrix<float>& P, \
                                                     float reg_scale){

    regfac *= reg_scale;
    float param_scale[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    int NN = nz * nx;
    Eigen::MatrixXf b1 = Eigen::MatrixXf(NN, 1);
    Eigen::MatrixXf b2 = Eigen::MatrixXf(NN, 1);
    b1.setZero(); b2.setZero();
    Eigen::MatrixXf b3 = Eigen::MatrixXf::Constant(NN, 1, 1.0);
    Eigen::MatrixXf b4 = Eigen::MatrixXf::Constant(NN, 1, -1.0);
    b1.block(0, 0, nz * (nx - 1), 1) = Eigen::MatrixXf::Constant(nz * (nx - 1), 1, -1.0);
    b2.block(nz, 0, nz * (nx - 1), 1) = Eigen::MatrixXf::Constant(nz * (nx - 1), 1, 1.0);
    for (int i = 0; i < nx; i++){
        b3(nz * i, 0) = 0.0;
        b4(nz - 1 + nz * i, 0) = 0.0;
    }
    Eigen::MatrixXf b1_term = Eigen::MatrixXf(5 * b1.size(), 1);
    Eigen::MatrixXf b2_term = Eigen::MatrixXf(5 * b2.size(), 1);
    Eigen::MatrixXf b3_term = Eigen::MatrixXf(5 * b3.size(), 1);
    Eigen::MatrixXf b4_term = Eigen::MatrixXf(5 * b4.size(), 1);
    b1_term.setZero(); b2_term.setZero(); b3_term.setZero();b4_term.setZero();

    for (int n = 0; n < 5; n++){
        b1_term.block(n * NN, 0, NN, 1) = b1.array() * param_scale[n];
        b2_term.block(n * NN, 0, NN, 1) = b2.array() * param_scale[n];
        b3_term.block(n * NN, 0, NN, 1) = b3.array() * param_scale[n];
        b4_term.block(n * NN, 0, NN, 1) = b4.array() * param_scale[n];
    }
    Eigen::MatrixXf B1 = Eigen::MatrixXf(5 * NN, 2);
    Eigen::MatrixXf B2 = Eigen::MatrixXf(5 * NN, 2);
    B1.setZero(); B2.setZero();
    B1.col(0) = b1_term;
    B1.col(1) = b2_term;
    B2.col(0) = b4_term;
    B2.col(1) = b3_term;
    
    Eigen::MatrixXi d1(1, 2); Eigen::MatrixXi d2(1, 2);
    d1 << -nz, 0; d2 << -1, 0;

    Eigen::SparseMatrix<float> derivteststorex(5 * NN, 5 * NN); 
    //derivteststorex.reserve(B1.size());
    spdiags_noncomplex<float>(derivteststorex, B1, d1, 5 * NN, 5 * NN);
    Eigen::SparseMatrix<float> derivteststorey(5 * NN, 5 * NN); 
    //derivteststorey.reserve(B2.size());
    spdiags_noncomplex<float>(derivteststorey, B2, d2, 5 * NN, 5 * NN);
    
    Eigen::SparseMatrix<float> rH = regfac * (derivteststorey.transpose() * derivteststorey + \
                                       derivteststorex.transpose() * derivteststorex);
    Eigen::SparseMatrix<float> temp = Eigen::SparseMatrix<float>(5 * NN, 5 * NN);
    spdiags_noncomplex<float>(temp, Eigen::MatrixXf::Constant(5 * NN, 1, 1.0 * regfac), \
                              Eigen::MatrixXi::Zero(1, 1), 5 * NN, 5 * NN);
    rH += temp;
    rH = P.transpose() * rH * P;
    
    //Eigen::SparseMatrix<float> rH_big(model.size(), model.size());
    //rH_big.reserve(rH.nonZeros() + model.size());
    rH_big.setZero();
    //for (int i = 0; i < P.cols(); i++){
    //    for (int j = 0; j < P.cols(); j++)
    //        rH_big.coeffRef(i, j) += rH.coeffRef(i, j);
    //}
    typedef Eigen::Triplet<float> T;
    std::vector<T> triplets;
    //triplets.reserve(rH.nonZeros() + model.size());
    for (int i = 0; i < rH.outerSize(); i++){
        for (Eigen::SparseMatrix<float>::InnerIterator it(rH, i); it; ++it){
            if (it.value())
                triplets.push_back(T(it.row(), it.col(), it.value()));
        }
    }
    rH_big.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::SparseMatrix<float> temp_prod(rH_big.rows(), rH_big.cols()); 
    spdiags_noncomplex<float>(temp_prod, Eigen::MatrixXf::Constant(model.size(), 1, 1.0), \
                                     Eigen::MatrixXi::Zero(1, 1), rH_big.rows(), rH_big.cols());
    rH_big += stabregfac * reg_scale * temp_prod;
    
    //for (int i = 0; i < rH_big.outerSize(); i++){
    //    for (Eigen::SparseMatrix<float>::InnerIterator it(rH_big, i); it; ++it){
    //        if (it.value() != 0){
    //            std::cout << "(" << it.row() << ",";
    //            std::cout << it.col() << ")";
    //            std::cout << "= " << it.value() << std::endl;
    //        }
    //    }
    //}

    b1.resize(0, 0); b2.resize(0, 0); b3.resize(0, 0); b4.resize(0, 0);
    b1_term.resize(0, 0); b2_term.resize(0, 0);
    b3_term.resize(0, 0); b4_term.resize(0, 0);
    B1.resize(0, 0); B2.resize(0, 0);
    d1.resize(0, 0); d2.resize(0, 0);
    derivteststorex.resize(0, 0); derivteststorex.data().squeeze();
    derivteststorey.resize(0, 0); derivteststorey.data().squeeze();
    temp.resize(0, 0); temp.data().squeeze();
    temp_prod.resize(0, 0); temp_prod.data().squeeze();
    rH.resize(0, 0); rH.data().squeeze();
    triplets.clear(); triplets.shrink_to_fit();

    //return rH_big;
    return 1;
}
  

