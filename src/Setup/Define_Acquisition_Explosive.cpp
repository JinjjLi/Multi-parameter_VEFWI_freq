#include "Define_Acquisition_Explosive.h"

std::tuple<Eigen::SparseMatrix<float>, Eigen::SparseMatrix<float>> Define_Acquisition_Explosive(Eigen::RowVectorXf& sz,\
                                        Eigen::RowVectorXf& sx,\
                                        Eigen::RowVectorXf& rz,\
                                        Eigen::RowVectorXf& rx,\
                                        int nx, int nz, int PML_thick){

    int nzPML = nz + 2 * PML_thick;
    int nxPML = nx + 2 * PML_thick;
    int NPML = nzPML * nxPML;

    int ns = sx.size();
    Eigen::SparseMatrix<float> S(2 * NPML, ns);
    S.setZero();
    Eigen::SparseMatrix<float> R(2 * NPML, ns);
    R.setZero();

    Eigen::RowVectorXf splus_ind(4);
    Eigen::RowVectorXf sminus_ind(4);
    splus_ind.setZero();
    sminus_ind.setZero();
    
    typedef Eigen::Triplet<float> T;
    std::vector<T> triplet;

    for (int n = 0; n < ns; n++){
        splus_ind << 2 * (((sx(n) + 1) + PML_thick) * (nzPML) + PML_thick + sz(n)), \
                     2 * (((sx(n) + 1) + PML_thick) * (nzPML) + PML_thick + sz(n) + 1), \
                     2 * (((sx(n)) + PML_thick) * (nzPML) + PML_thick + sz(n) + 1) + 1, \
                     2 * (((sx(n) + 1) + PML_thick) * (nzPML) + PML_thick + sz(n) + 1) + 1;

        sminus_ind << 2 * (((sx(n)) + PML_thick) * (nzPML) + PML_thick + sz(n)), \
                     2 * (((sx(n)) + PML_thick) * (nzPML) + PML_thick + sz(n) + 1), \
                     2 * (((sx(n)) + PML_thick) * (nzPML) + PML_thick + sz(n)) + 1, \
                     2 * (((sx(n) + 1) + PML_thick) * (nzPML) + PML_thick + sz(n)) + 1;
        
        for (int i = 0; i < 4; i++){
            triplet.push_back(T(splus_ind(i), n, 1));
            triplet.push_back(T(sminus_ind(i), n, -1));
        }
    }
    S.setFromTriplets(triplet.begin(), triplet.end());

    R = Define_MC_point_receivers(rz, rx, nz, nx, PML_thick);

    //std::cout << "Matrix:" << std::endl;
    //for (int i = 0; i < R.outerSize(); i++){
    //    for (Eigen::SparseMatrix<float>::InnerIterator it(R, i); it; ++it)
    //    {
    //        std::cout << "(" << it.row() << ","; // row index
    //        std::cout << it.col() << ")\t"; // col index (here it is equal to i)
    //        std::cout << " = " << it.value() << std::endl;
    //    }
    //}                                                                                                                                              

    splus_ind.resize(0);
    sminus_ind.resize(0);
    triplet.clear(); triplet.shrink_to_fit();

    return std::make_tuple(S, R);

}
