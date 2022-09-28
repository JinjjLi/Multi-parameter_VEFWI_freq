#include "../General/includefile.h"
#include "Define_MC_point_receivers.h"

Eigen::SparseMatrix<float> Define_MC_point_receivers(Eigen::RowVectorXf& rz, Eigen::RowVectorXf& rx, int nz, int nx, int PML_thick)  
{
    int nzPML = nz + 2 * PML_thick; int nxPML = nx + 2 * PML_thick;
    int NPML = nzPML * nxPML;
    Eigen::RowVectorXf rind = Eigen::RowVectorXf(rx.size());
    rind = 2 * ((rx.array() + PML_thick) * nzPML + PML_thick + rz.array()) + 1;
    Eigen::SparseMatrix<float> Rx(rind.cols(), 2 * NPML);
    Eigen::SparseMatrix<float> Rz(rind.cols(), 2 * NPML);
    Rx.reserve(rind.cols()); Rz.reserve(rind.cols());
    for(int i = 0; i < rind.cols(); i++){
        Rx.coeffRef(i, rind(i)) = 1.0;
        Rz.coeffRef(i, rind(i) - 1) = 1.0;
    }
    Eigen::SparseMatrix<float> R(Rx.rows() + Rz.rows(), Rx.cols());
    R.reserve(Rx.nonZeros() + Rz.nonZeros());
    for(Eigen::Index c = 0; c < Rx.cols(); c++){
        R.startVec(c); /* Important: Must be called once for each column before inserting!*/ 
        for(Eigen::SparseMatrix<float>::InnerIterator itRx(Rx, c); itRx; ++itRx)
            R.insertBack(itRx.row(), c) = itRx.value();
        for(Eigen::SparseMatrix<float>::InnerIterator itRz(Rz, c); itRz; ++itRz)
            R.insertBack(itRz.row() + Rx.rows(), c) = itRz.value();
    }
    R.finalize();
    //std::cout << "Receiver matrix:" << std::endl;
    //for (int i = 0; i < R.outerSize(); i++){
    //    for (Eigen::SparseMatrix<float>::InnerIterator itR(R, i); itR; ++itR)
    //    {
    //        std::cout << "(" << itR.row() << ","; // row index
    //        std::cout << itR.col() << ")\t"; // col index (here it is equal to i)
    //        std::cout << " = " << itR.value() << std::endl;
    //    }
    //}
    rind.resize(0);
    Rx.resize(0, 0); Rx.data().squeeze(); Rz.resize(0, 0); Rz.data().squeeze();
    return R;
}

