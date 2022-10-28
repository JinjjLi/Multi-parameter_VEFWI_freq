#include "../General/includefile.h"
#include "Define_MC_point_receivers.h"

Eigen::SparseMatrix<float> Define_MC_point_receivers(Eigen::RowVectorXf& rz, Eigen::RowVectorXf& rx, int nz, int nx, int PML_thick)  
{
    int nzPML = nz + 2 * PML_thick; int nxPML = nx + 2 * PML_thick;
    int NPML = nzPML * nxPML;
    Eigen::RowVectorXf rind(rx.size());
    rind = 2 * ((rx.array() + PML_thick) * nzPML + PML_thick + rz.array()) + 1;
    Eigen::SparseMatrix<float> R(2 * rind.cols(), 2 * NPML);

    typedef Eigen::Triplet<float> T;
    std::vector<T> triplet;

    for(int i = 0; i < rind.cols(); i++){
        triplet.push_back(T(i, rind(i), 1));
        triplet.push_back(T(i + rind.cols(), rind(i) - 1, 1));
    }
    R.setFromTriplets(triplet.begin(), triplet.end());
    
    rind.resize(0);
    triplet.clear(); triplet.shrink_to_fit();
    return R;
}

