#include "../General/includefile.h"
#include "MakePML.h"

std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf> MakePML(int nz, int nx, int PML_thick){
    int Index = 4; float D0 = 4000.0;
    int nzPML = nz + 2 * PML_thick; int nxPML = nx + 2 * PML_thick;
    Eigen::RowVectorXf PMLz0 = Eigen::RowVectorXf(nzPML); PMLz0.setZero();
    Eigen::RowVectorXf PMLz1 = Eigen::RowVectorXf(nzPML); PMLz1.setZero();
    Eigen::RowVectorXf PMLx0 = Eigen::RowVectorXf(nxPML); PMLx0.setZero();
    Eigen::RowVectorXf PMLx1 = Eigen::RowVectorXf(nxPML); PMLx1.setZero();

//Eigen::initParallel();
//int nthreads = Eigen::nbThreads( );
//#pragma omp parallel
//#pragma omp for collapse(2)

    for (int ii = 0; ii < nxPML; ii++){
        if(ii < PML_thick){
            PMLx0(ii) = D0 * std::pow(((std::real(PML_thick - ii - 1) + 1.0) / std::real(PML_thick)), Index); 
            PMLx1(ii) = D0 * std::pow(((std::real(PML_thick - ii - 1) + 0.5) / std::real(PML_thick)), Index); 
        }
        else if (ii >= nxPML - PML_thick - 1){
            PMLx0(ii) = D0 * std::pow((std::real(PML_thick - nxPML + ii + 1) / std::real(PML_thick)), Index);
            PMLx1(ii) = D0 * std::pow((std::real(PML_thick - nxPML + ii + 1) + 0.5) / std::real(PML_thick), Index);
        }
    }
//#pragma omp parallel
//#pragma omp for collapse(2)
    for (int ii = 0; ii < nzPML; ii++){
        if(ii < PML_thick){
            PMLz0(ii) = D0 * std::pow(((std::real(PML_thick - ii - 1) + 1.0) / std::real(PML_thick)), Index); 
            PMLz1(ii) = D0 * std::pow(((std::real(PML_thick - ii - 1) + 0.5) / std::real(PML_thick)), Index); 
        }
        else if (ii >= nzPML - PML_thick - 1){
            PMLz0(ii) = D0 * std::pow((std::real(PML_thick - nzPML + ii + 1) / std::real(PML_thick)), Index);
            PMLz1(ii) = D0 * std::pow((std::real(PML_thick - nzPML + ii + 1) + 0.5) / std::real(PML_thick), Index);
        }
    }
    return std::make_tuple(PMLx0, PMLx1, PMLz0, PMLz1);
}

