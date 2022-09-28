#include "../General/includefile.h"
#include "PML_pad.h"

void PML_pad(Eigen::MatrixXf &V, int PML_thick){
    Eigen::MatrixXf VPML = Eigen::MatrixXf(V.rows() + 2 * PML_thick, V.cols() + 2 * PML_thick);
    VPML.setZero();
    VPML.block(PML_thick, PML_thick, V.rows(), V.cols()) = V;
    int nx = V.cols(); int nz = V.rows();
    for (int i = 0; i < PML_thick; i++){
        VPML.block(PML_thick, i, nz, 1) = VPML.block(PML_thick, PML_thick, nz, 1);
        VPML.block(PML_thick, i + nx + PML_thick, nz, 1) = VPML.block(PML_thick, nx + PML_thick - 1, nz, 1);
    }
    for (int i = 0; i < PML_thick; i++){
        VPML.block(i, 0, 1, nx + 2 * PML_thick) = VPML.block(PML_thick, 0, 1, nx + 2 * PML_thick);
        VPML.block(nz + PML_thick + i, 0, 1, nx + 2 * PML_thick) = VPML.block(nz + PML_thick - 1, 0, 1, nx + 2 * PML_thick);
    }
    V = VPML;
    VPML.resize(0, 0);
}


