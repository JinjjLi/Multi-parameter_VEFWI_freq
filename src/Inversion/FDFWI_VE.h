#pragma once

void FDFWI_VE(std::vector<Eigen::SparseMatrix<std::complex<float>>>& D, Eigen::RowVectorXf& freq, int step, Eigen::MatrixXcf& fwave, int nz, int nx, \
                     int dz, Eigen::MatrixXf& model, Eigen::MatrixXf& ssmodel0, float omega0, Eigen::SparseMatrix<float>& R, \
                     Eigen::SparseMatrix<float>& S, \
                     int optype, int numits, \
                     int PML_thick, float tol, int maxits, std::tuple<float *, float, float, float>& scale,\
                     float reg_fac, float stabregfac, \
                     Eigen::SparseMatrix<float>& P, Eigen::SparseMatrix<float>& P_big);

