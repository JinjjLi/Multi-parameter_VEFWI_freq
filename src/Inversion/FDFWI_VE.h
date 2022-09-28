#pragma once

void FDFWI_VE(std::vector<Eigen::SparseMatrix<std::complex<float>>>& D, Eigen::RowVectorXf& freq, int step, Eigen::MatrixXcf& fwave, int nz, int nx, \
                     int dz, Eigen::MatrixXf& model, Eigen::MatrixXf& ssmodel0, float omega0, Eigen::SparseMatrix<float>& R, \
                     std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf>& sz, std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf>& sx, \
                     std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf>& MT_sur, int optype, int numits, \
                     int PML_thick, float tol, int maxits, std::tuple<float *, float, float, float>& scale,\
                     std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf>& ind, float reg_fac, float stabregfac, \
                     Eigen::SparseMatrix<float>& P, Eigen::SparseMatrix<float>& P_big);

