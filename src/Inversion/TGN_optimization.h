#pragma once

void TGN_optimization(Eigen::MatrixXf& model, Eigen::MatrixXf& ssmodel0, Eigen::MatrixXf& g, Eigen::RowVectorXf& frequency, \
            std::vector<Eigen::SparseMatrix<std::complex<float>>>& D, \
            float omega0, int nz, int nx, int dz, int PML_thick, Eigen::SparseMatrix<float>& R, \
            Eigen::SparseMatrix<float>& S, \
            Eigen::MatrixXcf& fwave, \
            std::tuple<float *, float, float, float>& scale, \
            Eigen::SparseMatrix<float>& P, Eigen::SparseMatrix<float>& P_big, float tol, int maxits, \
            float reg_fac, float stabregfac);

