#pragma once

Eigen::MatrixXf LBFGS_solve_linear(Eigen::MatrixXf &model, Eigen::MatrixXf& ssmodel0, \
                                   Eigen::MatrixXf& b, Eigen::RowVectorXf& frequency, \
            float omega0, int nz, int nx, int dz, int PML_thick, Eigen::SparseMatrix<float>& R, \
            Eigen::SparseMatrix<float>& S, \
            Eigen::MatrixXcf& fwave, float reg_fac, float stabregfac, \
            std::tuple<float *, float, float, float>& scale, \
            Eigen::SparseMatrix<float>& P, Eigen::SparseMatrix<float>& P_big, float tol, int maxits);

