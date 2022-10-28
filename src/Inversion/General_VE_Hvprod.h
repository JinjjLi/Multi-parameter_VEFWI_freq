#pragma once

Eigen::MatrixXf General_VE_Hvprod(Eigen::MatrixXf &model, Eigen::MatrixXf& ssmodel0, Eigen::MatrixXf& v, Eigen::RowVectorXf& frequency, \
                                 float omega0, int nz, int nx, int dz, int PML_thick, Eigen::SparseMatrix<float>& R, \
                                 Eigen::SparseMatrix<float>& S, \
                                 Eigen::MatrixXcf& fwave, float reg_fac, float stabregfac,\
                                 std::tuple<float *, float, float, float>& scale, \
                                 Eigen::SparseMatrix<float>& P, Eigen::SparseMatrix<float>& P_big);    
