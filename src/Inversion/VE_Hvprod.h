#pragma once

Eigen::MatrixXf VE_Hvprod(Eigen::MatrixXf &model, Eigen::MatrixXf& ssmodel0, Eigen::MatrixXf& v, Eigen::RowVectorXf& frequency, \
                                 float omega0, int nz, int nx, int dz, int PML_thick, Eigen::SparseMatrix<float>& R, \
                                 std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf>& sz, \
                                 std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf>& sx, \
                                 std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf>& MT_sur, \
                                 Eigen::MatrixXcf& fwave, float reg_fac, float stabregfac,\
                                 std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf>& ind, \
                                 std::tuple<float *, float, float, float>& scale, \
                                 Eigen::SparseMatrix<float>& P, Eigen::SparseMatrix<float>& P_big);    
