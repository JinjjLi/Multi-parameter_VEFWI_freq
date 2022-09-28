#pragma once

float Linesearch_Nocedal_Full_new(Eigen::MatrixXf &x0, Eigen::MatrixXf& descentd, \
                                  float c1, float c2, float alpha0, Eigen::RowVectorXf& freq, \
                                  Eigen::MatrixXcf& fwave, float omega0, \
                                  Eigen::MatrixXf& ssmodel0, std::vector<Eigen::SparseMatrix<std::complex<float>>>& D, Eigen::SparseMatrix<float>& R, \
                                  std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf>& sz, std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf>& sx, \
                                  std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf>& MT_sur, \
                                  int nz, int nx, int dz, int PML_thick, \
                                  std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf>& ind, \
                                  std::tuple<float *, float, float, float>& scale, Eigen::SparseMatrix<float>& P);

