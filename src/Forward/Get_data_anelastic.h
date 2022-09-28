#pragma once

bool Get_data_anelastic(std::vector<Eigen::SparseMatrix<std::complex<float>>>& D, Eigen::RowVectorXf& freq, Eigen::MatrixXcf& fwave, Eigen::MatrixXf& MODEL, Eigen::SparseMatrix<float>& R, \
                                                 float omega0, Eigen::SparseMatrix<float>& S, int PML_thick, int nz, int nx, int dz);

