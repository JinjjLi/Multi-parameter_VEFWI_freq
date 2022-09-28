#pragma once

Eigen::MatrixXf IP_dpSA_B(Eigen::SparseMatrix<std::complex<float>>& A, Eigen::SparseMatrix<std::complex<float>>& B, std::vector<Eigen::MatrixXcf>& MADI, \
                           int nz, int nx, int PML_thick);

