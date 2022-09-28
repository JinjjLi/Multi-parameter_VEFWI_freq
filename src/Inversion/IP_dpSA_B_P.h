#pragma once

Eigen::MatrixXf IP_dpSA_B_P(Eigen::SparseMatrix<std::complex<float>>& A, Eigen::SparseMatrix<std::complex<float>>& B, \
                            std::vector<Eigen::SparseMatrix<std::complex<float>>>& MADI, \
                           int nz, int nx, int PML_thick);

