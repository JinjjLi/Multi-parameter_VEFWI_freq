#pragma once

bool par_FDFD_anelastic(Eigen::MatrixXf& model, Eigen::RowVectorXf& frequency, \
                        float omega0, Eigen::SparseMatrix<float>& S, Eigen::MatrixXcf& fwave, \
                        std::vector<Eigen::SparseMatrix<std::complex<float>>>& U, \
                        int PML_thick, int nz, int nx, int dz);

