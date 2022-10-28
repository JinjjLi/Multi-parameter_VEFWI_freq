#pragma once

float Zoom_Nocedal_Count_new(Eigen::MatrixXf &x0, Eigen::MatrixXf& descentd, \
                             float c1, float c2, float phi0, float g0, \
                             float alphalo, float alphahi, float philo, float phihi, \
                             float glo, float ghi, Eigen::RowVectorXf& freq, \
                             Eigen::MatrixXcf& fwave, float omega0, Eigen::MatrixXf& ssmodel0, \
                             std::vector<Eigen::SparseMatrix<std::complex<float>>>& D, Eigen::SparseMatrix<float>& R, \
                             Eigen::SparseMatrix<float>& S, \
                             int nz, int nx, int dz, int PML_thick, \
                             std::tuple<float *, float, float, float>& scale, Eigen::SparseMatrix<float>& P);                                    


