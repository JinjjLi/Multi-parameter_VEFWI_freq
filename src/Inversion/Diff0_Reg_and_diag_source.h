#pragma once

bool Diff0_Reg_and_diag_source(Eigen::MatrixXf& model, float regfac, float stabregfac, \
                                                     int nz, int nx, \
                                                     Eigen::SparseMatrix<float> rH_big, \
                                                     Eigen::SparseMatrix<float>& P, \
                                                     float reg_scale);


