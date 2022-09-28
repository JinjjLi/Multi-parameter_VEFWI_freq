#pragma once

std::tuple<Eigen::SparseMatrix<float>, Eigen::SparseMatrix<float>> Make_P_sparse_alt(int nz, \
                                                                                     int nx, \
                                                                                     int PML_thick, \
                                                                                     int grid_int, \
                                                                                     int var_smooth);

