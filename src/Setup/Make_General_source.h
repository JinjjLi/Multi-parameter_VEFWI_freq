#pragma once

std::tuple<Eigen::SparseMatrix<float>, Eigen::SparseMatrix<float>> Make_General_source(Eigen::RowVectorXf& sx,\
                                              Eigen::RowVectorXf& sz,\
                                              Eigen::RowVectorXf& M11,\
                                              Eigen::RowVectorXf& M12,\
                                              Eigen::RowVectorXf& M22,\
                                              int nz, int nx, int PML_thick);

