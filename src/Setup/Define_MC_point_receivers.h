#pragma once

Eigen::SparseMatrix<float> Define_MC_point_receivers(Eigen::RowVectorXf& rz, Eigen::RowVectorXf& rx, int nz, int nx, int PML_thick);

