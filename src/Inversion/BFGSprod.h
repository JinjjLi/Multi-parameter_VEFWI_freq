#pragma once

Eigen::MatrixXf BFGSprod(Eigen::MatrixXf& v, Eigen::MatrixXf& gstore, Eigen::MatrixXf& mstore, Eigen::SparseMatrix<float>& H0);

