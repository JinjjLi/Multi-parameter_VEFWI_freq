#pragma once

std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf> MakePML(int nz, int nx, int PML_thick);

