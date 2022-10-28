#pragma once
#include "../General/includefile.h"
#include "Define_MC_point_receivers.h"

std::tuple<Eigen::SparseMatrix<float>, Eigen::SparseMatrix<float>> Define_Acquisition_Explosive(Eigen::RowVectorXf& sz,\
                                        Eigen::RowVectorXf& sx,\
                                        Eigen::RowVectorXf& rz,\
                                        Eigen::RowVectorXf& rx,\
                                        int nx, int nz, int PML_thick);

