#pragma once

bool Make_Helm_Anelastic_and_Derivative_efficiency(std::vector<Eigen::MatrixXcf>& MADI, Eigen::MatrixXf model, \
                                                               int nx, int nz, float omega, \
                                                               float omega0, int dz, \
                                                               int PML_thick, float * scale);

