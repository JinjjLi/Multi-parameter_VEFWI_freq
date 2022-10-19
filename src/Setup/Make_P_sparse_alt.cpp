#include "../General/includefile.h"
#include "Make_P_sparse_alt.h"
#include "../General/intersect.h"

std::tuple<Eigen::SparseMatrix<float>, Eigen::SparseMatrix<float>> \
        Make_P_sparse_alt(int nz, int nx, int PML_thick, int grid_int, \
                          int var_smooth){
            
            int nzPML = nz + 2 * PML_thick; int nxPML = nx + 2 * PML_thick;
            int NN = nz * nx; int NPML = nzPML * nxPML;

            int z_cent_num = (nz - round(grid_int/2)) / grid_int;
            int x_cent_num = (nx - round(grid_int/2)) / grid_int;

            Eigen::RowVectorXf z_cent = Eigen::RowVectorXf::LinSpaced(z_cent_num, round(grid_int / 2), nz - 1);
            Eigen::RowVectorXf x_cent = Eigen::RowVectorXf::LinSpaced(x_cent_num, round(grid_int / 2), nx - 1);
            z_cent.cast<int>(); x_cent.cast<int>();


            int Plength = z_cent.size() * x_cent.size();
            Eigen::MatrixXf temp = Eigen::MatrixXf(6 * var_smooth + 1, 6 * var_smooth + 1);
            temp.setZero();
            int midp = 3 * var_smooth;
            for (int n = 0; n < 6 * var_smooth + 1; n++){
                for (int m = 0; m < 6 * var_smooth + 1; m++){
                    float dist = sqrt((n - midp) * (n - midp) + (m - midp) * (m - midp));
                    temp(n, m) = exp(-(dist/var_smooth) * (dist/var_smooth));
                }
            }
            float th = 0.005;
            std::vector<std::pair<int,int>> indices;
            visit_lambda(temp, [&indices,th](double v, int i, int j) {
                         if(v>th)
                         indices.push_back(std::make_pair(i, j));
                         });
            Eigen::RowVectorXf temp_small(indices.size());
            Eigen::RowVectorXf z_in(indices.size());
            Eigen::RowVectorXf x_in(indices.size());
            z_in.cast<int>(); x_in.cast<int>();
            for (int i = 0; i < indices.size(); i++){
                z_in[i] = indices[i].first; x_in[i] = indices[i].second;
                temp_small(i) = temp(indices[i].first, indices[i].second);
            }
            z_in = z_in.array() - midp; x_in = x_in.array() - midp;


            Eigen::MatrixXi z_inds = Eigen::MatrixXi(temp_small.size(), Plength);
            Eigen::MatrixXi x_inds = Eigen::MatrixXi(temp_small.size(), Plength);
            z_inds.setZero();
            x_inds.setZero();
            Eigen::MatrixXf vals(temp_small.size(), Plength);
            vals.setZero();

            for (int n = 0; n < z_cent.size(); n++){
                Eigen::RowVectorXf z_in_shift = z_in.array() + z_cent(n);
                z_in_shift.cast<int>();
                auto z_in_shift_lessnz = z_in_shift.array() < nz;
                //auto z_in_shift_eqgreat0 = z_in_shift_eq0.array() + z_in_shift_greater0.array();
                auto z_in_shift_eqgreat0 = z_in_shift.array() >= 0;   

                std::vector<int> z_in_inds;
                for (int i = 0; i < z_in_shift.size(); i++){                                                                         
                    if (z_in_shift_eqgreat0(i) && z_in_shift_lessnz(i))
                        z_in_inds.push_back(i);
                }
                for (int m = 0; m < x_cent.size(); m++){
                    int P_ind = n + m * z_cent.size();
                    Eigen::RowVectorXf x_in_shift = x_in.array() + x_cent(m);
                    x_in_shift.cast<int>();
                    auto x_in_shift_lessnx = x_in_shift.array() < nx;
                    //auto x_in_shift_eqgreat0 = x_in_shift_eq0.array() + x_in_shift_greater0.array();
                    auto x_in_shift_eqgreat0 = x_in_shift.array() >= 0;

                    std::vector<int> x_in_inds;
                    for (int i = 0; i < x_in_shift.size(); i++){
                        if (x_in_shift_eqgreat0(i) && x_in_shift_lessnx(i))
                            x_in_inds.push_back(i);
                    }
                    auto in_inds = intersect(z_in_inds, x_in_inds);
                    float tempsum = 0.0;
                    for (int i = 0; i < in_inds.size(); i++)
                        tempsum += temp_small(in_inds[i]);
                    for (int i = 0; i < in_inds.size(); i++){
                        z_inds(i, P_ind) = z_in_shift(in_inds[i]);
                        x_inds(i, P_ind) = x_in_shift(in_inds[i]);
                        vals(i, P_ind) = temp_small(in_inds[i]) / tempsum;
                    }
                    x_in_inds.clear(); x_in_inds.shrink_to_fit();
                 }
                z_in_inds.clear(); z_in_inds.shrink_to_fit();
            }


            Eigen::MatrixXi Loc_inds = z_inds.array() + x_inds.array() * nz;
            Eigen::MatrixXi P_inds = Eigen::MatrixXi::Constant(Loc_inds.rows(), Loc_inds.cols(), 1);
            Eigen::RowVectorXf Plen(Plength);
            Plen.cast<int>();
            Plen = Eigen::RowVectorXf::LinSpaced(Plength, 0, Plength - 1);
            for (int i = 0; i < Plen.size(); i++)
                P_inds.col(i).array() *= Plen(i);
            Eigen::SparseMatrix<float> P(NN, Plength);
            Eigen::SparseMatrix<float> P_big(NPML, Plength);
            //P.reserve((vals.array() > 0).count()); 

            typedef Eigen::Triplet<float> T;
            std::vector<T> triplet;
            //triplet.reserve(P.size() * 5);

            for (int i = 0; i < vals.rows(); i++){
                for (int j = 0; j < Plength; j++){
                    if (vals(i, j))
                        triplet.push_back(T(Loc_inds(i, j), P_inds(i, j), vals(i, j)));
                }   
            }

            P.setFromTriplets(triplet.begin(), triplet.end());                                                                       


            triplet.clear(); triplet.shrink_to_fit();
            Eigen::SparseMatrix<float> P_diag(P.rows() * 5, P.cols() * 5);
            //P_diag.reserve(P.nonZeros() * 5);
            typedef Eigen::Triplet<float> T;
            std::vector<T> triplets;
            //triplets.reserve(P.size() * 5);
            for (int i = 0; i < 5; i++){
                for (int m = 0; m < P.outerSize(); m++){
                    for (Eigen::SparseMatrix<float>::InnerIterator it(P, m); it; ++it){
                        if (it.value())
                            triplets.push_back(T(i * P.rows() + it.row(), i * P.cols() + it.col(), it.value()));
                    }
                }
            }
            P_diag.setFromTriplets(triplets.begin(), triplets.end());
            triplets.clear(); triplets.shrink_to_fit();


            P_big.reserve(P.nonZeros() * 2);
            for (int n = 0; n < Plength; n++){
                Eigen::MatrixXf big_frame = Eigen::MatrixXf(nzPML, nxPML);
                big_frame.setZero();
                Eigen::MatrixXf temp_P = P.col(n);
                temp_P.resize(nz, nx);
                big_frame.block(PML_thick, PML_thick, nz, nx) = temp_P;
                for (int i = 0; i < PML_thick; i++){
                    big_frame.block(i, PML_thick, 1, nx) = big_frame.block(PML_thick, PML_thick, 1, nx);
                    big_frame.block(i + nz + PML_thick, PML_thick, 1, nx) = big_frame.block(nz + PML_thick - 1, PML_thick, 1, nx);}
                for (int i = 0; i < PML_thick; i++){
                    big_frame.block(0, i, nzPML, 1) = big_frame.block(0, PML_thick, nzPML, 1);
                    big_frame.block(0, i + nx + PML_thick, nzPML, 1) = big_frame.block(0, nx + PML_thick - 1, nzPML, 1);  
                }
                big_frame.resize(1, big_frame.size());
                for(int i = 0; i < big_frame.size(); i++){                                                                           
                    if(big_frame(0, i))
                        P_big.coeffRef(i, n) = big_frame(0, i);
                }
                big_frame.resize(0, 0); temp_P.resize(0, 0);
            }
            Eigen::SparseMatrix<float> P_big_diag(P_big.rows() * 5, P_big.cols() * 5);
            //P_big_diag.reserve(P_big.nonZeros() * 5);
            std::vector<T> triplets2;
            //triplets2.reserve(P_big.size() * 5);
            for (int i = 0; i < 5; i++){
                for (int m = 0; m < P.outerSize(); m++){
                    for (Eigen::SparseMatrix<float>::InnerIterator it(P_big, m); it; ++it){
                        if (it.value())
                            triplets2.push_back(T(i * P_big.rows() + it.row(), i * P_big.cols() + it.col(), it.value()));
                    }
                }
            }
            P_big_diag.setFromTriplets(triplets2.begin(), triplets2.end());
            triplets2.clear(); triplets2.shrink_to_fit();
            z_cent.resize(0); x_cent.resize(0);
            temp.resize(0, 0); temp_small.resize(0);
            indices.clear(); indices.shrink_to_fit();
            z_in.resize(0); x_in.resize(0);
            z_inds.resize(0, 0); x_inds(0, 0); vals.resize(0, 0);
            Loc_inds.resize(0, 0); P_inds.resize(0, 0);
            Plen.resize(0);
            P.resize(0, 0); P.data().squeeze();
            P_big.resize(0, 0); P_big.data().squeeze();
            return std::make_tuple(P_diag, P_big_diag);
        }


