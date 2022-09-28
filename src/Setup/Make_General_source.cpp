#include "../General/includefile.h"
#include "Make_General_source.h"

std::tuple<Eigen::SparseMatrix<float>, Eigen::SparseMatrix<float>> Make_General_source(Eigen::RowVectorXf& sx,\
                                        Eigen::RowVectorXf& sz,\
                                        Eigen::RowVectorXf& M11,\
                                        Eigen::RowVectorXf& M12,\
                                        Eigen::RowVectorXf& M22,\
                                        int nz, int nx, int PML_thick){
    int nzPML = nz + 2 * PML_thick; int nxPML = nx + 2 * PML_thick; int NPML = nzPML * nxPML;
    int ns = sx.cols();
    Eigen::RowVectorXf z_skew = Eigen::RowVectorXf::Zero(ns);
    Eigen::RowVectorXf x_skew = Eigen::RowVectorXf::Zero(ns);
    Eigen::RowVectorXf roundsz = sz.array().round();
    Eigen::RowVectorXf roundsx = sx.array().round();

    auto sz_eq_rsz = sz.cwiseEqual(roundsz);
    auto sx_eq_rsx = sx.cwiseEqual(roundsx);

    for (int i = 0; i < sz_eq_rsz.cols(); i++){
        if (sz_eq_rsz(i))
            z_skew(i) = sz(i) - (round(sz(i)) - 1);
        else
            z_skew(i) = sz(i) - (round(sz(i)) - 0.5);
    }

    for (int i = 0; i < sx_eq_rsx.cols(); i++){
        if (sx_eq_rsx(i))
            x_skew(i) = sx(i) - (round(sx(i)) - 1);
        else
            x_skew(i) = sx(i) - (round(sx(i)) - 0.5);
    }

    int NE_stencil[4] = {nzPML - 1, nzPML, 0, -1};
    int SE_stencil[4] = {nzPML, nzPML + 1, 1, 0};
    int SW_stencil[4] = {0, 1, -nzPML + 1, -nzPML};
    int NW_stencil[4] = {-1, 0, -nzPML, -nzPML - 1};
    
    int dx_weights[4] = {1, 1, -1, -1};
    int dz_weights[4] = {-1, 1, 1, -1};

    Eigen::RowVectorXf cell_in = roundsz.array() + PML_thick + (PML_thick + roundsx.array()) * nzPML;
    cell_in.cast<int>();

    Eigen::SparseMatrix<float> Sx(NPML, ns); 
    Eigen::SparseMatrix<float> Sz(NPML, ns); 
    Eigen::SparseMatrix<float> dSxx(NPML, ns);
    Eigen::SparseMatrix<float> dSxz(NPML, ns);
    Eigen::SparseMatrix<float> dSzx(NPML, ns);
    Eigen::SparseMatrix<float> dSzz(NPML, ns);
    Eigen::SparseMatrix<float> dSxM11(NPML, ns);
    Eigen::SparseMatrix<float> dSxM12(NPML, ns);
    Eigen::SparseMatrix<float> dSzM12(NPML, ns);
    Eigen::SparseMatrix<float> dSzM22(NPML, ns);
    Sx.reserve(9 * ns); Sz.reserve(9 * ns);         
    dSxx.reserve(9 * ns); dSxz.reserve(9 * ns); dSzx.reserve(9 * ns); dSzz.reserve(9 * ns);
    dSxM11.reserve(9 * ns); dSxM12.reserve(9 * ns);
    dSzM12.reserve(9 * ns); dSzM22.reserve(9 * ns);
    Sx.setZero(); Sz.setZero(); dSxx.setZero(); dSxz.setZero();
    dSzx.setZero(); dSzz.setZero(); 
    dSxM11.setZero(); dSxM12.setZero();
    dSzM12.setZero(); dSzM22.setZero();
    
    for (int n = 0; n < ns; n++){
        float NE_weight = (1 - z_skew(n)) * x_skew(n);
        float SE_weight = z_skew(n) * x_skew(n);
        float SW_weight = z_skew(n) * (1 - x_skew(n));
        float NW_weight = (1 - z_skew(n)) * (1 - x_skew(n));
        /*std::cout << "=====================" << std::endl;
        std::cout << "NE-SE-SW-NW:" << NE_weight << "-" << SE_weight << "-" << SW_weight << "-" << NW_weight << std::endl;
        std::cout << "=====================" << std::endl;*/
        for (int i = 0; i < 4; i++){
            Sx.coeffRef(cell_in(n) + NE_stencil[i], n) = Sx.coeffRef(cell_in(n) + NE_stencil[i], n) + \
                                                NE_weight * (M11(n) * dx_weights[i] + \
                                                M12(n) * dz_weights[i]);
            Sz.coeffRef(cell_in(n) + NE_stencil[i], n) = Sz.coeffRef(cell_in(n) + NE_stencil[i], n) + \
                                                NE_weight * (M22(n) * dz_weights[i] + \
                                                M12(n) * dx_weights[i]);
            dSxx.coeffRef(cell_in(n) + NE_stencil[i], n) = dSxx.coeffRef(cell_in(n) + NE_stencil[i], n) + \
                                                           (1 - z_skew(n)) * (M11(n) * dx_weights[i] + \
                                                                              M12(n) * dz_weights[i]);
            dSzx.coeffRef(cell_in(n) + NE_stencil[i], n) = dSzx.coeffRef(cell_in(n) + NE_stencil[i], n) + \
                                                           (1 - z_skew(n)) * (M22(n) * dz_weights[i] + \
                                                                              M12(n) * dx_weights[i]);
            dSxz.coeffRef(cell_in(n) + NE_stencil[i], n) = dSxz.coeffRef(cell_in(n) + NE_stencil[i], n) + \
                                                           (-1 * x_skew(n)) * (M11(n) * dx_weights[i] + \
                                                                              M12(n) * dz_weights[i]);
            dSzz.coeffRef(cell_in(n) + NE_stencil[i], n) = dSzz.coeffRef(cell_in(n) + NE_stencil[i], n) + \
                                                           (-1 * x_skew(n)) * (M22(n) * dz_weights[i] + \
                                                                              M12(n) * dx_weights[i]);

            Sx.coeffRef(cell_in(n) + SE_stencil[i], n) = Sx.coeffRef(cell_in(n) + SE_stencil[i], n) + \
                                                SE_weight * (M11(n) * dx_weights[i] + \
                                                M12(n) * dz_weights[i]);
            Sz.coeffRef(cell_in(n) + SE_stencil[i], n) = Sz.coeffRef(cell_in(n) + SE_stencil[i], n) + \
                                                SE_weight * (M22(n) * dz_weights[i] + \
                                                M12(n) * dx_weights[i]);
            dSxx.coeffRef(cell_in(n) + SE_stencil[i], n) = dSxx.coeffRef(cell_in(n) + SE_stencil[i], n) + \
                                                           (z_skew(n)) * (M11(n) * dx_weights[i] + \
                                                            M12(n) * dz_weights[i]);
            dSzx.coeffRef(cell_in(n) + SE_stencil[i], n) = dSzx.coeffRef(cell_in(n) + SE_stencil[i], n) + \
                                                           (z_skew(n)) * (M22(n) * dz_weights[i] + \
                                                            M12(n) * dx_weights[i]);
            dSxz.coeffRef(cell_in(n) + SE_stencil[i], n) = dSxz.coeffRef(cell_in(n) + SE_stencil[i], n) + \
                                                           (x_skew(n)) * (M11(n) * dx_weights[i] + \
                                                            M12(n) * dz_weights[i]);
            dSzz.coeffRef(cell_in(n) + SE_stencil[i], n) = dSzz.coeffRef(cell_in(n) + SE_stencil[i], n) + \
                                                           (x_skew(n)) * (M22(n) * dz_weights[i] + \
                                                            M12(n) * dx_weights[i]);
            
            Sx.coeffRef(cell_in(n) + SW_stencil[i], n) = Sx.coeffRef(cell_in(n) + SW_stencil[i], n) + \
                                                SW_weight * (M11(n) * dx_weights[i] + \
                                                M12(n) * dz_weights[i]);
            Sz.coeffRef(cell_in(n) + SW_stencil[i], n) = Sz.coeffRef(cell_in(n) + SW_stencil[i], n) + \
                                                SW_weight * (M22(n) * dz_weights[i] + \
                                                M12(n) * dx_weights[i]);
            dSxx.coeffRef(cell_in(n) + SW_stencil[i], n) = dSxx.coeffRef(cell_in(n) + SW_stencil[i], n) + \
                                                           (-z_skew(n)) * (M11(n) * dx_weights[i] + \
                                                            M12(n) * dz_weights[i]);
            dSzx.coeffRef(cell_in(n) + SW_stencil[i], n) = dSzx.coeffRef(cell_in(n) + SW_stencil[i], n) + \
                                                           (-z_skew(n)) * (M22(n) * dz_weights[i] + \
                                                            M12(n) * dx_weights[i]);
            dSxz.coeffRef(cell_in(n) + SW_stencil[i], n) = dSxz.coeffRef(cell_in(n) + SW_stencil[i], n) + \
                                                           (1 - x_skew(n)) * (M11(n) * dx_weights[i] + \
                                                            M12(n) * dz_weights[i]);
            dSzz.coeffRef(cell_in(n) + SW_stencil[i], n) = dSzz.coeffRef(cell_in(n) + SW_stencil[i], n) + \
                                                           (1 - x_skew(n)) * (M22(n) * dz_weights[i] + \
                                                            M12(n) * dx_weights[i]);
            
            Sx.coeffRef(cell_in(n) + NW_stencil[i], n) = Sx.coeffRef(cell_in(n) + NW_stencil[i], n) + \
                                                NW_weight * (M11(n) * dx_weights[i] + \
                                                M12(n) * dz_weights[i]);
            Sz.coeffRef(cell_in(n) + NW_stencil[i], n) = Sz.coeffRef(cell_in(n) + NW_stencil[i], n) + \
                                                NW_weight * (M22(n) * dz_weights[i] + \
                                                M12(n) * dx_weights[i]);
            dSxx.coeffRef(cell_in(n) + NW_stencil[i], n) = dSxx.coeffRef(cell_in(n) + NW_stencil[i], n) + \
                                                           (-(1 - z_skew(n))) * (M11(n) * dx_weights[i] + \
                                                            M12(n) * dz_weights[i]);
            dSzx.coeffRef(cell_in(n) + SW_stencil[i], n) = dSzx.coeffRef(cell_in(n) + SW_stencil[i], n) + \
                                                           (-(1 - z_skew(n))) * (M22(n) * dz_weights[i] + \
                                                            M12(n) * dx_weights[i]);
            dSxz.coeffRef(cell_in(n) + SW_stencil[i], n) = dSxz.coeffRef(cell_in(n) + SW_stencil[i], n) + \
                                                           (-(1 - x_skew(n))) * (M11(n) * dx_weights[i] + \
                                                            M12(n) * dz_weights[i]);
            dSzz.coeffRef(cell_in(n) + SW_stencil[i], n) = dSzz.coeffRef(cell_in(n) + SW_stencil[i], n) + \
                                                           (-(1 - x_skew(n))) * (M22(n) * dz_weights[i] + \
                                                            M12(n) * dx_weights[i]);

            dSxM11.coeffRef(cell_in(n) + NE_stencil[i], n) = dSxM11.coeffRef(cell_in(n) + NE_stencil[i], n) + \
                                                             NE_weight * dx_weights[i];
            dSxM12.coeffRef(cell_in(n) + NE_stencil[i], n) = dSxM12.coeffRef(cell_in(n) + NE_stencil[i], n) + \
                                                             NE_weight * dz_weights[i];
            dSzM12.coeffRef(cell_in(n) + NE_stencil[i], n) = dSzM12.coeffRef(cell_in(n) + NE_stencil[i], n) + \
                                                             NE_weight * dx_weights[i];
            dSzM22.coeffRef(cell_in(n) + NE_stencil[i], n) = dSzM22.coeffRef(cell_in(n) + NE_stencil[i], n) + \
                                                             NE_weight * dz_weights[i];
            
            dSxM11.coeffRef(cell_in(n) + SE_stencil[i], n) = dSxM11.coeffRef(cell_in(n) + SE_stencil[i], n) + \
                                                             SE_weight * dx_weights[i];                                                                                            
            dSxM12.coeffRef(cell_in(n) + SE_stencil[i], n) = dSxM12.coeffRef(cell_in(n) + SE_stencil[i], n) + \
                                                             SE_weight * dz_weights[i];
            dSzM12.coeffRef(cell_in(n) + SE_stencil[i], n) = dSzM12.coeffRef(cell_in(n) + SE_stencil[i], n) + \
                                                             SE_weight * dx_weights[i];
            dSzM22.coeffRef(cell_in(n) + SE_stencil[i], n) = dSzM22.coeffRef(cell_in(n) + SE_stencil[i], n) + \
                                                             SE_weight * dz_weights[i];

            dSxM11.coeffRef(cell_in(n) + SW_stencil[i], n) = dSxM11.coeffRef(cell_in(n) + SW_stencil[i], n) + \
                                                             SW_weight * dx_weights[i];
            dSxM12.coeffRef(cell_in(n) + SW_stencil[i], n) = dSxM12.coeffRef(cell_in(n) + SW_stencil[i], n) + \
                                                             SW_weight * dz_weights[i];
            dSzM12.coeffRef(cell_in(n) + SW_stencil[i], n) = dSzM12.coeffRef(cell_in(n) + SW_stencil[i], n) + \
                                                             SW_weight * dx_weights[i];
            dSzM22.coeffRef(cell_in(n) + SW_stencil[i], n) = dSzM22.coeffRef(cell_in(n) + SW_stencil[i], n) + \
                                                             SW_weight * dz_weights[i];

            dSxM11.coeffRef(cell_in(n) + NW_stencil[i], n) = dSxM11.coeffRef(cell_in(n) + NW_stencil[i], n) + \
                                                             NW_weight * dx_weights[i];
            dSxM12.coeffRef(cell_in(n) + NW_stencil[i], n) = dSxM12.coeffRef(cell_in(n) + NW_stencil[i], n) + \
                                                             NW_weight * dz_weights[i];
            dSzM12.coeffRef(cell_in(n) + NW_stencil[i], n) = dSzM12.coeffRef(cell_in(n) + NW_stencil[i], n) + \
                                                             NW_weight * dx_weights[i];
            dSzM22.coeffRef(cell_in(n) + NW_stencil[i], n) = dSzM22.coeffRef(cell_in(n) + NW_stencil[i], n) + \
                                                             NW_weight * dz_weights[i];
        }
    }
    
    Eigen::SparseMatrix<float> S(2 * NPML, ns); S.reserve(9 * ns);
    typedef Eigen::Triplet<float> T;
    std::vector<T> triplet;
    triplet.reserve(9 * ns);
    for (int i = 0; i < Sx.outerSize(); i++){
        for (Eigen::SparseMatrix<float>::InnerIterator it(Sx, i); it; ++it){
            if(it.value())
                triplet.push_back(T(2 * it.row(), it.col(), it.value()));
        }
        for (Eigen::SparseMatrix<float>::InnerIterator itz(Sz, i); itz; ++itz){
            if(itz.value())
                triplet.push_back(T(2 * itz.row() + 1, itz.col(), itz.value()));
        }
    }
    S.setFromTriplets(triplet.begin(), triplet.end());
    triplet.clear(); triplet.shrink_to_fit();
    Eigen::SparseMatrix<float> dS(2 * NPML, 5 * ns); dS.reserve(18 * ns);
    std::vector<T> triplets;
    triplets.reserve(18 * ns);
    for (int i = 0; i < dSxx.outerSize(); i++){
        for (Eigen::SparseMatrix<float>::InnerIterator itxx(dSxx, i); itxx; ++itxx){
            if(itxx.value())
                triplets.push_back(T(2 * itxx.row(), itxx.col(), itxx.value()));
        }
        for (Eigen::SparseMatrix<float>::InnerIterator itzx(dSzx, i); itzx; ++itzx){
            if(itzx.value())
                triplets.push_back(T(2 * itzx.row() + 1, itzx.col(), itzx.value()));
        }
        for (Eigen::SparseMatrix<float>::InnerIterator itxz(dSxz, i); itxz; ++itxz){
            if(itxz.value())
                triplets.push_back(T(2 * itxz.row(), itxz.col() + ns, itxz.value()));
        }
        for (Eigen::SparseMatrix<float>::InnerIterator itzz(dSzz, i); itzz; ++itzz){
            if(itzz.value())
                triplets.push_back(T(2 * itzz.row() + 1, itzz.col() + ns, itzz.value()));
        }
        for (Eigen::SparseMatrix<float>::InnerIterator itxM11(dSxM11, i); itxM11; ++itxM11){
            if(itxM11.value())
                triplets.push_back(T(2 * itxM11.row(), itxM11.col() + 2 * ns, itxM11.value()));
        }
        for (Eigen::SparseMatrix<float>::InnerIterator itxM12(dSxM12, i); itxM12; ++itxM12){
            if(itxM12.value())
                triplets.push_back(T(2 * itxM12.row(), itxM12.col() + 3 * ns, itxM12.value()));
        }
        for (Eigen::SparseMatrix<float>::InnerIterator itzM12(dSzM12, i); itzM12; ++itzM12){
            if(itzM12.value())
                triplets.push_back(T(2 * itzM12.row() + 1, itzM12.col() + 3 * ns, itzM12.value()));
        }
        for (Eigen::SparseMatrix<float>::InnerIterator itzM22(dSzM22, i); itzM22; ++itzM22){
            if(itzM22.value())
                triplets.push_back(T(2 * itzM22.row() + 1, itzM22.col() + 4 * ns, itzM22.value()));
        }
    }
    dS.setFromTriplets(triplets.begin(), triplets.end());
    triplets.clear(); triplets.shrink_to_fit();
    z_skew.resize(0); x_skew.resize(0);
    roundsz.resize(0); roundsx.resize(0);
    sz_eq_rsz.resize(0); sx_eq_rsx.resize(0);
    cell_in.resize(0);
    Sx.resize(0, 0); Sx.data().squeeze();
    Sz.resize(0, 0); Sz.data().squeeze();
    dSxx.resize(0, 0); dSxx.data().squeeze();
    dSxz.resize(0, 0); dSxz.data().squeeze();
    dSzx.resize(0, 0); dSzx.data().squeeze();
    dSzz.resize(0, 0); dSzz.data().squeeze();
    dSxM11.resize(0, 0); dSxM11.data().squeeze();
    dSxM12.resize(0, 0); dSxM12.data().squeeze();
    dSzM12.resize(0, 0); dSzM12.data().squeeze();
    dSzM22.resize(0, 0); dSzM22.data().squeeze();

// Releasing, and make tuples
    return std::make_tuple(S, dS);
}

