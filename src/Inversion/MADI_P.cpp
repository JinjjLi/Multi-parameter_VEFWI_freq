#include "../General/includefile.h"
#include "MADI_P.h"

bool MADI_P(int nzPML, int NPML, std::vector<Eigen::SparseMatrix<std::complex<float>>>& MADI_new, std::vector<Eigen::MatrixXcf>& MADI, Eigen::SparseMatrix<float>& P){
    Eigen::MatrixXf c_star_space(1, 5), c_star_var(1, 5);
    c_star_space << 0, nzPML, -nzPML, 1, -1;
    c_star_var << 0, NPML, 2 * NPML, 3 * NPML, 4 * NPML;
    Eigen::MatrixXf c_star = c_star_space.transpose() * Eigen::MatrixXf::Ones(1, 5) + \
                             Eigen::MatrixXf::Ones(5, 1) * c_star_var;
    c_star.resize(c_star.size(), 1);
    Eigen::MatrixXf tempone = Eigen::MatrixXf::Ones(2, 1);
    Eigen::ArrayXf templin = Eigen::ArrayXf::LinSpaced(NPML, 1, NPML);
    Eigen::MatrixXf templinmat(1, templin.size());
    templinmat.row(0) = templin;
    Eigen::MatrixXf loc_ind = tempone * templinmat;
    loc_ind.resize(loc_ind.size(), 1);
    Eigen::MatrixXf c = c_star * Eigen::MatrixXf::Ones(1, loc_ind.size()) + \
                        Eigen::MatrixXf::Ones(25, 1) * loc_ind.transpose();

    Eigen::MatrixXf c_xcheck = ((c.array() - 1) / NPML).floor() + 1;
    Eigen::MatrixXf c_xcheck2 = Eigen::MatrixXf(c_xcheck.rows(), c_xcheck.cols());
    c_xcheck2.setZero();
    for (int n = 0; n < 5; n++)
        for (int m = 0; m < 5; m++)
            c_xcheck2.row(n + m * 5) = c_xcheck.row(n + m * 5).array() - m - 1;

    Eigen::MatrixXf c_zcheck = ((c.array() - 1) / nzPML).floor();
    Eigen::MatrixXf c_zcheck2 = Eigen::MatrixXf(c_zcheck.rows(), c_zcheck.cols());
    c_zcheck2.setZero();
    for (int m = 0; m < 5; m++){
        c_zcheck2.row(0 + m * 5) = c_zcheck.row(0 + m * 5).array() - c_zcheck.row(0 + m * 5).array();
        c_zcheck2.row(3 + m * 5) = c_zcheck.row(3 + m * 5).array() - c_zcheck.row(0 + m * 5).array();
        c_zcheck2.row(4 + m * 5) = c_zcheck.row(4 + m * 5).array() - c_zcheck.row(0 + m * 5).array();
    }

    c = (c_xcheck2.array() != 0).select(0, c);
    c = (c_zcheck2.array() != 0).select(0, c);
    Eigen::SparseMatrix<std::complex<float>> phi_diag1(5 * NPML, NPML);
    Eigen::SparseMatrix<std::complex<float>> phi_diag2(5 * NPML, NPML);
    //std::vector<Eigen::SparseMatrix<std::complex<float>>> MADI_der(9);
    Eigen::SparseMatrix<std::complex<float>> MADIn1(5 * NPML, NPML); 
    Eigen::SparseMatrix<std::complex<float>> MADIn2(5 * NPML, NPML); 
    Eigen::MatrixXcf phitemp1 = Eigen::MatrixXcf(NPML, 1);
    Eigen::MatrixXcf phitemp2 = Eigen::MatrixXcf(NPML, 1);
    phitemp1.setZero(); phitemp2.setZero();
    for (int qqq = 0; qqq < 9; qqq++){
        MADIn1.reserve(25 * MADI[0].cols()); 
        MADIn2.reserve(25 * MADI[0].cols());             
        for (int nnn = 0; nnn < 25; nnn++){
            Eigen::MatrixXcf MADI_temp = MADI[nnn + 1].row(qqq).adjoint();
            phitemp1 = MADI_temp(Eigen::seq(0, 2 * NPML - 1, 2), Eigen::all);
            phitemp2 = MADI_temp(Eigen::seq(1, 2 * NPML - 1, 2), Eigen::all);

            Eigen::MatrixXi c_star_int = c_star.block(nnn, 0, 1, 1).cast<int>();
            spdiags<std::complex<float>>(phi_diag1, phitemp1, -c_star_int, 5 * NPML, NPML);
            spdiags<std::complex<float>>(phi_diag2, phitemp2, -c_star_int, 5 * NPML, NPML);
            MADIn1 += phi_diag1; MADIn2 += phi_diag2;     
            MADI_temp.resize(0, 0); 
            c_star_int.resize(0, 0); 
            phitemp1.setZero(); 
            phitemp2.setZero();
            phi_diag1.setZero();
            phi_diag2.setZero();
        }
        Eigen::SparseMatrix<std::complex<float>> MADI_der_temp1 = (P.transpose() * MADIn1).adjoint();
        Eigen::SparseMatrix<std::complex<float>> MADI_der_temp2 = (P.transpose() * MADIn2).adjoint();
        typedef Eigen::Triplet<std::complex<float>> T;
        std::vector<T> triplets;
        //triplets.reserve(MADI_der_temp1.nonZeros() + MADI_der_temp2.nonZeros());
        for (int i = 0; i < MADI_der_temp1.outerSize(); i++){
            for (Eigen::SparseMatrix<std::complex<float>>::InnerIterator it(MADI_der_temp1, i); it; ++it){
                if (it.value() != std::complex<float>(0, 0))
                    triplets.push_back(T(2 * it.row(), it.col(), it.value()));
            }
        }
        for (int i = 0; i < MADI_der_temp2.outerSize(); i++){
            for (Eigen::SparseMatrix<std::complex<float>>::InnerIterator it(MADI_der_temp2, i); it; ++it){
                if (it.value() != std::complex<float>(0, 0))
                    triplets.push_back(T(2 * it.row() + 1, it.col(), it.value()));
            }
        }
        MADI_new[qqq].setFromTriplets(triplets.begin(), triplets.end());
        MADIn1.setZero(); MADIn2.setZero();
        MADI_der_temp1.resize(0, 0); MADI_der_temp1.data().squeeze();
        MADI_der_temp2.resize(0, 0); MADI_der_temp2.data().squeeze();
        triplets.clear(); triplets.shrink_to_fit();
    }
    c_star_space.resize(0, 0); c_star_var.resize(0, 0);
    c_star.resize(0, 0); tempone.resize(0, 0); 
    templin.resize(0, 0); templinmat.resize(0, 0);
    loc_ind.resize(0, 0); c.resize(0, 0);
    c_xcheck.resize(0, 0); c_xcheck2.resize(0, 0);
    c_zcheck.resize(0, 0); c_zcheck2.resize(0, 0);
    c.resize(0, 0);
    MADIn1.data().squeeze(); MADIn2.data().squeeze();
    phitemp1.resize(0, 0); phitemp2.resize(0, 0);
    phi_diag1.resize(0, 0); phi_diag1.data().squeeze();
    phi_diag2.resize(0, 0); phi_diag2.data().squeeze();

    return 1;
}

