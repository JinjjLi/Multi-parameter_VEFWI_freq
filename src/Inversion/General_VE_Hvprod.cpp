#include "../General/includefile.h"
#include "General_VE_Hvprod.h"
#include "../Setup/Make_General_source.h"
#include "../Forward/Assemble_Helm.h"
#include "Make_Helm_Anelastic_and_Derivative_efficiency.h"
#include "MADI_P.h"
#include "IP_dpSA_B.h"
#include "../General/mat_indexing.h"
#include "Diff0_Reg_and_diag_source.h"

Eigen::MatrixXf General_VE_Hvprod(Eigen::MatrixXf &model, Eigen::MatrixXf& ssmodel0, Eigen::MatrixXf& v, Eigen::RowVectorXf& frequency, \
                                 float omega0, int nz, int nx, int dz, int PML_thick, Eigen::SparseMatrix<float>& R, \
                                 Eigen::SparseMatrix<float>& S, \
                                 Eigen::MatrixXcf& fwave, float reg_fac, float stabregfac,\
                                 std::tuple<float *, float, float, float>& scale, \
                                 Eigen::SparseMatrix<float>& P, Eigen::SparseMatrix<float>& P_big){
    
    float * scalem; float scaleS, scale_mod, scaleSM;
    std::tie(scalem, scaleS, scale_mod, scaleSM) = scale;

    int nzPML = nz + 2 * PML_thick; int nxPML = nx + 2 * PML_thick;
    int NPML = nzPML * nxPML;
    Eigen::MatrixXf ssmodel = ssmodel0 + P * model * scale_mod;

    Eigen::MatrixXf vm = scale_mod * v;

    Eigen::MatrixXf prodm = Eigen::MatrixXf(vm.rows(), vm.cols());
    prodm.setZero();

    std::vector<Eigen::MatrixXcf> MADI(26);
    for (int i = 0; i < 26; i++){
        MADI[i] = Eigen::MatrixXcf(9, 2 * nzPML * nxPML);
        MADI[i].setZero();
    }
    Eigen::SparseMatrix<std::complex<float>> A(2 * nzPML * nxPML, 2 * nzPML * nxPML);
    Eigen::SparseMatrix<std::complex<float>> MA(A.rows(), A.cols()); 
    Eigen::SparseMatrix<std::complex<float>> Gs(nzPML * nxPML * 2, S.cols());
    Eigen::SparseMatrix<std::complex<double>> Gs_temp(nzPML * nxPML * 2, S.cols());
    //Eigen::SparseMatrix<std::complex<float>> Gs_temp(nzPML * nxPML * 2, S.cols());

    typedef Eigen::Triplet<float> T;
    std::vector<T> triplets;          
    Eigen::MatrixXf Pvm; Pvm.setZero();
    Pvm = P_big * vm;
    Eigen::MatrixXf vbig = Eigen::MatrixXf(2 * NPML, 5);
    vbig.setZero();
    Pvm.resize(NPML, 5);
    vbig(Eigen::seq(0, 2 * NPML - 1, 2), Eigen::all) = Pvm;
    vbig(Eigen::seq(1, 2 * NPML - 1, 2), Eigen::all) = Pvm;

    Eigen::SparseMatrix<std::complex<float>> phi(nzPML * nxPML * 2, S.cols());
    phi.setZero();

Eigen::initParallel();
//int nthreads = Eigen::nbThreads( );
#pragma omp parallel num_threads(frequency.size()) firstprivate(ssmodel, S, MADI, A, MA, vbig, phi, \
                                  Gs, Gs_temp, triplets)
#pragma omp declare reduction (+: Eigen::MatrixXf: omp_out=omp_out+omp_in)\
     initializer(omp_priv=omp_orig)
#pragma omp for reduction(+:prodm)

    for (int iii = 0; iii < frequency.size(); iii++){
        float omega = 2 * pi * frequency(iii);
        bool flag = Make_Helm_Anelastic_and_Derivative_efficiency(MADI, ssmodel, \
                                                                  nx, nz, \
                                                                  omega, omega0, \
                                                                  dz, PML_thick, \
                                                                  scalem);
        std::vector<Eigen::MatrixXcf> MADI_phi = MADI;
        bool flag_A = Assemble_Helm(nzPML, nxPML, A, MADI[0]);
        Eigen::SparseLU<Eigen::SparseMatrix<std::complex<double>>> solver(A.cast<std::complex<double>>());
        //Eigen::UmfPackLU<Eigen::SparseMatrix<std::complex<double>>> solver(A.cast<std::complex<double>>());
        Gs_temp = solver.solve((S * fwave(iii, 0)).cast<std::complex<double>>());
        Gs = Gs_temp.cast<std::complex<float>>();
        Gs_temp.setZero();
        //Eigen::SparseLU<Eigen::SparseMatrix<std::complex<float>>> solver(A);
        //Gs = solver.solve(S * fwave(iii, 0));


        int offset;
        for (int m = 1; m < 26; m++){
            if (m - (std::floor((m - 1) / 5) * 5 + 1) == 0)
                offset = 0;
            else if (m - (std::floor((m - 1) / 5) * 5 + 1) == 1)
                offset = 2 * nzPML;
            else if (m - (std::floor((m - 1) / 5) * 5 + 1) == 2)
                offset = -2 * nzPML;
            else if (m - (std::floor((m - 1) / 5) * 5 + 1) == 3)
                offset = 2;
            else if (m - (std::floor((m - 1) / 5) * 5 + 1) == 4)
                offset = -2;
            int varind = std::floor((m - 1) / 5);

            for (int n = 0; n < 9; n++)
            {
                int th;
                int minind;
                int maxind;
                if ((n + 1) / 2.0 == std::floor((n + 1) / 2.0) || n == 4){
                    Eigen::ArrayXi ind = Eigen::ArrayXi::LinSpaced(2 * NPML, 0, 2 * NPML - 1);
                    th = 0;
                    std::vector<std::pair<int, int>> indices;
                    visit_lambda(ind + offset, [&indices, th](int v, int i, int j){
                        if (v >= th){
                            indices.push_back(std::make_pair(i, j));
                            return;
                        }
                    });
                    minind = indices[0].first;
                    indices.clear();

                    th = 2 * NPML;
                    visit_lambda(ind + offset, [&indices, th](int v, int i, int j){
                        if (v < th){
                            indices.push_back(std::make_pair(i, j));
                            return;
                        }
                    });
                    maxind = indices[indices.size() - 1].first;
                    indices.clear();

                    //Eigen::ArrayXi newind = ind(Eigen::ArrayXi::LinSpaced(maxind - minind + 1, minind, maxind), Eigen::all);
                    Eigen::ArrayXi newind = ind(Eigen::seq(minind, maxind), Eigen::all);
                    MADI_phi[m](n, newind) = MADI_phi[m](n, newind).cwiseProduct((vbig(newind + offset, varind)).transpose());
                    //MADI_phi[m](n, newind) = MADI_phi[m](n, newind).cwiseProduct(\
                    //                        mat_indexing(vbig, newind + offset, Eigen::ArrayXi::Constant(1, 1, varind)).transpose());
                    ind.resize(0, 0);
                    newind.resize(0, 0);
                }
                else{
                    Eigen::ArrayXi allind = Eigen::ArrayXi::LinSpaced(2 * NPML, 0, 2 * NPML - 1);
                    Eigen::ArrayXi ind = allind(Eigen::seq(0, 2 * NPML - 1, 2), Eigen::all);
                    //Eigen::ArrayXi ind = Eigen::ArrayXi::LinSpaced(NPML, 0, 2 * NPML - 2);
                    th = 0;
                    std::vector<std::pair<int, int>> indices;
                    visit_lambda(ind + offset, [&indices, th](int v, int i, int j){
                        if (v >= th){
                            indices.push_back(std::make_pair(i, j));
                            return;
                        }
                    });
                    minind = indices[0].first;
                    indices.clear();

                    th = 2 * NPML;
                    visit_lambda(ind + offset, [&indices, th](int v, int i, int j){
                        if (v < th){
                            indices.push_back(std::make_pair(i, j));
                            return;
                        }
                    });
                    maxind = indices[indices.size() - 1].first;
                    indices.clear();

                    Eigen::ArrayXi newind = ind(Eigen::seq(minind, maxind), Eigen::all);
                    MADI_phi[m](n, newind) = MADI_phi[m](n, newind).cwiseProduct((vbig(newind + offset, varind)).transpose());
                    ind.setZero();
                    newind.setZero();

                    ind = allind(Eigen::seq(1, 2 * NPML - 1, 2), Eigen::all);
                    //ind = Eigen::ArrayXi::LinSpaced(NPML, 1, 2 * NPML - 1);
                    th = 0;
                    visit_lambda(ind + offset, [&indices, th](int v, int i, int j){
                        if (v >= th){
                            indices.push_back(std::make_pair(i, j));
                            return;
                        }
                    });
                    minind = indices[0].first;
                    indices.clear();

                    th = 2 * NPML;
                    visit_lambda(ind + offset, [&indices, th](int v, int i, int j){
                        if (v < th){
                            indices.push_back(std::make_pair(i, j));
                            return;
                        }
                    });
                    maxind = indices[indices.size() - 1].first;
                    indices.clear();

                    newind = ind(Eigen::seq(minind, maxind), Eigen::all);
                    MADI_phi[m](n, newind) = MADI_phi[m](n, newind).cwiseProduct((vbig(newind + offset, varind)).transpose());
                    ind.resize(0, 0);
                    newind.resize(0, 0);
                    allind.resize(0, 0);
                    }
            }
            bool flag_MA = Assemble_Helm(nzPML, nxPML, MA, MADI_phi[m]);
            phi -= MA * Gs;
            MA.setZero();
        }
        vbig.setZero();
        MADI_phi.clear(); MADI_phi.shrink_to_fit();
        
        Eigen::SparseMatrix<std::complex<float>> J(A.rows(), phi.size());
        Eigen::SparseMatrix<std::complex<double>> J_temp(A.rows(), phi.size());
        J_temp = solver.solve(phi.cast<std::complex<double>>());
        J = J_temp.cast<std::complex<float>>();
        //Eigen::SparseLU<Eigen::SparseMatrix<std::complex<float>>> solver1(A);
        //J = solver1.solve(phi);
        phi.setZero();
        J_temp.resize(0, 0); J_temp.data().squeeze();

        Eigen::SparseMatrix<std::complex<float>> xi(R.rows(), J.cols());
        Eigen::SparseMatrix<std::complex<double>> xii(R.rows(), J.cols());
        Eigen::SparseLU<Eigen::SparseMatrix<std::complex<double>>> solver2(-A.adjoint().cast<std::complex<double>>());
        xii = solver2.solve((R.transpose() * R * J).cast<std::complex<double>>());
        xi = xii.cast<std::complex<float>>();
        xii.resize(0, 0); xii.data().squeeze();
        //Eigen::SparseLU<Eigen::SparseMatrix<std::complex<float>>> solver2(-A.adjoint());
        //xi = solver2.solve(R.transpose() * R * J);
        J.resize(0, 0); J.data().squeeze();

        prodm +=scale_mod * P.transpose() * IP_dpSA_B( Gs, xi, MADI, nz, nx, PML_thick); 

        A.setZero();
        MA.setZero();
        Gs.setZero();
        xi.setZero();
    }

    //Eigen::MatrixXf prod = Eigen::MatrixXf(v.rows(), v.cols());
    //prod.setZero();
    Eigen::MatrixXf prod = prodm;

    float reg_scale0 = prod.cast<double>().norm() / v.cast<double>().norm();
    if (v.cast<double>().norm() == 0)
        reg_scale0 = 0.0;
    //float reg_scale0 = prod.norm() / v.norm();
    //if (v.norm() == 0)
    //    reg_scale0 = 0.0;
    float reg_scale = reg_scale0;
    Eigen::SparseMatrix<float> rH(model.size(), model.size());
    bool flag_rH = Diff0_Reg_and_diag_source(model, reg_fac, stabregfac, \
                                                              nz, nx, rH, P, reg_scale);
    Eigen::MatrixXd prodtemp = rH.cast<double>() * v.cast<double>();
    prod += prodtemp.cast<float>();
    //Eigen::MatrixXf prodtemp = rH * v;
    //prod += prodtemp;

    A.resize(0, 0); A.data().squeeze();
    Gs.resize(0, 0); Gs.data().squeeze();
    Gs_temp.resize(0, 0); Gs_temp.data().squeeze();
    MA.resize(0, 0); MA.data().squeeze();
    vm.resize(0, 0); 
    ssmodel.resize(0, 0);
    prodm.resize(0, 0); 
    prodtemp.resize(0, 0);
    rH.resize(0, 0); rH.data().squeeze();
    MADI.clear(); MADI.shrink_to_fit();
    triplets.shrink_to_fit();
    Pvm.resize(0, 0);
    vbig.resize(0, 0);

    return prod;
}


