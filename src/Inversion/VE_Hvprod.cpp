#include "../General/includefile.h"
#include "VE_Hvprod.h"
#include "../Setup/Make_General_source.h"
#include "../Forward/Assemble_Helm.h"
#include "Make_Helm_Anelastic_and_Derivative_efficiency.h"
#include "MADI_P.h"
#include "IP_dpSA_B_P.h"
#include "Diff0_Reg_and_diag_source.h"

Eigen::MatrixXf VE_Hvprod(Eigen::MatrixXf &model, Eigen::MatrixXf& ssmodel0, Eigen::MatrixXf& v, Eigen::RowVectorXf& frequency, \
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
    std::vector<Eigen::SparseMatrix<std::complex<float>>> MADI_new(9);
    for (int i = 0; i < 9; i++)
        MADI_new[i] = Eigen::SparseMatrix<std::complex<float>>(MADI[0].cols(), P.cols());
    Eigen::SparseMatrix<std::complex<float>> A(2 * nzPML * nxPML, 2 * nzPML * nxPML);
    Eigen::SparseMatrix<std::complex<float>> MA(A.rows(), A.cols()); 
    Eigen::SparseMatrix<std::complex<float>> Gs(nzPML * nxPML * 2, S.cols());
    //Eigen::SparseMatrix<std::complex<double>> Gs_temp(nzPML * nxPML * 2, S.cols());
    Eigen::SparseMatrix<std::complex<float>> Gs_temp(nzPML * nxPML * 2, S.cols());

    typedef Eigen::Triplet<float> T;
    std::vector<T> triplets;          

Eigen::initParallel();
//int nthreads = Eigen::nbThreads( );
#pragma omp parallel num_threads(frequency.size()) firstprivate(ssmodel, S, MADI, MADI_new, A, MA,\
                                  Gs, Gs_temp, triplets, vm)
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
        bool flag_A = Assemble_Helm(nzPML, nxPML, A, MADI[0]);
        
        //Eigen::SparseLU<Eigen::SparseMatrix<std::complex<double>>> solver(A.cast<std::complex<double>>());
        Eigen::SparseLU<Eigen::SparseMatrix<std::complex<float>>> solver(A);

        //Gs_temp = solver.solve((S * fwave(iii, 0)).cast<std::complex<double>>());
        //Gs = Gs_temp.cast<std::complex<float>>();
        Gs = solver.solve(S * fwave(iii, 0));

        bool flag_new = MADI_P(nzPML, NPML, MADI_new, MADI, P_big);

        Eigen::MatrixXcf MADI_phi(MADI[0].rows(), MADI[0].cols());
        MADI_phi.setZero();
        Eigen::SparseMatrix<float> v_sparse(MADI_new[0].rows(), MADI_new[0].cols());
        Eigen::SparseMatrix<std::complex<float>> product(MADI_new[0].rows(), MADI_new[0].cols());         

        for (int n = 0; n < 9; n++){
            for (int i = 0; i < MADI_new[n].outerSize(); i++){
                for (Eigen::SparseMatrix<std::complex<float>>::InnerIterator it(MADI_new[n], i); it; ++it){
                    if (it.value() != std::complex<float>(0, 0))
                        triplets.push_back(T(it.row(), it.col(), vm(it.col())));
                }
            }
            v_sparse.setFromTriplets(triplets.begin(), triplets.end());
            product = MADI_new[n].cwiseProduct(v_sparse);
            MADI_phi.row(n) = product * Eigen::VectorXf::Ones(product.cols());

            triplets.clear(); 
            v_sparse.setZero();
            product.setZero();
        }

        bool flag_MA = Assemble_Helm(nzPML, nxPML, MA, MADI_phi);
        Eigen::SparseMatrix<std::complex<float>> phi = -MA * Gs;

        Eigen::SparseMatrix<std::complex<float>> J(A.rows(), phi.size());
        //Eigen::SparseMatrix<std::complex<double>> J_temp(A.rows(), phi.size());
        //J_temp = solver.solve(phi.cast<std::complex<double>>());
        //J = J_temp.cast<std::complex<float>>();;
        J = solver.solve(phi);
                                                     
        Eigen::SparseMatrix<std::complex<float>> xi(R.rows(), J.cols());                                                                                                                                                                                                                             
        //Eigen::SparseMatrix<std::complex<double>> xii(R.rows(), J.cols());
        //Eigen::SparseLU<Eigen::SparseMatrix<std::complex<double>>> solver2(-A.adjoint().cast<std::complex<double>>());
        //xii = solver2.solve((R.transpose() * R * J).cast<std::complex<double>>());
        Eigen::SparseLU<Eigen::SparseMatrix<std::complex<float>>> solver2(-A.adjoint());
        //xi = xii.cast<std::complex<float>>();
        xi = solver2.solve((R.transpose() * R * J));
        prodm += scale_mod * IP_dpSA_B_P( Gs, xi, MADI_new, nz, nx, PML_thick); 

        for (int i = 0; i < 26; i++)
            MADI[i].setZero();
        for (int i = 0; i < 9; i++)
            MADI_new[i].setZero();
        A.setZero();
        MA.setZero();
        Gs.setZero();
        Gs_temp.setZero();
        MADI_phi.resize(0, 0); 
        J.resize(0, 0); J.data().squeeze();
        //J_temp.resize(0, 0); J_temp.data().squeeze();
        xi.resize(0, 0); xi.data().squeeze();
        //xii.resize(0, 0); xii.data().squeeze();
        v_sparse.resize(0, 0); v_sparse.data().squeeze();
        product.resize(0, 0); product.data().squeeze();
    }

    Eigen::MatrixXf prod = Eigen::MatrixXf(v.rows(), v.cols());
    prod.setZero();
    prod.block(0, 0, vm.size(), 1) = prodm;

    //float reg_scale0 = prod.cast<double>().norm() / v.cast<double>().norm();
    //if (v.cast<double>().norm() == 0)
    //    reg_scale0 = 0.0;
    float reg_scale0 = prod.norm() / v.norm();
    if (v.norm() == 0)
        reg_scale0 = 0.0;
    float reg_scale = reg_scale0;
    Eigen::SparseMatrix<float> rH(model.size(), model.size());
    bool flag_rH = Diff0_Reg_and_diag_source(model, reg_fac, stabregfac, \
                                                              nz, nx, rH, P, reg_scale);
    //Eigen::MatrixXd prodtemp = rH.cast<double>() * v.cast<double>();
    Eigen::MatrixXf prodtemp = rH * v;
    //prod += prodtemp.cast<float>();
    prod += prodtemp;

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
    MADI_new.clear(); MADI_new.shrink_to_fit();
    triplets.shrink_to_fit();

    return prod;
}


