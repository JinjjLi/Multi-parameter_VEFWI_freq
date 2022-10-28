#include "../General/includefile.h"
#include "VE_Gradient.h"
#include "../Setup/Make_General_source.h"
#include "Make_Helm_Anelastic_and_Derivative_efficiency.h"
#include "../Forward/Assemble_Helm.h"
#include "IP_dpSA_B.h"

std::tuple<float, Eigen::MatrixXf> VE_Gradient(Eigen::RowVectorXf& freq, Eigen::MatrixXcf& fwave, float omega0, Eigen::MatrixXf& model, \
                                    Eigen::MatrixXf& ssmodel0, std::vector<Eigen::SparseMatrix<std::complex<float>>>& D, Eigen::SparseMatrix<float>& R, \
                                    Eigen::SparseMatrix<float>& S, \
                                    int nz, int nx, int dz, int PML_thick, \
                                    std::tuple<float *, float, float, float>& scale, Eigen::SparseMatrix<float>& P){
    
    float * scalem; float scaleS, scale_mod, scaleSM;
    std::tie(scalem, scaleS, scale_mod, scaleSM) = scale;

    int nzPML = nz + 2 * PML_thick; int nxPML = nx + 2 * PML_thick;
    Eigen::MatrixXf ssmodel = ssmodel0 + P * model * scale_mod;

    Eigen::MatrixXf gm = Eigen::MatrixXf(ssmodel.rows(), ssmodel.cols());
    gm.setZero();

    std::vector<Eigen::MatrixXcf> MADI(26);
    for (int i = 0; i < 26; i++){
        MADI[i] = Eigen::MatrixXcf(9, 2 * nzPML * nxPML);
        MADI[i].setZero();
    }
    Eigen::SparseMatrix<std::complex<float>> A(2 * nxPML * nzPML, 2 * nxPML * nzPML);
    Eigen::SparseMatrix<std::complex<float>> Gs(nzPML * nxPML * 2, S.cols());
    //Eigen::SparseMatrix<std::complex<double>> Gs_temp(nzPML * nxPML * 2, S.cols());
    //Eigen::SparseMatrix<std::complex<double>> lamb_temp(nzPML * nxPML * 2, S.cols());
    Eigen::SparseMatrix<std::complex<float>> temp2(nzPML * nxPML * 2, S.cols());
    Eigen::SparseMatrix<std::complex<float>> lamb(nzPML * nxPML * 2, S.cols());

    float f = 0;

Eigen::initParallel();
//int nthreads = Eigen::nbThreads( );
#pragma omp parallel num_threads(freq.size()) firstprivate(ssmodel, S, MADI, A, Gs, lamb, temp2)
#pragma omp declare reduction (+: Eigen::MatrixXf: omp_out=omp_out+omp_in)\
     initializer(omp_priv=omp_orig)  
#pragma omp for reduction(+:gm)

    for (int n = 0; n < freq.size(); n++){
        float omega = 2 * pi * freq(n);
        bool flag = Make_Helm_Anelastic_and_Derivative_efficiency(MADI, ssmodel, \
                                                      nx, nz, \
                                                      omega, omega0, \
                                                      dz, PML_thick, \
                                                      scalem);
        bool flag_A = Assemble_Helm(nzPML, nxPML, A, MADI[0]);
        Eigen::SparseLU<Eigen::SparseMatrix<std::complex<float>>> solver(A);
        //Eigen::SparseLU<Eigen::SparseMatrix<std::complex<double>>> solver(A.cast<std::complex<double>>());
        //Eigen::UmfPackLU<Eigen::SparseMatrix<std::complex<double>>> solver(A.cast<std::complex<double>>());                                          

        //spdiags<std::complex<float>>(fcoeff, fwave.row(n), Eigen::MatrixXi::Zero(1, 1), fwave.cols(), fwave.cols());
        //temp = S * fcoeff;
        //Gs_temp = solver.solve((S * fwave(n, 0)).cast<std::complex<double>>());
        //Gs = Gs_temp.cast<std::complex<float>>();
        Gs = solver.solve(S * fwave(n, 0));
        Eigen::MatrixXcf res = R * Gs - D[n];
        f += 0.5 * res.norm() * res.norm();

        //Eigen::SparseLU<Eigen::SparseMatrix<std::complex<double>>> solver2(A.adjoint().cast<std::complex<double>>());
        //Eigen::UmfPackLU<Eigen::SparseMatrix<std::complex<double>>> solver2(A.adjoint().cast<std::complex<double>>());
        Eigen::SparseLU<Eigen::SparseMatrix<std::complex<float>>> solver2(A.adjoint());
        temp2 = (-R.adjoint() * res).sparseView();
        //lamb_temp = solver2.solve(temp2.cast<std::complex<double>>());
        //lamb = lamb_temp.cast<std::complex<float>>();
        lamb = solver2.solve(temp2);
        A.setZero();
        res.resize(0, 0); 

        Eigen::MatrixXf prod = IP_dpSA_B(Gs, lamb, MADI, nz, nx, PML_thick);
        gm += prod;
        prod.resize(0, 0);
        for (int i = 0; i < 26; i++)
            MADI[i].setZero();
        //MADI.clear(); MADI.shrink_to_fit();
        Gs.setZero();
        ssmodel.setZero();
        temp2.setZero();
        lamb.setZero();
    }
    
    Eigen::MatrixXf g(model.rows(), model.cols());
    g.setZero();
    g = P.transpose() * (gm * scale_mod);

    A.resize(0, 0); A.data().squeeze();
    Gs.resize(0, 0); Gs.data().squeeze();
    temp2.resize(0, 0);
    lamb.resize(0, 0); lamb.data().squeeze();
    gm.resize(0, 0);   
    ssmodel.resize(0, 0);
    MADI.clear(); MADI.shrink_to_fit();

    return std::make_tuple(f, g);
}

