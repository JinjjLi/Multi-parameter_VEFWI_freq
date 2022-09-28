#include "../General/includefile.h"
#include "Make_Helm_Anelastic_and_Derivative_efficiency.h"
#include "../Forward/MakePML.h"
#include "../Forward/PML_pad.h"

bool Make_Helm_Anelastic_and_Derivative_efficiency(std::vector<Eigen::MatrixXcf>& MADI, Eigen::MatrixXf model, \
                                                                            int nx, int nz, float omega, \
                                                                            float omega0, int dz, \
                                                                            int PML_thick, float * scale){
    int NN = nz * nx;
    int nxPML = nx + 2 * PML_thick; int nzPML = nz + 2 * PML_thick;
    for (int iii = 0; iii < 5; iii++)
        model.block(iii * NN, 0, NN, 1) *= scale[iii];
    float del = 1.0 / (dz * dz);
    Eigen::RowVectorXf ABCx0, ABCx1, ABCz0, ABCz1;
    std::tie(ABCx0, ABCx1, ABCz0, ABCz1) = MakePML(nz, nx, PML_thick);
    model.resize(nz * nx, 5);
    Eigen::MatrixXf rho, SSp, Qp_inv, SSs, Qs_inv;
    rho = model.col(0); rho.resize(nz, nx);
    SSp = model.col(1); SSp.resize(nz, nx);
    Qp_inv = model.col(2); Qp_inv.resize(nz, nx);
    SSs = model.col(3); SSs.resize(nz, nx);
    Qs_inv = model.col(4); Qs_inv.resize(nz, nx);

    //std::vector<Eigen::MatrixXcf> MADI(26);
    PML_pad(rho, PML_thick); PML_pad(SSp, PML_thick); PML_pad(Qp_inv, PML_thick);
    PML_pad(SSs, PML_thick); PML_pad(Qs_inv, PML_thick);

    Eigen::MatrixXcf vp = SSp.cast<std::complex<float>>().array().inverse().sqrt();
    Eigen::MatrixXcf vs = SSs.cast<std::complex<float>>().array().inverse().sqrt();
    std::complex<float> Qp_term, Qs_term;
    Qp_term.real((1 / pi) * log(omega/omega0)); Qp_term.imag(0.5);
    Qs_term.real((1 / pi) * log(omega/omega0)); Qs_term.imag(0.5);

    Eigen::MatrixXcf vp_complex = vp.array() * (1 + Qp_inv.array() * Qp_term);
    Eigen::MatrixXcf vs_complex = vs.array() * (1 + Qs_inv.array() * Qs_term);

    Eigen::MatrixXcf c11 = vp_complex.array().pow(2) * rho.array();
    Eigen::MatrixXcf c44 = vs_complex.array().pow(2) * rho.array();
    Eigen::MatrixXcf c13 = c11.array() - c44.array() * 2;
    Eigen::MatrixXcf c11rho = vp_complex.array().pow(2);
    Eigen::MatrixXcf c11vp = -vp_complex.array().pow(2) * vp.array().pow(2) * rho.array();
    Eigen::MatrixXcf c11Qp = 2 * vp_complex.array() * vp.array() * rho.array() * Qp_term;
    Eigen::MatrixXcf c44rho = vs_complex.array().pow(2);
    Eigen::MatrixXcf c44vs = -vs_complex.array().pow(2) * vs.array().pow(2) * rho.array();
    Eigen::MatrixXcf c44Qs = 2 * vs_complex.array() * vs.array() * rho.array() * Qs_term;
    Eigen::MatrixXcf c13rho = c11rho.array() - 2 * c44rho.array();
    Eigen::MatrixXcf c13vp = c11vp;
    Eigen::MatrixXcf c13Qp = c11Qp;
    Eigen::MatrixXcf c13vs = -2 * c44vs.array();
    Eigen::MatrixXcf c13Qs = -2 * c44Qs.array();

    std::vector<Eigen::MatrixXf> rhoik(26);
    std::vector<Eigen::MatrixXcf> c11ik(26), c11iadk(26), c11imik(26), c11ikad(26), c11ikmi(26);
    std::vector<Eigen::MatrixXcf> c13iadk(26), c13imik(26), c13ikad(26), c13ikmi(26);
    std::vector<Eigen::MatrixXcf> c44ik(26), c44iadk(26), c44imik(26), c44ikad(26), c44ikmi(26);

    for (int i = 0; i < 26; i++){
        rhoik[i] = Eigen::MatrixXf(nzPML, nxPML);   
        c11ik[i] = Eigen::MatrixXcf(nzPML, nxPML);
        c11iadk[i] = Eigen::MatrixXcf(nzPML, nxPML);
        c11imik[i] = Eigen::MatrixXcf(nzPML, nxPML);
        c11ikad[i] = Eigen::MatrixXcf(nzPML, nxPML);
        c11ikmi[i] = Eigen::MatrixXcf(nzPML, nxPML);
        c13iadk[i] = Eigen::MatrixXcf(nzPML, nxPML);
        c13imik[i] = Eigen::MatrixXcf(nzPML, nxPML);
        c13ikad[i] = Eigen::MatrixXcf(nzPML, nxPML);
        c13ikmi[i] = Eigen::MatrixXcf(nzPML, nxPML);
        c44ik[i] = Eigen::MatrixXcf(nzPML, nxPML);
        c44iadk[i] = Eigen::MatrixXcf(nzPML, nxPML);
        c44imik[i] = Eigen::MatrixXcf(nzPML, nxPML);
        c44ikad[i] = Eigen::MatrixXcf(nzPML, nxPML);
        c44ikmi[i] = Eigen::MatrixXcf(nzPML, nxPML);
        rhoik[i].setZero(); 
        c11ik[i].setZero();
        c11iadk[i].setZero();
        c11imik[i].setZero();
        c11ikad[i].setZero();
        c11ikmi[i].setZero();
        c13iadk[i].setZero();
        c13imik[i].setZero();
        c13ikad[i].setZero();
        c13ikmi[i].setZero();
        c44ik[i].setZero(); 
        c44iadk[i].setZero();
        c44imik[i].setZero();
        c44ikad[i].setZero();
        c44ikmi[i].setZero();
    }
    Eigen::MatrixXcf PML1(nzPML, nxPML); PML1.setZero();
    Eigen::MatrixXcf PML2(nzPML, nxPML); PML2.setZero();
    Eigen::MatrixXcf PML5(nzPML, nxPML); PML5.setZero();
    Eigen::MatrixXcf PML6(nzPML, nxPML); PML6.setZero();
    Eigen::MatrixXcf PML9(nzPML, nxPML); PML9.setZero();
    
    for (int ii = 0; ii < nxPML; ii++){
        for (int kk = 0; kk < nzPML; kk++){
            int imi = ii - 1; int kmi = kk - 1;
            int iad = ii + 1; int kad = kk + 1;
            if (imi < 0)
                imi = 0;
            if (kmi < 0)
                kmi = 0;
            if (iad > nxPML - 1)
                iad = nxPML - 1;
            if (kad > nzPML - 1)
                kad = nzPML - 1;
            std::complex<float> tempcomp(0, omega);
            PML1(kk, ii) = (tempcomp + ABCx0(ii)) * (tempcomp + ABCx1(ii));
            PML1(kk, ii) = -1 * omega * omega / PML1(kk, ii);
            PML2(kk, ii) = (tempcomp + ABCx0(ii)) * (tempcomp + ABCx1(imi));
            PML2(kk, ii) = -1 * omega * omega / PML2(kk, ii);
            PML5(kk, ii) = (tempcomp + ABCz0(kk)) * (tempcomp + ABCz1(kk));     
            PML5(kk, ii) = -1 * omega * omega / PML5(kk, ii);
            PML6(kk, ii) = (tempcomp + ABCz0(kk)) * (tempcomp + ABCz1(kmi));
            PML6(kk, ii) = -1 * omega * omega / PML6(kk, ii);
            PML9(kk, ii) = (tempcomp + ABCx0(ii)) * (tempcomp + ABCz0(kk));
            PML9(kk, ii) = -1 * omega * omega / PML9(kk, ii);
        }
    }

//Eigen::initParallel();
//int nthreads = Eigen::nbThreads();
//#pragma omp parallel num_threads(2) firstprivate(c11, c13, c44, c11rho, c11vp, c11Qp, c44rho, c44vs, c44Qs, c13rho, cc13vp, c13Qp, c13vs, c13Qs)
//#pragma omp for

    for (int n = 0; n < 26; n++){
        if (n == 0){
            rhoik[n] = rho;
            
            c11ik[n] = c11;
            c11iadk[n].block(0, 0, nzPML, nxPML - 1) = c11.block(0, 1, nzPML, nxPML - 1);
            c11iadk[n].col(nxPML - 1) = c11.col(nxPML - 1);
            c11imik[n].block(0, 1, nzPML, nxPML - 1) = c11.block(0, 0, nzPML, nxPML - 1);
            c11imik[n].col(0) = c11.col(0);
            c11ikad[n].block(0, 0, nzPML - 1, nxPML) = c11.block(1, 0, nzPML - 1, nxPML);
            c11ikad[n].row(nzPML - 1) = c11.row(nzPML - 1);
            c11ikmi[n].block(1, 0, nzPML - 1, nxPML) = c11.block(0, 0, nzPML - 1, nxPML);
            c11ikmi[n].row(0) = c11.row(0);

            c13iadk[n].block(0, 0, nzPML, nxPML - 1) = c13.block(0, 1, nzPML, nxPML - 1);
            c13iadk[n].col(nxPML - 1) = c13.col(nxPML - 1);
            c13imik[n].block(0, 1, nzPML, nxPML - 1) = c13.block(0, 0, nzPML, nxPML - 1);
            c13imik[n].col(0) = c13.col(0);
            c13ikad[n].block(0, 0, nzPML - 1, nxPML) = c13.block(1, 0, nzPML - 1, nxPML);
            c13ikad[n].row(nzPML - 1) = c13.row(nzPML - 1);
            c13ikmi[n].block(1, 0, nzPML - 1, nxPML) = c13.block(0, 0, nzPML - 1, nxPML);
            c13ikmi[n].row(0) = c13.row(0);

            c44ik[n] = c44;
            c44iadk[n].block(0, 0, nzPML, nxPML - 1) = c44.block(0, 1, nzPML, nxPML - 1);
            c44iadk[n].col(nxPML - 1) = c44.col(nxPML - 1);
            c44imik[n].block(0, 1, nzPML, nxPML - 1) = c44.block(0, 0, nzPML, nxPML - 1);
            c44imik[n].col(0) = c44.col(0);
            c44ikad[n].block(0, 0, nzPML - 1, nxPML) = c44.block(1, 0, nzPML - 1, nxPML);
            c44ikad[n].row(nzPML - 1) = c44.row(nzPML - 1);
            c44ikmi[n].block(1, 0, nzPML - 1, nxPML) = c44.block(0, 0, nzPML - 1, nxPML);
            c44ikmi[n].row(0) = c44.row(0);
        }
        else if (n == 1){
            rhoik[n] = rhoik[n].array() + 1.0;
            c11ik[n] = c11rho;
            c44ik[n] = c44rho;
        }
        else if (n == 2){
            c11iadk[n].block(0, 0, nzPML, nxPML - 1) = c11rho.block(0, 1, nzPML, nxPML - 1);
            c44iadk[n].block(0, 0, nzPML, nxPML - 1) = c44rho.block(0, 1, nzPML, nxPML - 1);    
            c13iadk[n].block(0, 0, nzPML, nxPML - 1) = c13rho.block(0, 1, nzPML, nxPML - 1);
        }
        else if (n == 3){
            c11imik[n].block(0, 1, nzPML, nxPML - 1) = c11rho.block(0, 0, nzPML, nxPML - 1);
            c44imik[n].block(0, 1, nzPML, nxPML - 1) = c44rho.block(0, 0, nzPML, nxPML - 1);
            c13imik[n].block(0, 1, nzPML, nxPML - 1) = c13rho.block(0, 0, nzPML, nxPML - 1);
        }
        else if (n == 4){
            c11ikad[n].block(0, 0, nzPML - 1, nxPML) = c11rho.block(1, 0, nzPML - 1, nxPML);
            c44ikad[n].block(0, 0, nzPML - 1, nxPML) = c44rho.block(1, 0, nzPML - 1, nxPML);
            c13ikad[n].block(0, 0, nzPML - 1, nxPML) = c13rho.block(1, 0, nzPML - 1, nxPML);
        }
        else if (n == 5){
            c11ikmi[n].block(1, 0, nzPML - 1, nxPML) = c11rho.block(0, 0, nzPML - 1, nxPML);
            c44ikmi[n].block(1, 0, nzPML - 1, nxPML) = c44rho.block(0, 0, nzPML - 1, nxPML);
            c13ikmi[n].block(1, 0, nzPML - 1, nxPML) = c13rho.block(0, 0, nzPML - 1, nxPML);
        }
        else if (n == 6){
            c11ik[n] = c11vp;
            c11imik[n].col(0) = c11vp.col(0);
            c13imik[n].col(0) = c13vp.col(0);
            c11iadk[n].col(nxPML - 1) = c11vp.col(nxPML - 1);
            c13iadk[n].col(nxPML - 1) = c13vp.col(nxPML - 1);
            c11ikmi[n].row(0) = c11vp.row(0);
            c13ikmi[n].row(0) = c13vp.row(0);
            c11ikad[n].row(nzPML - 1) = c11vp.row(nzPML - 1);
            c13ikad[n].row(nzPML - 1) = c13vp.row(nzPML - 1);
        }
        else if (n == 7){
            c11iadk[n].block(0, 0, nzPML, nxPML - 1) = c11vp.block(0, 1, nzPML, nxPML - 1);
            c13iadk[n].block(0, 0, nzPML, nxPML - 1) = c13vp.block(0, 1, nzPML, nxPML - 1);
        }
        else if (n == 8){
            c11imik[n].block(0, 1, nzPML, nxPML - 1) = c11vp.block(0, 0, nzPML, nxPML - 1);
            c13imik[n].block(0, 1, nzPML, nxPML - 1) = c13vp.block(0, 0, nzPML, nxPML - 1);
        }
        else if (n == 9){
            c11ikad[n].block(0, 0, nzPML - 1, nxPML) = c11vp.block(1, 0, nzPML - 1, nxPML);
            c13ikad[n].block(0, 0, nzPML - 1, nxPML) = c13vp.block(1, 0, nzPML - 1, nxPML);
        }
        else if (n == 10){
            c11ikmi[n].block(1, 0, nzPML - 1, nxPML) = c11vp.block(0, 0, nzPML - 1, nxPML);
            c13ikmi[n].block(1, 0, nzPML - 1, nxPML) = c13vp.block(0, 0, nzPML - 1, nxPML);
        }
        else if (n == 11){
            c11ik[n] = c11Qp;
            c11imik[n].col(0) = c11Qp.col(0);
            c13imik[n].col(0) = c13Qp.col(0);
            c11iadk[n].col(nxPML - 1) = c11Qp.col(nxPML - 1);
            c13iadk[n].col(nxPML - 1) = c13Qp.col(nxPML - 1);
            c11ikmi[n].row(0) = c11Qp.row(0);
            c13ikmi[n].row(0) = c13Qp.row(0);
            c11ikad[n].row(nzPML - 1) = c11Qp.row(nzPML - 1);
            c13ikad[n].row(nzPML - 1) = c13Qp.row(nzPML - 1);  
        }
        else if (n == 12){
            c11iadk[n].block(0, 0, nzPML, nxPML - 1) = c11Qp.block(0, 1, nzPML, nxPML - 1);
            c13iadk[n].block(0, 0, nzPML, nxPML - 1) = c13Qp.block(0, 1, nzPML, nxPML - 1);
        }                                                 
        else if (n == 13){                                
            c11imik[n].block(0, 1, nzPML, nxPML - 1) = c11Qp.block(0, 0, nzPML, nxPML - 1);
            c13imik[n].block(0, 1, nzPML, nxPML - 1) = c13Qp.block(0, 0, nzPML, nxPML - 1);
        }                                                 
        else if (n == 14){                                
            c11ikad[n].block(0, 0, nzPML - 1, nxPML) = c11Qp.block(1, 0, nzPML - 1, nxPML);
            c13ikad[n].block(0, 0, nzPML - 1, nxPML) = c13Qp.block(1, 0, nzPML - 1, nxPML);
        }
        else if (n == 15){
            c11ikmi[n].block(1, 0, nzPML - 1, nxPML) = c13Qp.block(0, 0, nzPML - 1, nxPML);
        }
        else if (n == 16){
            c44ik[n] = c44vs;
            c44imik[n].col(0) = c44vs.col(0);
            c13imik[n].col(0) = c13vs.col(0);
            c44iadk[n].col(nxPML - 1) = c44vs.col(nxPML - 1);
            c13iadk[n].col(nxPML - 1) = c13vs.col(nxPML - 1);
            c44ikmi[n].row(0) = c44vs.row(0);
            c13ikmi[n].row(0) = c13vs.row(0);
            c44ikad[n].row(nzPML - 1) = c44vs.row(nzPML - 1);
            c13ikad[n].row(nzPML - 1) = c13vs.row(nzPML - 1);
        }
        else if (n == 17){
            c44iadk[n].block(0, 0, nzPML, nxPML - 1) = c44vs.block(0, 1, nzPML, nxPML - 1);
            c13iadk[n].block(0, 0, nzPML, nxPML - 1) = c13vs.block(0, 1, nzPML, nxPML - 1);
        }                                                  
        else if (n == 18){                                 
            c44imik[n].block(0, 1, nzPML, nxPML - 1) = c44vs.block(0, 0, nzPML, nxPML - 1);
            c13imik[n].block(0, 1, nzPML, nxPML - 1) = c13vs.block(0, 0, nzPML, nxPML - 1);
        }                                                  
        else if (n == 19){                                 
            c44ikad[n].block(0, 0, nzPML - 1, nxPML) = c44vs.block(1, 0, nzPML - 1, nxPML);
            c13ikad[n].block(0, 0, nzPML - 1, nxPML) = c13vs.block(1, 0, nzPML - 1, nxPML);
        }                                                  
        else if (n == 20){                                 
            c44ikmi[n].block(1, 0, nzPML - 1, nxPML) = c44vs.block(0, 0, nzPML - 1, nxPML);
            c13ikmi[n].block(1, 0, nzPML - 1, nxPML) = c13vs.block(0, 0, nzPML - 1, nxPML);
        }
        else if (n == 21){
            c44ik[n] = c44Qs;
            c44imik[n].col(0) = c44Qs.col(0);
            c13imik[n].col(0) = c13Qs.col(0);
            c44iadk[n].col(nxPML - 1) = c44Qs.col(nxPML - 1);
            c13iadk[n].col(nxPML - 1) = c13Qs.col(nxPML - 1);
            c44ikmi[n].row(0) = c44Qs.row(0);
            c13ikmi[n].row(0) = c13Qs.row(0);
            c11ikad[n].row(nzPML - 1) = c44Qs.row(nzPML - 1);
            c13ikad[n].row(nzPML - 1) = c13Qs.row(nzPML - 1);
        }   
        else if (n == 22){
            c44iadk[n].block(0, 0, nzPML, nxPML - 1) = c44Qs.block(0, 1, nzPML, nxPML - 1);
            c13iadk[n].block(0, 0, nzPML, nxPML - 1) = c13Qs.block(0, 1, nzPML, nxPML - 1);
        }                                                  
        else if (n == 23){                                 
            c44imik[n].block(0, 1, nzPML, nxPML - 1) = c44Qs.block(0, 0, nzPML, nxPML - 1);
            c13imik[n].block(0, 1, nzPML, nxPML - 1) = c13Qs.block(0, 0, nzPML, nxPML - 1);
        }                                                  
        else if (n == 24){                                 
            c44ikad[n].block(0, 0, nzPML - 1, nxPML) = c44Qs.block(1, 0, nzPML - 1, nxPML);
            c13ikad[n].block(0, 0, nzPML - 1, nxPML) = c13Qs.block(1, 0, nzPML - 1, nxPML);
        }
        else if (n == 25){
            c44ikmi[n].block(1, 0, nzPML - 1, nxPML) = c44Qs.block(0, 0, nzPML - 1, nxPML);
            c13ikmi[n].block(1, 0, nzPML - 1, nxPML) = c13Qs.block(0, 0, nzPML - 1, nxPML);
        }
 
    }

    PML1.resize(PML1.size(), 1); PML2.resize(PML2.size(), 1); PML5.resize(PML5.size(), 1);
    PML6.resize(PML6.size(), 1); PML9.resize(PML9.size(), 1);
    int NNPML = nzPML * nxPML;

//Eigen::initParallel();
//#pragma omp parallel num_threads(2) firstprivate(c11, c13, c44, c11rho, c11vp, c11Qp, c44rho, c44vs, c44Qs, c13rho, c13vp, c13Qp, c13vs, c13Qs)                     
//#pragma omp for

    for (int n = 0; n < 26; n++){
        Eigen::MatrixXf rhoik_use = rhoik[n];
        rhoik_use.resize(NNPML, 1);
        Eigen::MatrixXcf c11ik_use = c11ik[n];
        c11ik_use.resize(NNPML, 1);
        Eigen::MatrixXcf c11iadk_use = c11iadk[n];
        c11iadk_use.resize(NNPML, 1);
        Eigen::MatrixXcf c11imik_use = c11imik[n]; 
        c11imik_use.resize(NNPML, 1);
        Eigen::MatrixXcf c11ikad_use = c11ikad[n];
        c11ikad_use.resize(NNPML, 1);
        Eigen::MatrixXcf c11ikmi_use = c11ikmi[n];
        c11ikmi_use.resize(NNPML, 1);

        Eigen::MatrixXcf c13iadk_use = c13iadk[n];
        c13iadk_use.resize(NNPML, 1);
        Eigen::MatrixXcf c13imik_use = c13imik[n]; 
        c13imik_use.resize(NNPML, 1);
        Eigen::MatrixXcf c13ikad_use = c13ikad[n];
        c13ikad_use.resize(NNPML, 1);
        Eigen::MatrixXcf c13ikmi_use = c13ikmi[n];
        c13ikmi_use.resize(NNPML, 1);

        Eigen::MatrixXcf c44ik_use = c44ik[n];
        c44ik_use.resize(NNPML, 1);
        Eigen::MatrixXcf c44iadk_use = c44iadk[n];
        c44iadk_use.resize(NNPML, 1);    
        Eigen::MatrixXcf c44imik_use = c44imik[n]; 
        c44imik_use.resize(NNPML, 1);   
        Eigen::MatrixXcf c44ikad_use = c44ikad[n];
        c44ikad_use.resize(NNPML, 1);   
        Eigen::MatrixXcf c44ikmi_use = c44ikmi[n];
        c44ikmi_use.resize(NNPML, 1);

        Eigen::MatrixXi all_ind(nzPML, nxPML);
        for (int i = 0; i < nxPML; i++)
            all_ind.col(i) = Eigen::RowVectorXf::LinSpaced(nzPML, i * nzPML, (i + 1) * nzPML - 1).cast<int>();

        Eigen::MatrixXi kk_ne_1 = all_ind.block(1, 0, nzPML - 1, nxPML);
        kk_ne_1.resize(kk_ne_1.size(), 1);
        Eigen::ArrayXi k_ne_1 = kk_ne_1;
        kk_ne_1.resize(0, 0);
        Eigen::MatrixXi kk_ne_nz = all_ind.block(0, 0, nzPML - 1, nxPML);
        kk_ne_nz.resize(kk_ne_nz.size(), 1);
        Eigen::ArrayXi k_ne_nz = kk_ne_nz;
        kk_ne_nz.resize(0, 0);

        Eigen::MatrixXi ii_ne_1 = all_ind.block(0, 1, nzPML, nxPML - 1);
        ii_ne_1.resize(ii_ne_1.size(), 1);
        Eigen::ArrayXi i_ne_1 = ii_ne_1;
        ii_ne_1.resize(0, 0);
        Eigen::MatrixXi ii_ne_nx = all_ind.block(0, 0, nzPML, nxPML - 1);
        ii_ne_nx.resize(ii_ne_nx.size(), 1);
        Eigen::ArrayXi i_ne_nx = ii_ne_nx;
        ii_ne_nx.resize(0, 0);

        Eigen::MatrixXi kk_ne_1_ii_ne_1 = all_ind.block(1, 1, nzPML - 1, nxPML - 1);
        kk_ne_1_ii_ne_1.resize(kk_ne_1_ii_ne_1.size(), 1);
        Eigen::ArrayXi k_ne_1_i_ne_1 = kk_ne_1_ii_ne_1;
        kk_ne_1_ii_ne_1.resize(0, 0);

        Eigen::MatrixXi kk_ne_nz_ii_ne_1 = all_ind.block(0, 1, nzPML - 1, nxPML - 1);
        kk_ne_nz_ii_ne_1.resize(kk_ne_nz_ii_ne_1.size(), 1);
        Eigen::ArrayXi k_ne_nz_i_ne_1 = kk_ne_nz_ii_ne_1;
        kk_ne_nz_ii_ne_1.resize(0, 0);

        Eigen::MatrixXi kk_ne_1_ii_ne_nx = all_ind.block(1, 0, nzPML - 1, nxPML - 1);
        kk_ne_1_ii_ne_nx.resize(kk_ne_1_ii_ne_nx.size(), 1);
        Eigen::ArrayXi k_ne_1_i_ne_nx = kk_ne_1_ii_ne_nx;
        kk_ne_1_ii_ne_nx.resize(0, 0);

        Eigen::MatrixXi kk_ne_nz_ii_ne_nx = all_ind.block(0, 0, nzPML - 1, nxPML - 1);
        kk_ne_nz_ii_ne_nx.resize(kk_ne_nz_ii_ne_nx.size(), 1);
        Eigen::ArrayXi k_ne_nz_i_ne_nx = kk_ne_nz_ii_ne_nx;
        kk_ne_nz_ii_ne_nx.resize(0, 0);

        Eigen::MatrixXcf A5_0 = std::pow(omega, 2) * rhoik_use.array() - \
                              del * (PML1.array() * (c11ik_use.array() + c11iadk_use.array()) / 2.0 + \
                              PML2.array() * (c11imik_use.array() + c11ik_use.array()) / 2.0 + \
                              PML5.array() * (c44ik_use.array() + c44ikad_use.array()) / 2.0 + \
                              PML6.array() * (c44ikmi_use.array() + c44ik_use.array()) / 2.0); 
        Eigen::MatrixXcf A5_1 = std::pow(omega, 2) * rhoik_use.array() - \
                              del * (PML5.array() * (c11ik_use.array() + c11ikad_use.array()) / 2.0 + \
                              PML6.array() * (c11ikmi_use.array() + c11ik_use.array()) / 2.0 + \
                              PML1.array() * (c44ik_use.array() + c44iadk_use.array()) / 2.0 + \
                              PML2.array() * (c44imik_use.array() + c44ik_use.array()) / 2.0);
        MADI[n](4, Eigen::seq(0, 2 * NNPML - 1, 2)) = A5_0.transpose();
        MADI[n](4, Eigen::seq(1, 2 * NNPML - 1, 2)) = A5_1.transpose();

        Eigen::MatrixXcf A6_0 = del * PML5.array() * (c44ik_use.array() + c44ikad_use.array()) / 2.0;
        Eigen::MatrixXcf A6_1 = del * PML5.array() * (c11ik_use.array() + c11ikad_use.array()) / 2.0;
        MADI[n](5, Eigen::seq(2 * k_ne_nz(0), 2 * k_ne_nz(Eigen::last), 2)) = A6_0(Eigen::seq(k_ne_nz(0), k_ne_nz(Eigen::last)), 0).transpose();
        MADI[n](5, Eigen::seq(2 * k_ne_nz(0) + 1, 2 * k_ne_nz(Eigen::last), 2)) = A6_1(Eigen::seq(k_ne_nz(0), k_ne_nz(Eigen::last)), 0).transpose();
        
        Eigen::MatrixXcf A7_0 = -del * PML9.array() * (c13iadk_use.array() + c44ikmi_use.array()) / 4.0;
        Eigen::MatrixXcf A7_1 = -del * PML9.array() * (c13ikmi_use.array() + c44iadk_use.array()) / 4.0;
        MADI[n](6, Eigen::seq(2 * k_ne_1_i_ne_nx(0), 2 * k_ne_1_i_ne_nx(Eigen::last), 2)) = A7_0(Eigen::seq(k_ne_1_i_ne_nx(0), k_ne_1_i_ne_nx(Eigen::last)), 0).transpose();
        MADI[n](6, Eigen::seq(2 * k_ne_1_i_ne_nx(0) + 1, 2 * k_ne_1_i_ne_nx(Eigen::last), 2)) = A7_1(Eigen::seq(k_ne_1_i_ne_nx(0), k_ne_1_i_ne_nx(Eigen::last)), 0).transpose();

        Eigen::MatrixXcf A8_0 = del * PML1.array() * (c11ik_use.array() + c11iadk_use.array()) / 2.0;
        Eigen::MatrixXcf A8_1 = del * PML1.array() * (c44ik_use.array() + c44iadk_use.array()) / 2.0;
        MADI[n](7, Eigen::seq(2 * i_ne_nx(0), 2 * i_ne_nx(Eigen::last), 2)) = A8_0(Eigen::seq(i_ne_nx(0), i_ne_nx(Eigen::last)), 0).transpose();
        MADI[n](7, Eigen::seq(2 * i_ne_nx(0) + 1, 2 * i_ne_nx(Eigen::last), 2)) = A8_1(Eigen::seq(i_ne_nx(0), i_ne_nx(Eigen::last)), 0).transpose();

        Eigen::MatrixXcf A9_0 = del * PML9.array() * (c13iadk_use.array() + c44ikad_use.array()) / 4.0;
        Eigen::MatrixXcf A9_1 = del * PML9.array() * (c13ikad_use.array() + c44iadk_use.array()) / 4.0;
        MADI[n](8, Eigen::seq(2 * k_ne_nz_i_ne_nx(0), 2 * k_ne_nz_i_ne_nx(Eigen::last), 2)) = A9_0(Eigen::seq(k_ne_nz_i_ne_nx(0), k_ne_nz_i_ne_nx(Eigen::last)), 0).transpose();
        MADI[n](8, Eigen::seq(2 * k_ne_nz_i_ne_nx(0) + 1, 2 * k_ne_nz_i_ne_nx(Eigen::last), 2)) = A9_1(Eigen::seq(k_ne_nz_i_ne_nx(0), k_ne_nz_i_ne_nx(Eigen::last)), 0).transpose();

        Eigen::MatrixXcf A1_0 = del * PML9.array() * (c13imik_use.array() + c44ikmi_use.array()) / 4.0;
        Eigen::MatrixXcf A1_1 = del * PML9.array() * (c13ikmi_use.array() + c44imik_use.array()) / 4.0;
        MADI[n](0, Eigen::seq(2 * k_ne_1_i_ne_1(0), 2 * k_ne_1_i_ne_1(Eigen::last), 2)) = A1_0(Eigen::seq(k_ne_1_i_ne_1(0), k_ne_1_i_ne_1(Eigen::last)), 0).transpose();
        MADI[n](0, Eigen::seq(2 * k_ne_1_i_ne_1(0) + 1, 2 * k_ne_1_i_ne_1(Eigen::last), 2)) = A1_1(Eigen::seq(k_ne_1_i_ne_1(0), k_ne_1_i_ne_1(Eigen::last)), 0).transpose();

        Eigen::MatrixXcf A2_0 = del * PML2.array() * (c11imik_use.array() + c11ik_use.array()) / 2.0;
        Eigen::MatrixXcf A2_1 = del * PML2.array() * (c44imik_use.array() + c44ik_use.array()) / 2.0;
        MADI[n](1, Eigen::seq(2 * i_ne_1(0), 2 * i_ne_1(Eigen::last), 2)) = A2_0(Eigen::seq(i_ne_1(0), i_ne_1(Eigen::last)), 0).transpose();
        MADI[n](1, Eigen::seq(2 * i_ne_1(0) + 1, 2 * i_ne_1(Eigen::last), 2)) = A2_1(Eigen::seq(i_ne_1(0), i_ne_1(Eigen::last)), 0).transpose();

        Eigen::MatrixXcf A3_0 = -del * PML9.array() * (c13imik_use.array() + c44ikad_use.array()) / 4.0;
        Eigen::MatrixXcf A3_1 = -del * PML9.array() * (c13ikad_use.array() + c44imik_use.array()) / 4.0;
        MADI[n](2, Eigen::seq(2 * k_ne_nz_i_ne_1(0), 2 * k_ne_nz_i_ne_1(Eigen::last), 2)) = A3_0(Eigen::seq(k_ne_nz_i_ne_1(0), k_ne_nz_i_ne_1(Eigen::last)), 0).transpose();
        MADI[n](2, Eigen::seq(2 * k_ne_nz_i_ne_1(0) + 1, 2 * k_ne_nz_i_ne_1(Eigen::last), 2)) = A3_1(Eigen::seq(k_ne_nz_i_ne_1(0), k_ne_nz_i_ne_1(Eigen::last)), 0).transpose();

        Eigen::MatrixXcf A4_0 = del * PML6.array() * (c44ikmi_use.array() + c44ik_use.array()) / 2.0;
        Eigen::MatrixXcf A4_1 = del * PML6.array() * (c11ikmi_use.array() + c11ik_use.array()) / 2.0;
        MADI[n](3, Eigen::seq(2 * k_ne_1(0), 2 * k_ne_1(Eigen::last), 2)) = A4_0(Eigen::seq(k_ne_1(0), k_ne_1(Eigen::last)), 0).transpose();
        MADI[n](3, Eigen::seq(2 * k_ne_1(0) + 1, 2 * k_ne_1(Eigen::last), 2)) = A4_1(Eigen::seq(k_ne_1(0), k_ne_1(Eigen::last)), 0).transpose();
        
        if (n > 0){
            int var = std::floor((n - 1) / 5);
            MADI[n] = MADI[n].array() * scale[var];
        }

        rhoik_use.resize(0, 0); c11ik_use.resize(0, 0); c11iadk_use.resize(0, 0);        
        c11imik_use.resize(0, 0); c11ikad_use.resize(0, 0); c11ikmi_use.resize(0, 0);
        c13iadk_use.resize(0, 0); c13imik_use.resize(0, 0); 
        c13ikad_use.resize(0, 0); c13ikmi_use.resize(0, 0);
        c44ik_use.resize(0, 0); c44iadk_use.resize(0, 0);
        c44imik_use.resize(0, 0); c44ikad_use.resize(0, 0); c44ikmi_use.resize(0, 0);
        all_ind.resize(0, 0);
        k_ne_1.resize(0); k_ne_nz.resize(0); i_ne_1.resize(0); i_ne_nx.resize(0);
        k_ne_1_i_ne_1.resize(0); k_ne_nz_i_ne_1.resize(0);
        k_ne_1_i_ne_nx.resize(0); k_ne_nz_i_ne_nx.resize(0);
        A5_0.resize(0, 0); A5_1.resize(0, 0);
        A1_0.resize(0, 0); A1_1.resize(0, 0);
        A2_0.resize(0, 0); A2_1.resize(0, 0);
        A3_0.resize(0, 0); A3_1.resize(0, 0);
        A4_0.resize(0, 0); A4_1.resize(0, 0);
        A6_0.resize(0, 0); A6_1.resize(0, 0);
        A7_0.resize(0, 0); A7_1.resize(0, 0);
        A8_0.resize(0, 0); A8_1.resize(0, 0);
        A9_0.resize(0, 0); A9_1.resize(0, 0);
    }


    ABCx0.resize(0); ABCx1.resize(0); ABCz0.resize(0); ABCz1.resize(0); 
    rho.resize(0, 0); SSp.resize(0, 0); Qp_inv.resize(0, 0); SSs.resize(0, 0);
    vp.resize(0, 0); vs.resize(0, 0);
    vp_complex.resize(0, 0); vp_complex.resize(0, 0);
    c11.resize(0, 0); c13.resize(0, 0); c44.resize(0, 0);
    c11rho.resize(0, 0); c11vp.resize(0, 0); c11Qp.resize(0, 0);
    c44rho.resize(0, 0); c44vs.resize(0, 0); c44Qs.resize(0, 0);
    c13rho.resize(0, 0); c13vp.resize(0, 0); c13Qp.resize(0, 0);
    c13vs.resize(0, 0); c13Qs.resize(0, 0);

    rhoik.clear(); rhoik.shrink_to_fit();
    c11ik.clear(); c11ik.shrink_to_fit(); c11iadk.clear(); c11iadk.shrink_to_fit();
    c11imik.clear(); c11imik.shrink_to_fit(); c11ikad.clear(); c11ikad.shrink_to_fit();
    c11ikmi.clear(); c11ikmi.shrink_to_fit();
    c13iadk.clear(); c13iadk.shrink_to_fit(); c13imik.clear(); c13imik.shrink_to_fit(); 
    c13ikad.clear(); c13ikad.shrink_to_fit(); c13ikmi.clear(); c13ikmi.shrink_to_fit();
    c44ik.clear(); c44ik.shrink_to_fit(); c44iadk.clear(); c44iadk.shrink_to_fit();
    c44imik.clear(); c44imik.shrink_to_fit(); c44ikad.clear(); c44ikad.shrink_to_fit();
    c44ikmi.clear(); c44ikmi.shrink_to_fit();

    PML1.resize(0, 0); PML2.resize(0, 0); PML5.resize(0, 0);
    PML6.resize(0, 0); PML9.resize(0, 0);

    return 1;
}

