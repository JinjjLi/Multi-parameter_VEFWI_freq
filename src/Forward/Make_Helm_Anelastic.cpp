#include "../General/includefile.h"
#include "Make_Helm_Anelastic.h"
#include "MakePML.h"
#include "PML_pad.h"

bool Make_Helm_Anelastic(Eigen::MatrixXcf& MADI, Eigen::MatrixXf& model_t, int nz, int nx, float omega, float omega0, int dz, int PML_thick){
    //using namespace std::complex_literals;
    float del = 1.0 / (dz * dz);
    int nxPML = nx + 2 * PML_thick; int nzPML = nz + 2 * PML_thick;
    Eigen::RowVectorXf ABCx0, ABCx1, ABCz0, ABCz1;
    Eigen::MatrixXf model = model_t;
    std::tie(ABCx0, ABCx1, ABCz0, ABCz1) = MakePML(nz, nx, PML_thick);
    model.resize(nz * nx, 5);
    Eigen::MatrixXf rho, SSp, Qp_inv, SSs, Qs_inv;
    rho = model.col(0); rho.resize(nz, nx);
    SSp = model.col(1); SSp.resize(nz, nx);
    Qp_inv = model.col(2); Qp_inv.resize(nz, nx);
    SSs = model.col(3); SSs.resize(nz, nx);
    Qs_inv = model.col(4); Qs_inv.resize(nz, nx);
    
    //Eigen::MatrixXcf MADI(9, 2 * nzPML * nxPML);
    //MADI.setZero();

    PML_pad(rho, PML_thick); PML_pad(SSp, PML_thick); PML_pad(Qp_inv, PML_thick);
    PML_pad(SSs, PML_thick); PML_pad(Qs_inv, PML_thick);

    Eigen::MatrixXcf vp = SSp.cast<std::complex<float>>().array().inverse().sqrt();
    Eigen::MatrixXcf vs = SSs.cast<std::complex<float>>().array().inverse().sqrt();
    Eigen::MatrixXcf temp = Eigen::MatrixXcf(Qp_inv.rows(), Qp_inv.cols());
    temp.setZero();
    temp.array().imag() = Qp_inv.array() * 0.5;
    temp.array().real() = 1.0 + Qp_inv.array() * ((1 / pi) * log(omega/omega0));
    vp = vp.array() * temp.array();
    temp.array().imag() = Qs_inv.array() * 0.5;
    temp.array().real() = 1.0 + Qs_inv.array() * ((1 / pi) * log(omega/omega0));
    vs = vs.array() * temp.array();
    Eigen::MatrixXcf c11 = vp.array().pow(2) * rho.array();
    Eigen::MatrixXcf c44 = vs.array().pow(2) * rho.array();
    Eigen::MatrixXcf c13 = c11.array() - c44.array() * 2;
    temp.resize(0, 0);

//Eigen::initParallel();
//int nthreads = Eigen::nbThreads();
//#pragma omp parallel firstprivate(vp, vs, rho, Qp_inv, Qs_inv, c11, c44, c13)
//#pragma omp for
    for (int i = 0; i < nxPML; i++){
        for (int k = 0; k < nzPML; k++){
            int ji = 2 * ((nzPML * i) + k);
            int imi = i - 1; int kmi = k - 1;
            int iad = i + 1; int kad = k + 1;
            if (imi < 0)
                imi = 0;
            if (kmi < 0)
                kmi = 0;
            if (iad > nxPML - 1)
                iad = nxPML - 1;
            if (kad > nzPML - 1)
                kad = nzPML - 1;
            std::complex<float> PML1, PML2, PML5, PML6, PML9;   
            std::complex<float> tempcomp(0, omega);
            PML1 = (tempcomp + ABCx0(i)) * (tempcomp + ABCx1(i));
            PML1 = -1 * omega * omega / PML1;
            PML2 = (tempcomp + ABCx0(i)) * (tempcomp + ABCx1(imi));
            PML2 = -1 * omega * omega / PML2;
            PML5 = (tempcomp + ABCz0(k)) * (tempcomp + ABCz1(k));
            PML5 = -1 * omega * omega / PML5;
            PML6 = (tempcomp + ABCz0(k)) * (tempcomp + ABCz1(kmi));
            PML6 = -1 * omega * omega / PML6;
            PML9 = (tempcomp + ABCx0(i)) * (tempcomp + ABCz0(k));
            PML9 = -1 * omega * omega / PML9;

            tempcomp = (2, 2);
            std::complex<float> A5 = omega * omega * rho(k, i) - del \
                 * (PML1 * (c11(k, i) + c11(k, iad)) / tempcomp \
                 + PML2 * (c11(k, imi) + c11(k, i)) / tempcomp \
                 + PML5 * (c44(k, i) + c44(kad, i)) / tempcomp \
                 + PML6 * (c44(kmi, i) + c44(k, i)) / tempcomp);
            MADI(4, ji) = A5;
            A5 = omega * omega * rho(k, i) - del \
                 * (PML5 * (c11(k, i) + c11(kad, i)) / tempcomp \
                 + PML6 * (c11(kmi, i) + c11(k, i)) / tempcomp \
                 + PML1 * (c44(k, i) + c44(k, iad)) / tempcomp \
                 + PML2 * (c44(k, imi) + c44(k, i)) / tempcomp);
            MADI(4, ji + 1) = A5;

            if(k != nzPML - 1){
                tempcomp = (2, 2);
                std::complex<float> A6 = del * PML5 * (c44(k, i) + c44(k + 1, i)) / tempcomp;
                MADI(5, ji) = A6;
                A6 = del * PML5 * (c11(k, i) + c11(k + 1, i)) / tempcomp;
                MADI(5, ji + 1) = A6;
            }
            if(i != nxPML - 1 && k != 0){
                tempcomp = (4, 4);
                std::complex<float> A7 = -del * PML9 * (c13(k, i + 1) + c44(k - 1, i)) / tempcomp;
                MADI(6, ji) = A7;
                A7 = -del * PML9 * (c13(k - 1, i) + c44(k, i + 1)) / tempcomp;
                MADI(6, ji + 1) = A7;
            }

            if(i != nxPML - 1){
                tempcomp = (2, 2);
                std::complex<float> A8 = del * PML1 * (c11(k, i) + c11(k, i + 1)) / tempcomp;
                MADI(7, ji) = A8;
                A8 = del * PML1 * (c44(k, i) + c44(k, i + 1)) / tempcomp;
                MADI(7, ji + 1) = A8;
            }
            if(i != nxPML - 1 && k != nzPML - 1){
                tempcomp = (4, 4);
                std::complex<float> A9 = del * PML9 * (c13(k, i + 1) + c44(k + 1, i)) / tempcomp;
                MADI(8, ji) = A9;
                A9 = del * PML9 * (c13(k + 1, i) + c44(k, i + 1)) / tempcomp;
                MADI(8, ji + 1) = A9;        
            }
            if(i != 0 && k != 0){
                tempcomp = (4, 4);
                std::complex<float> A1 = del * PML9 * (c13(k, i - 1) + c44(k - 1, i)) / tempcomp;
                MADI(0, ji) = A1;
                A1 = del * PML9 * (c13(k - 1, i) + c44(k, i - 1)) / tempcomp;
                MADI(0, ji + 1) = A1;
            }
            if(i != 0){
                tempcomp = (2, 2);
                std::complex<float> A2 = del * PML2 * (c11(k, i - 1) + c11(k, i)) / tempcomp;
                MADI(1, ji) = A2;
                A2 = del * PML2 * (c44(k, i - 1) + c44(k, i)) / tempcomp;
                MADI(1, ji + 1) = A2;
            }
            if(i != 0 && k != nzPML - 1){
                tempcomp = (4, 4);
                std::complex<float> A3 = -del * PML9 * (c13(k, i - 1) + c44(k + 1, i)) / tempcomp;
                MADI(2, ji) = A3;
                A3 = -del * PML9 * (c13(k + 1, i) + c44(k, i - 1)) / tempcomp;
                MADI(2, ji + 1) = A3;
            }
            if(k != 0){
                tempcomp = (2, 2);
                std::complex<float> A4 = del * PML6 * (c44(k - 1, i) + c44(k, i)) / tempcomp;
                MADI(3, ji) = A4;
                A4 = del * PML6 * (c11(k - 1, i) + c11(k, i)) / tempcomp;
                MADI(3, ji + 1) = A4;
            }
        }
    }

    ABCx0.resize(0); ABCx1.resize(0); ABCz0.resize(0); ABCz1.resize(0);
    rho.resize(0, 0); SSp.resize(0, 0); Qp_inv.resize(0, 0); SSs.resize(0, 0); Qs_inv.resize(0, 0);
    vp.resize(0, 0); vs.resize(0, 0);
    temp.resize(0, 0);
    model.resize(0, 0);
    c11.resize(0, 0); c13.resize(0, 0); c44.resize(0, 0);

    return 1;
}


