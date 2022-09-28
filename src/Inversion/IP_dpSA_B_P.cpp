#include "../General/includefile.h"
#include "IP_dpSA_B_P.h"

Eigen::MatrixXf IP_dpSA_B_P(Eigen::SparseMatrix<std::complex<float>>& A, Eigen::SparseMatrix<std::complex<float>>& B, \
                            std::vector<Eigen::SparseMatrix<std::complex<float>>>& MADI, \
                            int nz, int nx, int PML_thick){

    int nzPML = nz + 2 * PML_thick;
    int nxPML = nx + 2 * PML_thick;
    int NPML = nzPML * nxPML;
    int k2 = -2 * nzPML;
    int k4 = -2; int k5 = 0; int k6 = 2;
    int k8 = 2 * nzPML;

    int k1mi = k2 - 3; int k1ad = k2 - 1;
    int k3mi = k2 + 1; int k3ad = k2 + 3;
    int k7mi = k8 - 3; int k7ad = k8 - 1;
    int k9mi = k8 + 1; int k9ad = k8 + 3;

    int tempnum = 0;
    tempnum = std::min(2 * NPML - 1, 2 * NPML - 1 - k1mi) - std::max(0, 0 - k1mi) + 1;
    Eigen::ArrayXi ind1mi = Eigen::ArrayXi::LinSpaced(tempnum, std::max(0, 0 - k1mi), \
                                                      std::min(2 * NPML - 1, 2 * NPML - 1 - k1mi));

    tempnum = std::min(2 * NPML - 1, 2 * NPML - 1 - k1ad) - std::max(0, 0 - k1ad) + 1;
    Eigen::ArrayXi ind1ad = Eigen::ArrayXi::LinSpaced(tempnum, std::max(0, 0 - k1ad), \
                                                      std::min(2 * NPML - 1, 2 * NPML - 1 - k1ad));

    tempnum = std::min(2 * NPML - 1, 2 * NPML - 1 - k2) - std::max(0, 0 - k2) + 1;
    Eigen::ArrayXi ind2 = Eigen::ArrayXi::LinSpaced(tempnum, std::max(0, 0 - k2), \
                                                      std::min(2 * NPML - 1, 2 * NPML - 1 - k2));

    tempnum = std::min(2 * NPML - 1, 2 * NPML - 1 - k3mi) - std::max(0, 0 - k3mi) + 1;
    Eigen::ArrayXi ind3mi = Eigen::ArrayXi::LinSpaced(tempnum, std::max(0, 0 - k3mi), \
                                                      std::min(2 * NPML - 1, 2 * NPML - 1 - k3mi));
    
    tempnum = std::min(2 * NPML - 1, 2 * NPML - 1 - k3ad) - std::max(0, 0 - k3ad) + 1;
    Eigen::ArrayXi ind3ad = Eigen::ArrayXi::LinSpaced(tempnum, std::max(0, 0 - k3ad), \
                                                      std::min(2 * NPML - 1, 2 * NPML - 1 - k3ad));
    
    tempnum = std::min(2 * NPML - 1, 2 * NPML - 1 - k4) - std::max(0, 0 - k4) + 1;
    Eigen::ArrayXi ind4 = Eigen::ArrayXi::LinSpaced(tempnum, std::max(0, 0 - k4), \
                                                    std::min(2 * NPML - 1, 2 * NPML - 1 - k4));

    tempnum = std::min(2 * NPML - 1, 2 * NPML - 1 - k5) - std::max(0, 0 - k5) + 1;
    Eigen::ArrayXi ind5 = Eigen::ArrayXi::LinSpaced(tempnum, std::max(0, 0 - k5), \
                                                      std::min(2 * NPML - 1, 2 * NPML - 1 - k5));

    tempnum = std::min(2 * NPML - 1, 2 * NPML - 1 - k6) - std::max(0, 0 - k6) + 1;
    Eigen::ArrayXi ind6 = Eigen::ArrayXi::LinSpaced(tempnum, std::max(0, 0 - k6), \
                                                      std::min(2 * NPML - 1, 2 * NPML - 1 - k6));

    tempnum = std::min(2 * NPML - 1, 2 * NPML - 1 - k7mi) - std::max(0, 0 - k7mi) + 1;
    Eigen::ArrayXi ind7mi = Eigen::ArrayXi::LinSpaced(tempnum, std::max(0, 0 - k7mi), \
                                                      std::min(2 * NPML - 1, 2 * NPML - 1 - k7mi));

    tempnum = std::min(2 * NPML - 1, 2 * NPML - 1 - k7ad) - std::max(0, 0 - k7ad) + 1;     
    Eigen::ArrayXi ind7ad = Eigen::ArrayXi::LinSpaced(tempnum, std::max(0, 0 - k7ad), \
                                                      std::min(2 * NPML - 1, 2 * NPML - 1 - k7ad));

    tempnum = std::min(2 * NPML - 1, 2 * NPML - 1 - k8) - std::max(0, 0 - k8) + 1;
    Eigen::ArrayXi ind8 = Eigen::ArrayXi::LinSpaced(tempnum, std::max(0, 0 - k8), \
                                                      std::min(2 * NPML - 1, 2 * NPML - 1 - k8));

    tempnum = std::min(2 * NPML - 1, 2 * NPML - 1 - k9mi) - std::max(0, 0 - k9mi) + 1;
    Eigen::ArrayXi ind9mi = Eigen::ArrayXi::LinSpaced(tempnum, std::max(0, 0 - k9mi), \
                                                      std::min(2 * NPML - 1, 2 * NPML - 1 - k9mi));

    tempnum = std::min(2 * NPML - 1, 2 * NPML - 1 - k9ad) - std::max(0, 0 - k9ad) + 1;
    Eigen::ArrayXi ind9ad = Eigen::ArrayXi::LinSpaced(tempnum, std::max(0, 0 - k9ad), \
                                                      std::min(2 * NPML - 1, 2 * NPML - 1 - k9ad));

    Eigen::MatrixXcd IP = Eigen::MatrixXd(MADI[0].cols(), 1);
    IP.setZero();

    //Eigen::MatrixXcd AA = Eigen::MatrixXcf(A).cast<std::complex<double>>();
    //Eigen::MatrixXcd BB = Eigen::MatrixXcf(B).cast<std::complex<double>>();
    Eigen::SparseMatrix<std::complex<double>> AA = A.cast<std::complex<double>>();
    Eigen::SparseMatrix<std::complex<double>> BB = B.cast<std::complex<double>>();
    Eigen::SparseMatrix<std::complex<double>> temp_MADI;
    Eigen::SparseMatrix<std::complex<double>> temp_MADI_tar(MADI[0].rows(), MADI[0].cols());

    Eigen::ArrayXi colind = Eigen::ArrayXi::LinSpaced(AA.cols(), 0, AA.cols() - 1);
    Eigen::ArrayXi colind2 = Eigen::ArrayXi::LinSpaced(MADI[0].cols(), 0, MADI[0].cols() - 1);
    Eigen::ArrayXi indtemp;

    Eigen::SparseMatrix<std::complex<double>> product1;
    Eigen::MatrixXcd productmid;
    Eigen::MatrixXcd product2;

    //std::vector<Eigen::Triplet<std::complex<double>>> triplet; 

    //row and cols
    Eigen::SparseMatrix<std::complex<double>> tmpA;
    Eigen::SparseMatrix<std::complex<double>> tmpB;

    temp_MADI = MADI[8].cast<std::complex<double>>();
    //triplet = set_triplet<std::complex<double>>(AA, ind1ad, colind);
    //tmpA.setFromTriplets(triplet.begin(), triplet.end());                                                                                                                                                
    tmpA = AA.block(ind1ad(0),  colind(0), ind1ad.size(), colind.size());
    indtemp = ind1ad + k1ad;
    tmpB = BB.block(indtemp(0), colind(0), indtemp.size(), colind.size());
    indtemp.setZero();
    product1 = tmpA.conjugate().cwiseProduct(tmpB);
    tmpA.setZero(); tmpB.setZero();
    productmid = product1 * Eigen::VectorXcd::Ones(product1.cols());
    product1.setZero();
    productmid(Eigen::seq(0, Eigen::last, 2), 0).setZero();
    //triplet = set_triplet<std::complex<double>>(temp_MADI, ind9mi, colind2);
    temp_MADI_tar = temp_MADI.block(ind9mi(0), colind2(0), ind9mi.size(), colind2.size());
    product2 = temp_MADI_tar.adjoint() * productmid;                      
    temp_MADI_tar.setZero(); productmid.setZero();
    IP += product2;
    product2.setZero();
    tmpA = AA.block(ind1mi(0),  colind(0), ind1mi.size(), colind.size());
    indtemp = ind1mi + k1mi;
    tmpB = BB.block(indtemp(0), colind(0), indtemp.size(), colind.size());
    indtemp.setZero();
    product1 = tmpA.conjugate().cwiseProduct(tmpB);
    tmpA.setZero(); tmpB.setZero();
    productmid = product1 * Eigen::VectorXcd::Ones(product1.cols());
    product1.setZero();
    productmid(Eigen::seq(1, Eigen::last, 2), 0).setZero();
    temp_MADI_tar = temp_MADI.block(ind9ad(0), colind2(0), ind9ad.size(), colind2.size());
    product2 = temp_MADI_tar.adjoint() * productmid;
    temp_MADI_tar.setZero(); productmid.setZero();
    IP += product2;
    product2.setZero(); temp_MADI.setZero();


    temp_MADI = MADI[7].cast<std::complex<double>>();
    tmpA = AA.block(ind2(0), colind(0), ind2.size(), colind.size());
    indtemp = ind2 + k2;
    tmpB = BB.block(indtemp(0), colind(0), indtemp.size(), colind.size());
    indtemp.setZero();
    product1 = tmpA.conjugate().cwiseProduct(tmpB);
    tmpA.setZero(); tmpB.setZero();
    productmid = product1 * Eigen::VectorXcd::Ones(product1.cols());
    product1.setZero();
    temp_MADI_tar = temp_MADI.block(ind8(0), colind2(0), ind8.size(), colind2.size());
    product2 = temp_MADI_tar.adjoint() * productmid;
    temp_MADI_tar.setZero(); productmid.setZero();
    IP += product2;
    product2.setZero(); temp_MADI.setZero();


    temp_MADI = MADI[6].cast<std::complex<double>>();
    tmpA = AA.block(ind3ad(0), colind(0), ind3ad.size(), colind.size());
    indtemp = ind3ad + k3ad;
    tmpB = BB.block(indtemp(0), colind(0), indtemp.size(), colind.size());
    indtemp.setZero();
    product1 = tmpA.conjugate().cwiseProduct(tmpB);
    tmpA.setZero(); tmpB.setZero();
    productmid = product1 * Eigen::VectorXcd::Ones(product1.cols());
    product1.setZero();
    productmid(Eigen::seq(0, Eigen::last, 2), 0).setZero();
    temp_MADI_tar = temp_MADI.block(ind7mi(0), colind2(0), ind7mi.size(), colind2.size());
    product2 = temp_MADI_tar.adjoint() * productmid;
    temp_MADI_tar.setZero(); productmid.setZero();
    IP += product2;
    product2.setZero();
    tmpA = AA.block(ind3mi(0), colind(0), ind3mi.size(), colind.size());
    indtemp = ind3mi + k3mi;
    tmpB = BB.block(indtemp(0), colind(0), indtemp.size(), colind.size());
    indtemp.setZero();                           
    product1 = tmpA.conjugate().cwiseProduct(tmpB);
    tmpA.setZero(); tmpB.setZero();
    productmid = product1 * Eigen::VectorXcd::Ones(product1.cols());
    product1.setZero();
    productmid(Eigen::seq(1, Eigen::last, 2), 0).setZero();
    temp_MADI_tar = temp_MADI.block(ind7ad(0), colind2(0), ind7ad.size(), colind2.size());
    product2 = temp_MADI_tar.adjoint() * productmid;
    temp_MADI_tar.setZero(); productmid.setZero();
    IP += product2;
    product2.setZero(); temp_MADI.setZero();


    temp_MADI = MADI[5].cast<std::complex<double>>();
    tmpA = AA.block(ind4(0), colind(0), ind4.size(), colind.size());
    indtemp = ind4 + k4;
    tmpB = BB.block(indtemp(0), colind(0), indtemp.size(), colind.size());
    indtemp.setZero();
    product1 = tmpA.conjugate().cwiseProduct(tmpB);
    tmpA.setZero(); tmpB.setZero();
    productmid = product1 * Eigen::VectorXcd::Ones(product1.cols());
    product1.setZero();
    temp_MADI_tar = temp_MADI.block(ind6(0), colind2(0), ind6.size(), colind2.size());
    product2 = temp_MADI_tar.adjoint() * productmid;
    temp_MADI_tar.setZero(); productmid.setZero();
    IP += product2;
    product2.setZero(); temp_MADI.setZero();

    temp_MADI = MADI[4].cast<std::complex<double>>();
    tmpA = AA.block(ind5(0), colind(0), ind5.size(), colind.size());
    indtemp = ind5 + k5;
    tmpB = BB.block(indtemp(0), colind(0), indtemp.size(), colind.size());
    indtemp.setZero();
    product1 = tmpA.conjugate().cwiseProduct(tmpB);
    tmpA.setZero(); tmpB.setZero();
    productmid = product1 * Eigen::VectorXcd::Ones(product1.cols());
    product1.setZero();
    temp_MADI_tar = temp_MADI.block(ind5(0), colind2(0), ind5.size(), colind2.size());
    product2 = temp_MADI_tar.adjoint() * productmid;
    temp_MADI_tar.setZero(); productmid.setZero();
    IP += product2;
    product2.setZero(); temp_MADI.setZero();


    temp_MADI = MADI[3].cast<std::complex<double>>();
    tmpA = AA.block(ind6(0), colind(0), ind6.size(), colind.size());
    indtemp = ind6 + k6;
    tmpB = BB.block(indtemp(0), colind(0), indtemp.size(), colind.size());
    indtemp.setZero();
    product1 = tmpA.conjugate().cwiseProduct(tmpB);
    tmpA.setZero(); tmpB.setZero();
    productmid = product1 * Eigen::VectorXcd::Ones(product1.cols());
    product1.setZero();
    temp_MADI_tar = temp_MADI.block(ind4(0), colind2(0), ind4.size(), colind2.size());
    product2 = temp_MADI_tar.adjoint() * productmid;   
    temp_MADI_tar.setZero(); productmid.setZero();
    IP += product2;
    product2.setZero(); temp_MADI.setZero();


    temp_MADI = MADI[2].cast<std::complex<double>>();
    tmpA = AA.block(ind7ad(0), colind(0), ind7ad.size(), colind.size());
    indtemp = ind7ad + k7ad;
    tmpB = BB.block(indtemp(0), colind(0), indtemp.size(), colind.size());
    indtemp.setZero();
    product1 = tmpA.conjugate().cwiseProduct(tmpB);
    tmpA.setZero(); tmpB.setZero();
    productmid = product1 * Eigen::VectorXcd::Ones(product1.cols());
    product1.setZero();
    productmid(Eigen::seq(1, Eigen::last, 2), 0).setZero();
    temp_MADI_tar = temp_MADI.block(ind3mi(0), colind2(0), ind3mi.size(), colind2.size());
    product2 = temp_MADI_tar.adjoint() * productmid;
    temp_MADI_tar.setZero(); productmid.setZero();
    IP += product2;
    product2.setZero();
    tmpA = AA.block(ind7mi(0), colind(0), ind7mi.size(), colind.size());
    indtemp = ind7mi + k7mi;
    tmpB = BB.block(indtemp(0), colind(0), indtemp.size(), colind.size());
    indtemp.setZero();
    product1 = tmpA.conjugate().cwiseProduct(tmpB);
    tmpA.setZero(); tmpB.setZero();
    productmid = product1 * Eigen::VectorXcd::Ones(product1.cols());
    product1.setZero();
    productmid(Eigen::seq(0, Eigen::last, 2), 0).setZero();
    temp_MADI_tar = temp_MADI.block(ind3ad(0), colind2(0), ind3ad.size(), colind2.size());
    product2 = temp_MADI_tar.adjoint() * productmid;
    temp_MADI_tar.setZero(); productmid.setZero();
    IP += product2;
    product2.setZero(); temp_MADI.setZero();


    temp_MADI = MADI[1].cast<std::complex<double>>();
    tmpA = AA.block(ind8(0), colind(0), ind8.size(), colind.size());
    indtemp = ind8 + k8;
    tmpB = BB.block(indtemp(0), colind(0), indtemp.size(), colind.size());
    indtemp.setZero();
    product1 = tmpA.conjugate().cwiseProduct(tmpB);
    tmpA.setZero(); tmpB.setZero();
    productmid = product1 * Eigen::VectorXcd::Ones(product1.cols());
    product1.setZero();
    temp_MADI_tar = temp_MADI.block(ind2(0), colind2(0), ind2.size(), colind2.size());
    product2 = temp_MADI_tar.adjoint() * productmid;
    temp_MADI_tar.setZero(); productmid.setZero();
    IP += product2;
    product2.setZero(); temp_MADI.setZero();

    temp_MADI = MADI[0].cast<std::complex<double>>();
    tmpA = AA.block(ind9ad(0), colind(0), ind9ad.size(), colind.size());
    indtemp = ind9ad + k9ad;                        
    tmpB = BB.block(indtemp(0), colind(0), indtemp.size(), colind.size());
    indtemp.setZero();
    product1 = tmpA.conjugate().cwiseProduct(tmpB);
    tmpA.setZero(); tmpB.setZero();
    productmid = product1 * Eigen::VectorXcd::Ones(product1.cols());
    product1.setZero();
    productmid(Eigen::seq(1, Eigen::last, 2), 0).setZero();
    temp_MADI_tar = temp_MADI.block(ind1mi(0), colind2(0), ind1mi.size(), colind2.size());
    product2 = temp_MADI_tar.adjoint() * productmid;
    temp_MADI_tar.setZero(); productmid.setZero();
    IP += product2;
    product2.setZero();
    tmpA = AA.block(ind9mi(0), colind(0), ind9mi.size(), colind.size());
    indtemp = ind9mi + k9mi;   
    tmpB = BB.block(indtemp(0), colind(0), indtemp.size(), colind.size());
    indtemp.setZero();
    product1 = tmpA.conjugate().cwiseProduct(tmpB);
    tmpA.setZero(); tmpB.setZero();
    productmid = product1 * Eigen::VectorXcd::Ones(product1.cols());
    product1.setZero();
    productmid(Eigen::seq(0, Eigen::last, 2), 0).setZero();
    temp_MADI_tar = temp_MADI.block(ind1ad(0), colind2(0), ind1ad.size(), colind2.size());
    product2 = temp_MADI_tar.adjoint() * productmid;
    temp_MADI_tar.setZero(); productmid.setZero();
    IP += product2;
    product2.setZero(); temp_MADI.setZero();

    Eigen::MatrixXf IP_out = Eigen::MatrixXf(IP.cast<std::complex<float>>().real());

    ind1mi.resize(0); ind1ad.resize(0); ind2.resize(0);
    ind3mi.resize(0); ind3ad.resize(0); ind4.resize(0);
    ind5.resize(0); ind6.resize(0);
    ind7mi.resize(0); ind7ad.resize(0); ind8.resize(0);
    ind9mi.resize(0); ind9ad.resize(0);
    colind.resize(0); colind2.resize(0);
    indtemp.resize(0);
    AA.resize(0, 0); BB.resize(0, 0);
    AA.data().squeeze(); BB.data().squeeze();
    temp_MADI.resize(0, 0); temp_MADI.data().squeeze();
    temp_MADI_tar.resize(0, 0); temp_MADI_tar.data().squeeze();
    product1.resize(0, 0); product1.data().squeeze();
    product2.resize(0, 0); productmid.resize(0, 0);
    IP.resize(0, 0);

    return IP_out;
}


