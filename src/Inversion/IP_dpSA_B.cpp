#include "../General/includefile.h"
#include "../General/mat_indexing.h"
#include "IP_dpSA_B.h"

Eigen::MatrixXf IP_dpSA_B(Eigen::SparseMatrix<std::complex<float>>& A, Eigen::SparseMatrix<std::complex<float>>& B, std::vector<Eigen::MatrixXcf>& MADI, \
                           int nz, int nx, int PML_thick){
    int nzPML = nz + 2 * PML_thick; int nxPML = nx + 2 * PML_thick;
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

    Eigen::MatrixXf prod = Eigen::MatrixXf(2 * NPML, 5);
    prod.setZero();

    Eigen::ArrayXi colind = Eigen::ArrayXi::LinSpaced(A.cols(), 0, A.cols() - 1);
    Eigen::ArrayXi tmpind;
    Eigen::SparseMatrix<std::complex<float>> tmpA;
    Eigen::SparseMatrix<std::complex<float>> tmpB;
    Eigen::SparseMatrix<std::complex<float>> product;

    tmpind = ind1ad + k1ad;
    tmpA = A.block(ind1ad(0), colind(0), ind1ad.size(), colind.size());
    tmpB = B.block(tmpind(0), colind(0), tmpind.size(), colind.size());
    product = tmpA.conjugate().cwiseProduct(tmpB);
    Eigen::MatrixXcf AB1ad = product * Eigen::VectorXcf::Ones(product.cols());
    AB1ad(Eigen::seq(0, Eigen::last, 2), 0).setZero(); 
    tmpA.setZero(); tmpB.setZero(); product.setZero(); tmpind.setZero();
    tmpind = ind1mi + k1mi;
    tmpA = A.block(ind1mi(0), colind(0), ind1mi.size(), colind.size());
    tmpB = B.block(tmpind(0), colind(0), tmpind.size(), colind.size());
    product = tmpA.conjugate().cwiseProduct(tmpB);
    Eigen::MatrixXcf AB1mi = product * Eigen::VectorXcf::Ones(product.cols());
    AB1mi(Eigen::seq(1, Eigen::last, 2), 0).setZero();
    tmpA.setZero(); tmpB.setZero(); product.setZero(); tmpind.setZero();

    tmpind = ind2 + k2;
    tmpA = A.block(ind2(0), colind(0), ind2.size(), colind.size());
    tmpB = B.block(tmpind(0), colind(0), tmpind.size(), colind.size());
    product = tmpA.conjugate().cwiseProduct(tmpB);
    Eigen::MatrixXcf AB2 = product * Eigen::VectorXcf::Ones(product.cols());    
    tmpA.setZero(); tmpB.setZero(); product.setZero(); tmpind.setZero();
    
    tmpind = ind3ad + k3ad;
    tmpA = A.block(ind3ad(0), colind(0), ind3ad.size(), colind.size());
    tmpB = B.block(tmpind(0), colind(0), tmpind.size(), colind.size());
    product = tmpA.conjugate().cwiseProduct(tmpB);
    Eigen::MatrixXcf AB3ad = product * Eigen::VectorXcf::Ones(product.cols());
    AB3ad(Eigen::seq(0, Eigen::last, 2), 0).setZero(); 
    tmpA.setZero(); tmpB.setZero(); product.setZero(); tmpind.setZero();
    tmpind = ind3mi + k3mi;
    tmpA = A.block(ind3mi(0), colind(0), ind3mi.size(), colind.size());
    tmpB = B.block(tmpind(0), colind(0), tmpind.size(), colind.size());
    product = tmpA.conjugate().cwiseProduct(tmpB);
    Eigen::MatrixXcf AB3mi = product * Eigen::VectorXcf::Ones(product.cols());
    AB3mi(Eigen::seq(1, Eigen::last, 2), 0).setZero();
    tmpA.setZero(); tmpB.setZero(); product.setZero(); tmpind.setZero();

    tmpind = ind4 + k4;
    tmpA = A.block(ind4(0), colind(0), ind4.size(), colind.size());
    tmpB = B.block(tmpind(0), colind(0), tmpind.size(), colind.size());
    product = tmpA.conjugate().cwiseProduct(tmpB);
    Eigen::MatrixXcf AB4 = product * Eigen::VectorXcf::Ones(product.cols());
    tmpA.setZero(); tmpB.setZero(); product.setZero(); tmpind.setZero();

    tmpind = ind5 + k5;
    tmpA = A.block(ind5(0), colind(0), ind5.size(), colind.size());
    tmpB = B.block(tmpind(0), colind(0), tmpind.size(), colind.size());
    product = tmpA.conjugate().cwiseProduct(tmpB);
    Eigen::MatrixXcf AB5 = product * Eigen::VectorXcf::Ones(product.cols());
    tmpA.setZero(); tmpB.setZero(); product.setZero(); tmpind.setZero();

    tmpind = ind6 + k6;
    tmpA = A.block(ind6(0), colind(0), ind6.size(), colind.size());
    tmpB = B.block(tmpind(0), colind(0), tmpind.size(), colind.size());
    product = tmpA.conjugate().cwiseProduct(tmpB);                                                                                                                               
    Eigen::MatrixXcf AB6 = product * Eigen::VectorXcf::Ones(product.cols());
    tmpA.setZero(); tmpB.setZero(); product.setZero(); tmpind.setZero();

    tmpind = ind7ad + k7ad;
    tmpA = A.block(ind7ad(0), colind(0), ind7ad.size(), colind.size());
    tmpB = B.block(tmpind(0), colind(0), tmpind.size(), colind.size());
    product = tmpA.conjugate().cwiseProduct(tmpB);
    Eigen::MatrixXcf AB7ad = product * Eigen::VectorXcf::Ones(product.cols());
    AB7ad(Eigen::seq(1, Eigen::last, 2), 0).setZero();
    tmpA.setZero(); tmpB.setZero(); product.setZero(); tmpind.setZero();
    tmpind = ind7mi + k7mi;
    tmpA = A.block(ind7mi(0), colind(0), ind7mi.size(), colind.size());
    tmpB = B.block(tmpind(0), colind(0), tmpind.size(), colind.size());
    product = tmpA.conjugate().cwiseProduct(tmpB);
    Eigen::MatrixXcf AB7mi = product * Eigen::VectorXcf::Ones(product.cols());
    AB7mi(Eigen::seq(0, Eigen::last, 2), 0).setZero();
    tmpA.setZero(); tmpB.setZero(); product.setZero(); tmpind.setZero();

    tmpind = ind8 + k8;
    tmpA = A.block(ind8(0), colind(0), ind8.size(), colind.size());
    tmpB = B.block(tmpind(0), colind(0), tmpind.size(), colind.size());
    product = tmpA.conjugate().cwiseProduct(tmpB);
    Eigen::MatrixXcf AB8 = product * Eigen::VectorXcf::Ones(product.cols());
    tmpA.setZero(); tmpB.setZero(); product.setZero(); tmpind.setZero();

    tmpind = ind9ad + k9ad;
    tmpA = A.block(ind9ad(0), colind(0), ind9ad.size(), colind.size());
    tmpB = B.block(tmpind(0), colind(0), tmpind.size(), colind.size());
    product = tmpA.conjugate().cwiseProduct(tmpB);
    Eigen::MatrixXcf AB9ad = product * Eigen::VectorXcf::Ones(product.cols());
    AB9ad(Eigen::seq(1, Eigen::last, 2), 0).setZero();
    tmpA.setZero(); tmpB.setZero(); product.setZero(); tmpind.setZero();
    tmpind = ind9mi + k9mi;
    tmpA = A.block(ind9mi(0), colind(0), ind9mi.size(), colind.size());
    tmpB = B.block(tmpind(0), colind(0), tmpind.size(), colind.size());
    product = tmpA.conjugate().cwiseProduct(tmpB);
    Eigen::MatrixXcf AB9mi = product * Eigen::VectorXcf::Ones(product.cols());
    AB9mi(Eigen::seq(0, Eigen::last, 2), 0).setZero();
    tmpA.setZero(); tmpB.setZero(); product.setZero(); tmpind.setZero();

    Eigen::ArrayXi rowind(1); 
    int offset, varind;
    for (int m = 1; m < 26; m++){
        rowind = Eigen::ArrayXi::Zero(1);
        Eigen::MatrixXf term1ad = (AB9mi.array() * mat_indexing(MADI[m], rowind, ind1ad).transpose().conjugate().array()).real();
        Eigen::MatrixXf term1mi = (AB9ad.array() * mat_indexing(MADI[m], rowind, ind1mi).transpose().conjugate().array()).real();
        rowind = Eigen::ArrayXi::Ones(1);
        Eigen::MatrixXf term2 = (AB8.array() * mat_indexing(MADI[m], rowind, ind2).transpose().conjugate().array()).real();
        rowind = Eigen::ArrayXi::Constant(1, 2);
        Eigen::MatrixXf term3ad = (AB7mi.array() * mat_indexing(MADI[m], rowind, ind3ad).transpose().conjugate().array()).real();
        Eigen::MatrixXf term3mi = (AB7ad.array() * mat_indexing(MADI[m], rowind, ind3mi).transpose().conjugate().array()).real();
        rowind = Eigen::ArrayXi::Constant(1, 3);
        Eigen::MatrixXf term4 = (AB6.array() * mat_indexing(MADI[m], rowind, ind4).transpose().conjugate().array()).real();
        Eigen::MatrixXf term5 = (AB5.array() * MADI[m].row(4).transpose().conjugate().array()).real();
        rowind = Eigen::ArrayXi::Constant(1, 5);
        Eigen::MatrixXf term6 = (AB4.array() * mat_indexing(MADI[m], rowind, ind6).transpose().conjugate().array()).real();
        rowind = Eigen::ArrayXi::Constant(1, 6);
        Eigen::MatrixXf term7ad = (AB3mi.array() * mat_indexing(MADI[m], rowind, ind7ad).transpose().conjugate().array()).real();
        Eigen::MatrixXf term7mi = (AB3ad.array() * mat_indexing(MADI[m], rowind, ind7mi).transpose().conjugate().array()).real();
        rowind = Eigen::ArrayXi::Constant(1, 7);
        Eigen::MatrixXf term8 = (AB2.array() * mat_indexing(MADI[m], rowind, ind8).transpose().conjugate().array()).real();
        rowind = Eigen::ArrayXi::Constant(1, 8);
        Eigen::MatrixXf term9ad = (AB1mi.array() * mat_indexing(MADI[m], rowind, ind9ad).transpose().conjugate().array()).real();
        Eigen::MatrixXf term9mi = (AB1ad.array() * mat_indexing(MADI[m], rowind, ind9mi).transpose().conjugate().array()).real();

        if (m - (std::floor((m - 1.0) / 5.0) * 5 + 1) == 0)
            offset = 0;
        else if (m - (std::floor((m - 1.0) / 5.0) * 5 + 1) == 1)
            offset = 2 * nzPML;
        else if (m - (std::floor((m - 1.0) / 5.0) * 5 + 1) == 2)
            offset = -2 * nzPML;
        else if (m - (std::floor((m - 1.0) / 5.0) * 5 + 1) == 3)
            offset = 2;
        else if (m - (std::floor((m - 1.0) / 5.0) * 5 + 1) == 4)
            offset = -2;

        varind = std::floor((m - 1.0) / 5.0);

        int th = 0;
        std::vector<std::pair<int, int>> indices;
        visit_lambda(ind1ad + offset, [&indices, th](int v, int i, int j){
            if (v >= th){
                indices.push_back(std::make_pair(i, j));
                return;
            }
        });
        int minind = indices[0].first;
        indices.clear();
        th = 2 * NPML;
        visit_lambda(ind1ad + offset, [&indices, th](int v, int i, int j){
            if (v < th){
                indices.push_back(std::make_pair(i, j));
            }
        });
        int maxind = indices[indices.size() - 1].first;
        indices.clear();
        prod.block(ind1ad(minind) + offset, varind, ind1ad(maxind) - ind1ad(minind) + 1, 1) \
            += mat_indexing(term1ad, \
                            Eigen::ArrayXi::LinSpaced(maxind - minind + 1, minind, maxind), \
                            Eigen::ArrayXi::Zero(1));
        term1ad.resize(0, 0);

        th = 0;
        visit_lambda(ind1mi + offset, [&indices, th](int v, int i, int j){
            if (v >= th){
                indices.push_back(std::make_pair(i, j));
                return;
            }
        });
        minind = indices[0].first;
        indices.clear();
        th = 2 * NPML;
        visit_lambda(ind1mi + offset, [&indices, th](int v, int i, int j){
            if (v < th){
                indices.push_back(std::make_pair(i, j));
            }
        });
        maxind = indices[indices.size() - 1].first;
        indices.clear();
        prod.block(ind1mi(minind) + offset, varind, ind1mi(maxind) - ind1mi(minind) + 1, 1) \
            += mat_indexing(term1mi, \
                            Eigen::ArrayXi::LinSpaced(maxind - minind + 1, minind, maxind), \
                            Eigen::ArrayXi::Zero(1));
        term1mi.resize(0, 0);

        th = 0;
        visit_lambda(ind2 + offset, [&indices, th](int v, int i, int j){
            if (v >= th){
                indices.push_back(std::make_pair(i, j));
                return;
            }
        });
        minind = indices[0].first;
        indices.clear();
        th = 2 * NPML;
        visit_lambda(ind2 + offset, [&indices, th](int v, int i, int j){
            if (v < th){
                indices.push_back(std::make_pair(i, j));
            }
        });     
        maxind = indices[indices.size() - 1].first;
        indices.clear();
        prod.block(ind2(minind) + offset, varind, ind2(maxind) - ind2(minind) + 1, 1) \
            += mat_indexing(term2, \
                            Eigen::ArrayXi::LinSpaced(maxind - minind + 1, minind, maxind), \
                            Eigen::ArrayXi::Zero(1));
        term2.resize(0, 0);

        th = 0;
        visit_lambda(ind3ad + offset, [&indices, th](int v, int i, int j){
            if (v >= th){
                indices.push_back(std::make_pair(i, j));
                return;
            }
        });
        minind = indices[0].first;
        indices.clear();
        th = 2 * NPML;
        visit_lambda(ind3ad + offset, [&indices, th](int v, int i, int j){
            if (v < th){
                indices.push_back(std::make_pair(i, j));
            }
        });
        maxind = indices[indices.size() - 1].first;
        indices.clear();
        prod.block(ind3ad(minind) + offset, varind, ind3ad(maxind) - ind3ad(minind) + 1, 1) \
            += mat_indexing(term3ad, \
                            Eigen::ArrayXi::LinSpaced(maxind - minind + 1, minind, maxind), \
                            Eigen::ArrayXi::Zero(1));
        term3ad.resize(0, 0);

        th = 0;
        visit_lambda(ind3mi + offset, [&indices, th](int v, int i, int j){
            if (v >= th){
                indices.push_back(std::make_pair(i, j));
                return;
            }
        });
        minind = indices[0].first;
        indices.clear();
        th = 2 * NPML;
        visit_lambda(ind3mi + offset, [&indices, th](int v, int i, int j){
            if (v < th){
                indices.push_back(std::make_pair(i, j));
            }
        });
        maxind = indices[indices.size() - 1].first;
        indices.clear();
        prod.block(ind3mi(minind) + offset, varind, ind3mi(maxind) - ind3mi(minind) + 1, 1) \
            += mat_indexing(term3mi, \
                            Eigen::ArrayXi::LinSpaced(maxind - minind + 1, minind, maxind), \
                            Eigen::ArrayXi::Zero(1));
        term3mi.resize(0, 0);

        th = 0;
        visit_lambda(ind4 + offset, [&indices, th](int v, int i, int j){
            if (v >= th){
                indices.push_back(std::make_pair(i, j));
                return;
            }
        });
        minind = indices[0].first;
        indices.clear();
        th = 2 * NPML;
        visit_lambda(ind4 + offset, [&indices, th](int v, int i, int j){
            if (v < th){
                indices.push_back(std::make_pair(i, j));
            }
        });
        maxind = indices[indices.size() - 1].first;
        indices.clear();
        prod.block(ind4(minind) + offset, varind, ind4(maxind) - ind4(minind) + 1, 1) \
            += mat_indexing(term4, \
                            Eigen::ArrayXi::LinSpaced(maxind - minind + 1, minind, maxind), \
                            Eigen::ArrayXi::Zero(1));
        term4.resize(0, 0);

        th = 0;
        visit_lambda(ind5 + offset, [&indices, th](int v, int i, int j){
            if (v >= th){
                indices.push_back(std::make_pair(i, j));
                return;
            }
        });
        minind = indices[0].first;
        indices.clear();
        th = 2 * NPML;
        visit_lambda(ind5 + offset, [&indices, th](int v, int i, int j){
            if (v < th){
                indices.push_back(std::make_pair(i, j));
            }
        });
        maxind = indices[indices.size() - 1].first;
        indices.clear();
        prod.block(ind5(minind) + offset, varind, ind5(maxind) - ind5(minind) + 1, 1) \
            += mat_indexing(term5, \
                            Eigen::ArrayXi::LinSpaced(maxind - minind + 1, minind, maxind), \
                            Eigen::ArrayXi::Zero(1));
        term5.resize(0, 0);

        th = 0;                                                                                                                                                             
        visit_lambda(ind6 + offset, [&indices, th](int v, int i, int j){
            if (v >= th){
                indices.push_back(std::make_pair(i, j));
                return;
            }
        });
        minind = indices[0].first;
        indices.clear();
        th = 2 * NPML;
        visit_lambda(ind6 + offset, [&indices, th](int v, int i, int j){
            if (v < th){
                indices.push_back(std::make_pair(i, j));
            }
        });
        maxind = indices[indices.size() - 1].first;
        indices.clear();
        prod.block(ind6(minind) + offset, varind, ind6(maxind) - ind6(minind) + 1, 1) \
            += mat_indexing(term6, \
                            Eigen::ArrayXi::LinSpaced(maxind - minind + 1, minind, maxind), \
                            Eigen::ArrayXi::Zero(1));
        term6.resize(0, 0);

        th = 0;
        visit_lambda(ind7ad + offset, [&indices, th](int v, int i, int j){
            if (v >= th){
                indices.push_back(std::make_pair(i, j));
                return;
            }
        });
        minind = indices[0].first;
        indices.clear();
        th = 2 * NPML;
        visit_lambda(ind7ad + offset, [&indices, th](int v, int i, int j){
            if (v < th){
                indices.push_back(std::make_pair(i, j));
            }
        });
        maxind = indices[indices.size() - 1].first;
        indices.clear();
        prod.block(ind7ad(minind) + offset, varind, ind7ad(maxind) - ind7ad(minind) + 1, 1) \
            += mat_indexing(term7ad, \
                            Eigen::ArrayXi::LinSpaced(maxind - minind + 1, minind, maxind), \
                            Eigen::ArrayXi::Zero(1));
        term7ad.resize(0, 0);

        th = 0;
        visit_lambda(ind7mi + offset, [&indices, th](int v, int i, int j){
            if (v >= th){
                indices.push_back(std::make_pair(i, j));
                return;
            }
        });
        minind = indices[0].first;
        indices.clear();
        th = 2 * NPML;
        visit_lambda(ind7mi + offset, [&indices, th](int v, int i, int j){
            if (v < th){
                indices.push_back(std::make_pair(i, j));
            }
        });
        maxind = indices[indices.size() - 1].first;
        indices.clear();
        prod.block(ind7mi(minind) + offset, varind, ind7mi(maxind) - ind7mi(minind) + 1, 1) \
            += mat_indexing(term7mi, \
                            Eigen::ArrayXi::LinSpaced(maxind - minind + 1, minind, maxind), \
                            Eigen::ArrayXi::Zero(1));
        term7mi.resize(0, 0);

        th = 0;
        visit_lambda(ind8 + offset, [&indices, th](int v, int i, int j){
            if (v >= th){
                indices.push_back(std::make_pair(i, j));
                return;
            }
        });
        minind = indices[0].first;
        indices.clear();
        th = 2 * NPML;
        visit_lambda(ind8 + offset, [&indices, th](int v, int i, int j){
            if (v < th){
                indices.push_back(std::make_pair(i, j));
            }
        });
        maxind = indices[indices.size() - 1].first;
        indices.clear();
        prod.block(ind8(minind) + offset, varind, ind8(maxind) - ind8(minind) + 1, 1) \
            += mat_indexing(term8, \
                            Eigen::ArrayXi::LinSpaced(maxind - minind + 1, minind, maxind), \
                            Eigen::ArrayXi::Zero(1));
        term8.resize(0, 0);

        th = 0;
        visit_lambda(ind9ad + offset, [&indices, th](int v, int i, int j){
            if (v >= th){
                indices.push_back(std::make_pair(i, j));
                return;
            }
        });
        minind = indices[0].first;
        indices.clear();    
        th = 2 * NPML;
        visit_lambda(ind9ad + offset, [&indices, th](int v, int i, int j){
            if (v < th){
                indices.push_back(std::make_pair(i, j));
            }
        });
        maxind = indices[indices.size() - 1].first;
        indices.clear();
        prod.block(ind9ad(minind) + offset, varind, ind9ad(maxind) - ind9ad(minind) + 1, 1) \
            += mat_indexing(term9ad, \
                            Eigen::ArrayXi::LinSpaced(maxind - minind + 1, minind, maxind), \
                            Eigen::ArrayXi::Zero(1));
        term9ad.resize(0, 0);

        th = 0;
        visit_lambda(ind9mi + offset, [&indices, th](int v, int i, int j){
            if (v >= th){
                indices.push_back(std::make_pair(i, j));
                return;
            }
        });
        minind = indices[0].first;
        indices.clear();
        th = 2 * NPML;
        visit_lambda(ind9mi + offset, [&indices, th](int v, int i, int j){
            if (v < th){
                indices.push_back(std::make_pair(i, j));
            }
        });
        maxind = indices[indices.size() - 1].first;
        indices.clear();
        prod.block(ind9mi(minind) + offset, varind, ind9mi(maxind) - ind9mi(minind) + 1, 1) \
            += mat_indexing(term9mi, \
                            Eigen::ArrayXi::LinSpaced(maxind - minind + 1, minind, maxind), \
                            Eigen::ArrayXi::Zero(1));
        term9mi.resize(0, 0);

        indices.shrink_to_fit();
    }

    Eigen::MatrixXf prod_temp(NPML, 5); prod_temp.setZero();
    //for (int i = 0; i < NPML; i++)
    //    prod_temp.row(i) = prod.row(2 * i) + prod.row(2 * i + 1);
    prod_temp = prod(Eigen::seq(0, 2 * NPML - 1, 2), Eigen::all) + prod(Eigen::seq(1, 2 * NPML - 1, 2), Eigen::all);

    Eigen::MatrixXf prod_out = Eigen::MatrixXf(nx * nz, 5);
    prod_out.setZero();

    for (int n = 0; n < prod_temp.rows(); n++){
        int x_pos = std::floor((n + 1) / nzPML) - PML_thick;
        int z_pos = n - (x_pos + PML_thick) * nzPML - PML_thick;
        if (z_pos < 0)
            z_pos = 0;
        else if (z_pos > nz - 1)
            z_pos = nz - 1;
        if (x_pos < 0)
            x_pos = 0;
        else if (x_pos > nx - 1)
            x_pos = nx - 1;
        int boundind = z_pos + nz * x_pos;
        prod_out.row(boundind) += prod_temp.row(n);
    }
    prod_out.resize(prod_out.size(), 1);

    ind1mi.resize(0); ind1ad.resize(0); ind2.resize(0);
    ind3mi.resize(0); ind3ad.resize(0); ind4.resize(0);
    ind5.resize(0); ind6.resize(0);
    ind7mi.resize(0); ind7ad.resize(0); ind8.resize(0);
    ind9mi.resize(0); ind9ad.resize(0);
    colind.resize(0); rowind.resize(0);
    AB1ad.resize(0, 0); AB1mi.resize(0, 0); AB2.resize(0, 0); AB4.resize(0, 0);
    AB3ad.resize(0, 0); AB3mi.resize(0, 0); AB5.resize(0, 0); AB6.resize(0, 0);
    AB7ad.resize(0, 0); AB7mi.resize(0, 0); AB9ad.resize(0, 0); AB9mi.resize(0, 0);
    prod_temp.resize(0, 0);
    prod.resize(0, 0);
    tmpind.resize(0);
    tmpA.resize(0, 0); tmpA.data().squeeze();
    tmpB.resize(0, 0); tmpB.data().squeeze();
    product.resize(0, 0); product.data().squeeze();

    return prod_out;
}

