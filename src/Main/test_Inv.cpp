#include "../General/Generalfuncs.h"
#include "../Setup/Make_model.h"
#include "../Setup/Set_Acq.h"
#include "../Setup/Define_MC_point_receivers.h"
#include "../Setup/Make_General_source.h"
#include "../Setup/Make_P_sparse_alt.h"
#include "../Forward/Get_data_anelastic.h"
#include "../Inversion/FDFWI_VE.h"

int main(){                                                                                                 
    int nx = 50; int nz = 50; int modeltype = 2;
    int PML_thick = 10;
    int nxPML = nx + 2 * PML_thick; int nzPML = nz + 2 * PML_thick;
    int NN = nz * nx; int NPML = nzPML * nxPML;
    int dx = 10; int dz = 10;
    Eigen::MatrixXf model0, model_true;
    std::tie(model0, model_true) = Make_model(nz, nx, modeltype);
    model0.block(NN, 0, NN, 1) = model0.block(NN, 0, NN, 1).array().abs2().inverse();
    model0.block(3 * NN, 0, NN, 1) = model0.block(3 * NN, 0, NN, 1).array().abs2().inverse();
    model_true.block(NN, 0, NN, 1) = model_true.block(NN, 0, NN, 1).array().abs2().inverse();
    model_true.block(3 * NN, 0, NN, 1) = model_true.block(3 * NN, 0, NN, 1).array().abs2().inverse();
    int sAcq = 11; int rAcq = 1; int roffset = 0; int soffset = 0;
    Eigen::RowVectorXf sx_sur, sx_SWD, sz_sur, sz_SWD;
    Eigen::RowVectorXf rx, rz;
    std::tie(sx_sur, sx_SWD, sz_sur, sz_SWD, rx, rz) = Set_Acq(sAcq, soffset, rAcq, roffset, nz, nx);
    std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf> sx = std::make_tuple(sx_sur, sx_SWD);
    std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf> sz = std::make_tuple(sz_sur, sz_SWD);

    Eigen::SparseMatrix<float> R = Define_MC_point_receivers(rz, rx, nz, nx, PML_thick);
    std::cout << "R defined." << std::endl;
    int Amp_scale = 1;
    int ns_sur = sx_sur.size(); int ns_SWD = sx_SWD.size();
    Eigen::RowVectorXf M11_sur = Eigen::RowVectorXf::Constant(ns_sur, 1.0);
    Eigen::RowVectorXf M12_sur = Eigen::RowVectorXf::Constant(ns_sur, 0.0);
    Eigen::RowVectorXf M22_sur = Eigen::RowVectorXf::Constant(ns_sur, 1.0);
    std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf> MT_sur \
        = std::make_tuple(M11_sur, M12_sur, M22_sur);
    std::cout << "Source start." << std::endl;
    Eigen::SparseMatrix<float> S_sur;
    std::tie(S_sur, std::ignore) = Make_General_source(sx_sur, sz_sur, M11_sur, M12_sur, M22_sur, nz, nx, PML_thick);
    //std::tie(S_SWD, std::ignore) = Make_General_source(sx_SWD, sz_SWD, M11_SWD, M12_SWD, M22_SWD, nz, nx, PML_thick);
    std::cout << "Source end." << std::endl;
    int P_grid = 1; int P_smooth = 1;
    std::cout << "P start." << std::endl;
    Eigen::SparseMatrix<float> P_all, P_big_all;
    std::tie(P_all, P_big_all) = Make_P_sparse_alt(nz, nx, PML_thick, P_grid, P_smooth);
    std::cout << "P end." << std::endl;
    int nP = 2;
    Eigen::SparseMatrix<float> P(P_all.rows(), nP * NN);
    Eigen::SparseMatrix<float> P_big(P_big_all.rows(), nP * NN);
    typedef Eigen::Triplet<float> T;
    std::vector<T> triplet2;
    for (int i = 0; i < P_all.outerSize(); i++){
        for (Eigen::SparseMatrix<float>::InnerIterator it(P_all, i); it; ++it){
            if (it.value() && (it.col() < nP * NN))
                triplet2.push_back(T(it.row(), it.col(), it.value()));
        }
    }
    P.setFromTriplets(triplet2.begin(), triplet2.end());
    std::vector<T> triplet3;
    for (int i = 0; i < P_big_all.outerSize(); i++){
        for (Eigen::SparseMatrix<float>::InnerIterator it(P_big_all, i); it; ++it){
            if (it.value() && (it.col() < nP * NN))
                triplet3.push_back(T(it.row(), it.col(), it.value()));
        }
    }
    P_big.setFromTriplets(triplet3.begin(), triplet3.end());                                
    triplet2.clear(); triplet2.shrink_to_fit();
    triplet3.clear(); triplet3.shrink_to_fit();

    Eigen::RowVectorXf ind_model = Eigen::RowVectorXf::LinSpaced(P.cols(), 0, P.cols() - 1);
    Eigen::RowVectorXf ind_M11 = Eigen::RowVectorXf::LinSpaced(ns_sur, P.cols(), P.cols() + ns_sur - 1);
    Eigen::RowVectorXf ind_M12 = Eigen::RowVectorXf::LinSpaced(ns_sur, P.cols() + ns_sur, P.cols() + 2 * ns_sur - 1);
    Eigen::RowVectorXf ind_M22 = Eigen::RowVectorXf::LinSpaced(ns_sur, P.cols() + 2 * ns_sur, P.cols() + 3 * ns_sur - 1);
    std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf> \
    ind = std::make_tuple(ind_model, ind_M11, ind_M12, ind_M22);

    std::cout << "Freq start." << std::endl;
    int numbands = 6; int step = 10;
    Eigen::RowVectorXf freq = Eigen::RowVectorXf::Zero(numbands * step);
    int startband = 1; int endband1 = 2; int endbandend = 15;
    Eigen::RowVectorXf startfreq = Eigen::RowVectorXf::Constant(numbands, 1);
    Eigen::RowVectorXf endfreq = Eigen::RowVectorXf::LinSpaced(numbands, endband1, endbandend);
    for (int n = 0; n < numbands; n++)
        freq.block(0, n * step, 1, step) = Eigen::RowVectorXf::LinSpaced(step, startfreq(n), endfreq(n));
    Eigen::MatrixXcf fwave = Eigen::MatrixXcf::Ones(freq.size(), S_sur.cols());
    std::cout << "Freq end." << std::endl;
    std::cout << "Parameterization over." << std::endl;

    float omega0 = 2 * pi * 30;
    std::vector<Eigen::SparseMatrix<std::complex<float>>> D(freq.size());
    bool flag_D = Get_data_anelastic(D, freq, fwave, model_true, R, omega0, S_sur, PML_thick, nz, nx, dz);

    std::cout << "Inversion preperation." << std::endl;
    float tol = 1e-5;
    Eigen::MatrixXf Pmodel = P.transpose() * model_true;
    float scalem[5] = {model_true.block(0, 0, NN, 1).maxCoeff(), \
            model_true.block(NN, 0, NN, 1).maxCoeff(), \
            model_true.block(2 * NN, 0, NN, 1).maxCoeff(), \
            model_true.block(3 * NN, 0, NN, 1).maxCoeff(), \
            model_true.block(4 * NN, 0, NN, 1).maxCoeff()};
    float scaleS = 0; float scaleSM = 1;
    std::tuple<float *, float, float, float> scale = std::make_tuple(scalem, scaleS, 1.0, scaleSM);
    float reg_fac = 1e-2; float stabregfac = reg_fac;
    int maxits = 20; int numits = 2; int optype = 2; 
    float glob_it_tol = 1e+5; float global_res_tol = 10;
    float global_decrease_tol = 0.95; float global_lock_tol = 1 / 20;

    Eigen::MatrixXf ss_model0_inv = Eigen::MatrixXf::Zero(model0.rows(), model0.cols());
    ss_model0_inv.block(0, 0, NN, 1) = model0.block(0, 0, NN, 1).array() / scalem[0];
    ss_model0_inv.block(NN, 0, NN, 1) = model0.block(NN, 0, NN, 1).array() / scalem[1];
    ss_model0_inv.block(2 * NN, 0, NN, 1) = model0.block(2 * NN, 0, NN, 1).array() / scalem[2];
    ss_model0_inv.block(3 * NN, 0, NN, 1) = model0.block(3 * NN, 0, NN, 1).array() / scalem[3];
    ss_model0_inv.block(4 * NN, 0, NN, 1) = model0.block(4 * NN, 0, NN, 1).array() / scalem[4];

    Eigen::MatrixXf model_start(P.cols(), 1);
    model_start.setZero();
    
    std::cout << "Inversion starts." << std::endl;
    FDFWI_VE(D, freq, step, fwave, nz, nx, dz, model_start, ss_model0_inv, \
                                                omega0, R, sz, sx, MT_sur, optype, numits, PML_thick, tol, \
                                                maxits, scale, ind, reg_fac, stabregfac, P, P_big);
    Eigen::MatrixXf model_out = P * model_start + ss_model0_inv;
    model_out.block(0, 0, NN, 1) = model_out.block(0, 0, NN, 1).array() * scalem[0];
    model_out.block(NN, 0, NN, 1) = model_out.block(NN, 0, NN, 1).array() * scalem[1];
    model_out.block(2 * NN, 0, NN, 1) = model_out.block(2 * NN, 0, NN, 1).array() * scalem[2];
    model_out.block(3 * NN, 0, NN, 1) = model_out.block(3 * NN, 0, NN, 1).array() * scalem[3];
    model_out.block(4 * NN, 0, NN, 1) = model_out.block(4 * NN, 0, NN, 1).array() * scalem[4];

    Eigen::MatrixXf vp_true = model_true.block(NN, 0, NN, 1).array().pow(-0.5);
    Eigen::MatrixXf rho_true = model_true.block(0, 0, NN, 1);
    Eigen::MatrixXf vp_out = model_out.block(NN, 0, NN, 1).array().pow(-0.5);
    Eigen::MatrixXf rho_out = model_out.block(0, 0, NN, 1);

    std::ofstream truemodel("../data/truemodel.dat", std::ios::out | std::ios::trunc);
    if(truemodel){
        truemodel << vp_true << "\n" << rho_true << "\n";
        truemodel.close();
    }
    std::ofstream invmodel("../data/invmodel.dat", std::ios::out | std::ios::trunc);
    if(invmodel){
        invmodel << vp_out << "\n" << rho_out << "\n";
        invmodel.close();
    }

    model0.resize(0, 0); model_true.resize(0, 0);
    sx_sur.resize(0); sx_SWD.resize(0); sz_sur.resize(0); sz_SWD.resize(0);
    rx.resize(0); rz.resize(0);
    R.resize(0, 0); R.data().squeeze();
    M11_sur.resize(0); M12_sur.resize(0); M22_sur.resize(0);
    S_sur.resize(0, 0); S_sur.data().squeeze();
    P_all.resize(0, 0); P_all.data().squeeze();
    P.resize(0, 0); P.data().squeeze();
    P_big_all.resize(0, 0); P_big_all.data().squeeze();
    P_big.resize(0, 0); P_big.data().squeeze();
    ind_model.resize(0); ind_M11.resize(0);
    ind_M12.resize(0); ind_M22.resize(0);
    freq.resize(0); fwave.resize(0, 0);
    startfreq.resize(0); endfreq.resize(0);
    D.clear(); D.shrink_to_fit();
    Pmodel.resize(0, 0);
    model_out.resize(0, 0); 
    vp_true.resize(0, 0); rho_true.resize(0, 0);
    vp_out.resize(0, 0); rho_out.resize(0, 0);
    std::cout << "Freed." << std::endl << "Done." << std::endl;


    return 0;

}
