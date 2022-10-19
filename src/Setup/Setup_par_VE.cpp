#include "Setup_par_VE.h"

Setup_par_VE::Setup_par_VE(){

    //nx = 50; nz = 50; modeltype = 2;
    //PML_thick = 10;
    //nxPML = nx + 2 * PML_thick; nzPML = nz + 2 * PML_thick;
    //NN = nz * nx; NPML = nzPML * nxPML;
    //dx = 10; dz = 10;

    //sAcq = 1; rAcq = 1;
    //roffset = 0; 
    //soffset = 0;

    //P_grid = 1;
    //P_smooth = 2;

    //nP = 2;

    //numbands = 6; step = 10;
    //startband = 1.0;
    //endband1 = 2.0;
    //endbandend = 15.0;
    //
    //Amp_scale = 1.0;
    //
    //omega0 = 2 * pi * 30;


    //std::cout << "Parameter settled." << std::endl;

    Read_par();
    Get_model();
    Get_Acq();
    std::tie(S_sur, std::ignore) = Make_General_source(sx_sur, sz_sur, M11_sur, M12_sur, M22_sur, nz, nx, PML_thick);
    R = Define_MC_point_receivers(rz, rx, nz, nx, PML_thick);
    Get_select_P();
    Get_freqs();
    Get_inds();
}

Setup_par_VE::~Setup_par_VE(){

    P.resize(0, 0); P.data().squeeze();
    P_big.resize(0, 0); P_big.data().squeeze();
    model0.resize(0, 0); model_true.resize(0, 0);
    sx_sur.resize(0); sx_SWD.resize(0); sz_sur.resize(0); sz_SWD.resize(0);
    rx.resize(0); rz.resize(0);
    R.resize(0, 0); R.data().squeeze();
    M11_sur.resize(0); M12_sur.resize(0); M22_sur.resize(0);
    S_sur.resize(0, 0); S_sur.data().squeeze();
    freq.resize(0); fwave.resize(0, 0);

    std::cout << "Parameter object destroyed." << std::endl;
}

void Setup_par_VE::Read_par(){

    std::ifstream f1;
    f1.open("../data/parameter.txt");
    if(!f1){
        std::cout << "?" << std::endl;
        exit(0);
    }
    while(!f1.eof()){
        f1 >> nx; f1 >> nz;
        f1 >> dx; f1 >> dz;
        f1 >> PML_thick;
        f1 >> sAcq; f1 >> rAcq;
        f1 >> soffset; f1 >> roffset;
        f1 >> P_grid; f1 >> P_smooth;
        f1 >> nP;
        f1 >> numbands; f1 >> step;
        f1 >> startband; f1 >> endband1; f1 >> endbandend;
        f1 >> Amp_scale;
        f1 >> f0;
    }
    std::cout << "Reading over." << std::endl;
    f1.close();
    omega0 = 2 * pi * f0;
    nxPML = nx + 2 * PML_thick; nzPML = nz + 2 * PML_thick;
    NN = nz * nx; NPML = nzPML * nxPML;
    modeltype = 2;
    std::cout << "**************************" << std::endl;
    std::cout << "nx =   " << nx << "    nz =     " << nz << std::endl;
    std::cout << "dx =   " << dx << "    dz =     " << dz << std::endl;
    std::cout << "PML_thick=       " << PML_thick << std::endl;
    std::cout << "sAcq =   " << sAcq << "    rAcq =     " << rAcq << std::endl;
    std::cout << "soffset =   " << soffset << "    roffset =     " << roffset << std::endl;
    std::cout << "P_grid = " << P_grid << std::endl;
    std::cout << "P_smooth = " << P_smooth << std::endl;
    std::cout << "numbands =   " << numbands << "    step =     " << step << std::endl;
    std::cout << "startband =   " << startband << "    endband1 =     " << endband1 << std::endl;
    std::cout << "endbandend =   " << endbandend << std::endl;
    std::cout << "Amp_scale =   " << Amp_scale << std::endl;
    std::cout << "Omega0 = " << omega0 << std::endl;
    std::cout << "**************************" << std::endl;

}


void Setup_par_VE::Get_model(){
    std::tie(model0, model_true) = Make_model(nz, nx, modeltype);
    model0.block(NN, 0, NN, 1) = model0.block(NN, 0, NN, 1).array().abs2().inverse();
    model0.block(3 * NN, 0, NN, 1) = model0.block(3 * NN, 0, NN, 1).array().abs2().inverse();
    model_true.block(NN, 0, NN, 1) = model_true.block(NN, 0, NN, 1).array().abs2().inverse();
    model_true.block(3 * NN, 0, NN, 1) = model_true.block(3 * NN, 0, NN, 1).array().abs2().inverse();

    std::cout << "Models inverted and settled." << std::endl;
}

void Setup_par_VE::Read_model(){

    Eigen::MatrixXf vp_true(nz * nx, 1);
    Eigen::MatrixXf vs_true(nz * nx, 1);
    Eigen::MatrixXf Qp_true(nz * nx, 1);
    Eigen::MatrixXf Qs_true(nz * nx, 1);
    Eigen::MatrixXf rho_true(nz * nx, 1);
    
    std::ifstream truevp("../data/truemodels/vp_true_300x100.dat", std::ios::in | std::ios::binary);
    if(truevp){
        while(!truevp.eof()){
            for (int i = 0; i < nz * nx; i++)
                truevp >> vp_true(i, 0);
        }
        truevp.close();
    }
    else{
        std::cout << "?" << std::endl;
        exit(0);
    }
    std::ifstream truevs("../data/truemodels/vs_true_300x100.dat", std::ios::in | std::ios::binary);
    if(truevs){
        while(!truevs.eof()){
            for (int i = 0; i < nz * nx; i++)
                truevs >> vs_true(i, 0);
        }
        truevs.close();
    }
    else{
        std::cout << "?" << std::endl;
        exit(0);
    }
    std::ifstream trueQp("../data/truemodels/Qp_true_300x100.dat", std::ios::in | std::ios::binary);
    if(trueQp){
        while(!trueQp.eof()){
            for (int i = 0; i < nz * nx; i++)
                trueQp >> Qp_true(i, 0);
        }
        trueQp.close();
    }
    else{
        std::cout << "?" << std::endl;
        exit(0);
    }
    std::ifstream trueQs("../data/truemodels/Qs_true_300x100.dat", std::ios::in | std::ios::binary);
    if(trueQs){
        while(!trueQs.eof()){
            for (int i = 0; i < nz * nx; i++)
                trueQs >> Qs_true(i, 0);
        }
        trueQs.close();
    }
    else{
        std::cout << "?" << std::endl;
        exit(0);
    }
    std::ifstream truerho("../data/truemodels/rho_true_300x100.dat", std::ios::in | std::ios::binary);
    if(truerho){
        while(!truerho.eof()){
            for (int i = 0; i < nz * nx; i++)
                truerho >> rho_true(i, 0);
        }
        truerho.close();
    }
    else{
        std::cout << "?" << std::endl;
        exit(0);
    }

    Eigen::MatrixXf model0 = Eigen::MatrixXf::Constant(nz * nx, 5, 0.0);
    Eigen::MatrixXf vp_0 = Eigen::MatrixXf::Constant(nz, nx, vp_true(0, 0));
    Eigen::MatrixXf vs_0 = Eigen::MatrixXf::Constant(nz, nx, vs_true(0, 0));
    Eigen::MatrixXf rho_0 = Eigen::MatrixXf::Constant(nz, nx, rho_true(0, 0));
    Eigen::MatrixXf Qp_inv_0 = Eigen::MatrixXf::Constant(nz, nx, Qp_true(0, 0));
    Eigen::MatrixXf Qs_inv_0 = Eigen::MatrixXf::Constant(nz, nx, Qs_true(0, 0));
    
    model0.col(0) = rho_0.array();
    model0.col(1) = vp_0.array(); model0.col(2) = Qp_inv_0.array();
    model0.col(3) = vs_0.array(); model0.col(4) = Qs_inv_0.array();
   
    Eigen::MatrixXf model_true = Eigen::MatrixXf::Constant(nz * nx, 5, 0.0);
    model_true.col(0) = rho_true.array(); model_true.col(1) = vp_true.array(); model_true.col(2) = Qp_inv_0.array();
    model_true.col(3) = vs_0.array(); model_true.col(4) = Qs_inv_0.array();
    
    
    vp_true.resize(0, 0); rho_true.resize(0, 0); vs_true.resize(0, 0);
    Qp_true.resize(0, 0); Qs_true.resize(0, 0);
    vp_0.resize(0, 0); rho_0.resize(0, 0); vs_0.resize(0, 0);
    Qp_inv_0.resize(0, 0); Qs_inv_0.resize(0, 0);

    model0.resize(nz * nx * 5, 1);
    model_true.resize(nz * nx * 5, 1);
    std::cout << "Model done." << std::endl;

}

void Setup_par_VE::Get_Acq(){
    std::tie(sx_sur, sx_SWD, sz_sur, sz_SWD, rx, rz) = Set_Acq(sAcq, soffset, rAcq, roffset, nz, nx);
    sx = std::make_tuple(sx_sur, sx_SWD);
    sz = std::make_tuple(sz_sur, sz_SWD);
    ns_sur = sx_sur.size(); ns_SWD = sx_SWD.size();

    M11_sur = Eigen::RowVectorXf::Constant(ns_sur, 1.0);
    M12_sur = Eigen::RowVectorXf::Constant(ns_sur, 0.0);
    M22_sur = Eigen::RowVectorXf::Constant(ns_sur, 1.0);

    std::cout << "Acquisition defined." << std::endl;
}

void Setup_par_VE::Get_select_P(){

    Eigen::SparseMatrix<float> P_all, P_big_all;
    std::tie(P_all, P_big_all) = Make_P_sparse_alt(nz, nx, PML_thick, P_grid, P_smooth);

    Eigen::SparseMatrix<float> P_temp(P_all.rows(), nP * NN);
    Eigen::SparseMatrix<float> P_big_temp(P_big_all.rows(), nP * NN);

    typedef Eigen::Triplet<float> T;
    std::vector<T> triplet2;
    for (int i = 0; i < P_all.outerSize(); i++){
        for (Eigen::SparseMatrix<float>::InnerIterator it(P_all, i); it; ++it){
            if (it.value() && (it.col() < nP * NN))
                triplet2.push_back(T(it.row(), it.col(), it.value()));
        }
    }
    P_temp.setFromTriplets(triplet2.begin(), triplet2.end());

    P = P_temp;

    std::vector<T> triplet3;
    for (int i = 0; i < P_big_all.outerSize(); i++){
        for (Eigen::SparseMatrix<float>::InnerIterator it(P_big_all, i); it; ++it){
            if (it.value() && (it.col() < nP * NN))
                triplet3.push_back(T(it.row(), it.col(), it.value()));
        }
    }
    P_big_temp.setFromTriplets(triplet3.begin(), triplet3.end());

    P_big = P_big_temp;

    triplet2.clear(); triplet2.shrink_to_fit();
    triplet3.clear(); triplet3.shrink_to_fit();

    P_all.resize(0, 0); P_all.data().squeeze();
    P_big_all.resize(0, 0); P_big_all.data().squeeze();
    P_temp.resize(0, 0); P_temp.data().squeeze();
    P_big_temp.resize(0, 0); P_big_temp.data().squeeze();
    std::cout << "P calculated." << std::endl;
}

void Setup_par_VE::Get_freqs(){

    freq = Eigen::RowVectorXf::Zero(numbands * step);
    Eigen::RowVectorXf startfreq = Eigen::RowVectorXf::Constant(numbands, 1);
    Eigen::RowVectorXf endfreq = Eigen::RowVectorXf::LinSpaced(numbands, endband1, endbandend);
    for (int n = 0; n < numbands; n++)
        freq.block(0, n * step, 1, step) = Eigen::RowVectorXf::LinSpaced(step, startfreq(n), endfreq(n));

    fwave = Eigen::MatrixXcf::Ones(freq.size(), S_sur.cols());

    startfreq.resize(0); endfreq.resize(0);

    std::cout << "Freq end." << std::endl;
}

void Setup_par_VE::Get_inds(){

    Eigen::RowVectorXf ind_model = Eigen::RowVectorXf::LinSpaced(P.cols(), 0, P.cols() - 1);
    Eigen::RowVectorXf ind_M11 = Eigen::RowVectorXf::LinSpaced(ns_sur, P.cols(), P.cols() + ns_sur - 1);
    Eigen::RowVectorXf ind_M12 = Eigen::RowVectorXf::LinSpaced(ns_sur, P.cols() + ns_sur, P.cols() + 2 * ns_sur - 1);
    Eigen::RowVectorXf ind_M22 = Eigen::RowVectorXf::LinSpaced(ns_sur, P.cols() + 2 * ns_sur, P.cols() + 3 * ns_sur - 1);
    ind = std::make_tuple(ind_model, ind_M11, ind_M12, ind_M22);

    ind_model.resize(0);
    ind_M11.resize(0);
    ind_M12.resize(0);
    ind_M22.resize(0);

}
