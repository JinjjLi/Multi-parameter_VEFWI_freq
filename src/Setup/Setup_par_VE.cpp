#include "Setup_par_VE.h"

Setup_par_VE::Setup_par_VE(){

    Read_par();
    //Get_model();
    Read_model();
    Get_Acq();
    std::tie(S, R) = Define_Acquisition_Explosive(sz, sx, rz, rx, nx, nz, PML_thick);
    Get_select_P();
    Get_freqs();
}

Setup_par_VE::~Setup_par_VE(){

    P.resize(0, 0); P.data().squeeze();
    P_big.resize(0, 0); P_big.data().squeeze();
    model0.resize(0, 0); model_true.resize(0, 0);
    sx.resize(0); sz.resize(0);
    rx.resize(0); rz.resize(0);
    R.resize(0, 0); R.data().squeeze();
    S.resize(0, 0); S.data().squeeze();
    freq.resize(0); fwave.resize(0, 0);

    std::cout << "Parameter object destroyed." << std::endl;
}

void Setup_par_VE::Read_par(){

    std::ifstream f1;
    f1.open("./data/parameter.txt");
    if(!f1){
        std::cout << "Error reading parameters!!!" << std::endl;
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
        f1 >> model_true_name;
        f1 >> model0_name;
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
    std::cout << "model_true = " << model_true_name << std::endl;
    std::cout << "model0 = " << model0_name << std::endl;
    std::cout << "**************************" << std::endl;
}


void Setup_par_VE::Get_model(){
    std::tie(model0, model_true) = Make_model(nz, nx, modeltype);
    //Read_model(nz, nx, modeltype);
    model0.block(NN, 0, NN, 1) = model0.block(NN, 0, NN, 1).array().abs2().inverse();
    model0.block(3 * NN, 0, NN, 1) = model0.block(3 * NN, 0, NN, 1).array().abs2().inverse();
    model_true.block(NN, 0, NN, 1) = model_true.block(NN, 0, NN, 1).array().abs2().inverse();
    model_true.block(3 * NN, 0, NN, 1) = model_true.block(3 * NN, 0, NN, 1).array().abs2().inverse();

    std::cout << "Models inverted and settled." << std::endl;
}

void Setup_par_VE::Read_model(){

    model_true = Eigen::MatrixXf::Zero(nz * nx * 5, 1);

    //std::ifstream truemodel("../data/model_true_Mar_300x150.dat", std::ios::in | std::ios::binary);
    std::ifstream truemodel(model_true_name, std::ios::in | std::ios::binary);
    if(truemodel){
        while(!truemodel.eof()){
            for (int i = 0; i < 5 * nz * nx; i++)
                truemodel >> model_true(i, 0);
        }
        truemodel.close();
    }
    else{
        std::cout << "Error reading models!!!" << std::endl;
        exit(0);
    }

    model0 = Eigen::MatrixXf::Zero(nz * nx * 5, 1);
    //std::ifstream backmodel("../data/model0_Mar_300x150.dat", std::ios::in | std::ios::binary);
    std::ifstream backmodel(model0_name, std::ios::in | std::ios::binary);
    if(backmodel){
        while(!backmodel.eof()){
            for (int i = 0; i < 5 * nz * nx; i++)
                backmodel >> model0(i, 0);
        }
        backmodel.close();
    }
    else{
        std::cout << "?" << std::endl;
        exit(0);
    }
    
    model0.block(NN, 0, NN, 1) = model0.block(NN, 0, NN, 1).array().abs2().inverse();
    model0.block(3 * NN, 0, NN, 1) = model0.block(3 * NN, 0, NN, 1).array().abs2().inverse();
    model_true.block(NN, 0, NN, 1) = model_true.block(NN, 0, NN, 1).array().abs2().inverse();
    model_true.block(3 * NN, 0, NN, 1) = model_true.block(3 * NN, 0, NN, 1).array().abs2().inverse();

    std::cout << "Model done." << std::endl;

}

void Setup_par_VE::Get_Acq(){
    std::tie(sx, sz, rx, rz) = Set_Acq_Explosive(sAcq, soffset, rAcq, roffset, nz, nx);
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

    fwave = Eigen::MatrixXcf::Ones(freq.size(), S.cols());

    startfreq.resize(0); endfreq.resize(0);

    std::cout << "Freq end." << std::endl;
}

