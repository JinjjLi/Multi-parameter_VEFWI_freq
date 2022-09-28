#include "../General/includefile.h"
#include "Set_Acq.h"

std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf,\
        Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf>\
Set_Acq(int sAcq, int soffset, int rAcq, int roffset, int nz, int nx)
{
    int s_interx = 2; int s_interz = 2;
    int s_initialx = 2; int s_initialz = 2;
    int s_endx = nx - 2; int s_endz = nz - 2;

    int sx_num = (s_endx - soffset - s_initialx) / s_interx + 1;
    std::cout << "S num: " << sx_num << std::endl;
    //if (sx_num == 1)
    //    Eigen::RowVectorXf sx_hor = Eigen::RowVectorXf::Constant(sx_num, s_initialx + soffset).array() - 1;
    //else
    Eigen::RowVectorXf sx_hor = Eigen::RowVectorXf::LinSpaced(sx_num, soffset + s_initialx, s_endx).array() - 1;
    Eigen::RowVectorXf sz_hor1 = Eigen::RowVectorXf::Constant(sx_num, 3).array() - 1;
    Eigen::RowVectorXf sz_hor2 = Eigen::RowVectorXf::Constant(sx_num, nz - 2).array() - 1;

    int offset_z_vert = 0;
    int sz_num = (s_endz - offset_z_vert - s_initialz) / s_interz + 1;
    Eigen::RowVectorXf sz_vert = Eigen::RowVectorXf::LinSpaced(sz_num, offset_z_vert + s_initialz, s_endz).array() - 1;
    Eigen::RowVectorXf sx_vert1 = Eigen::RowVectorXf::Constant(sz_num, 2).array() - 1;
    Eigen::RowVectorXf sx_vert2 = Eigen::RowVectorXf::Constant(sz_num, nx - 3).array() - 1;

    int r_interx = 1; int r_interz = 1;
    int r_initialx = 2; int r_initialz = 2;
    int r_endx = nx - 1; int r_endz = nz - 1;

    int rx_num = (r_endx - roffset - r_initialx) / r_interx + 1;
    std::cout << "R num: " << rx_num << std::endl;
    Eigen::RowVectorXf rx_hor = Eigen::RowVectorXf::LinSpaced(rx_num, roffset + r_initialx, r_endx).array() - 1;
    Eigen::RowVectorXf rz_hor1 = Eigen::RowVectorXf::Constant(rx_num, 1).array() - 1;
    Eigen::RowVectorXf rz_hor2 = Eigen::RowVectorXf::Constant(rx_num, nz).array() - 1;

    int offset_rz_vert = 0;
    int rz_num = (r_endz - offset_rz_vert - r_initialz) / r_interz + 1;
    Eigen::RowVectorXf rz_vert = Eigen::RowVectorXf::LinSpaced(rz_num, roffset + r_initialz, r_endz).array() - 1;
    Eigen::RowVectorXf rx_vert1 = Eigen::RowVectorXf::Constant(rz_num, 1).array() - 1;
    Eigen::RowVectorXf rx_vert2 = Eigen::RowVectorXf::Constant(rz_num, nx).array() - 1;

    Eigen::RowVectorXf sx_sur, sz_sur, sx_SWD, sz_SWD;
    switch(sAcq){
    case 11:
        {
            int SWD_source_num = 10;
            int horSWD_startx = 5; int horSWD_endx = 9;
            int horSWD_startz = 20; int horSWD_endz = 30;
            Eigen::RowVectorXf sx_hor1_SWD = Eigen::RowVectorXf::LinSpaced(SWD_source_num, horSWD_startx, horSWD_endx);
            Eigen::RowVectorXf sz_vert1_SWD = Eigen::RowVectorXf::LinSpaced(SWD_source_num, horSWD_startz, horSWD_endz);                                                           

            sx_sur = sx_hor;
            sz_sur = sz_hor1;
            sx_SWD = sx_hor1_SWD;
            sz_SWD = sz_vert1_SWD;

            std::cout << "sx:" << std::endl;
            std::cout << sx_sur << sx_SWD << std::endl;
            std::cout << "sz:" << std::endl;
            std::cout << sz_sur << sz_SWD << std::endl;
            sx_hor1_SWD.resize(0); sz_vert1_SWD.resize(0);
        }
        break;
    }
    Eigen::RowVectorXf rz, rx; 
    switch(rAcq){
    case 1:
        {
            rz = rz_hor1;
            rx = rx_hor;
        }
        break;
    }
    sx_hor.resize(0); sx_vert1.resize(0); sx_vert2.resize(0);
    sz_vert.resize(0); sz_hor1.resize(0); sz_hor2.resize(0);
    rx_hor.resize(0); rz_hor1.resize(0); rz_hor2.resize(0);
    rz_vert.resize(0); rx_vert1.resize(0); rx_vert2.resize(0);
    std::cout << "Acq done." << std::endl;
    return std::make_tuple(sx_sur, sx_SWD, sz_sur, sz_SWD, rx, rz);
}

