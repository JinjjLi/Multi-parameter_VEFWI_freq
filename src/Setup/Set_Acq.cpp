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
    std::cout << "S num for one dimension: " << sx_num << std::endl;

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
    case 1:
        {
            sx_sur = sx_hor;
            sz_sur = sz_hor1;

            std::cout << "sx:" << std::endl;
            std::cout << sx_sur << std::endl;
            std::cout << "sz:" << std::endl;
            std::cout << sz_sur << std::endl;
        }
        break;
    case 2:
        {
            sx_sur = sx_vert1;
            sz_sur = sz_vert;

            std::cout << "sx:" << std::endl;
            std::cout << sx_sur << std::endl;
            std::cout << "sz:" << std::endl;
            std::cout << sz_sur << std::endl;
        }
        break;
    case 3:
        {
            sx_sur = sx_vert2;
            sz_sur = sz_vert;

            std::cout << "sx:" << std::endl;
            std::cout << sx_sur << std::endl;
            std::cout << "sz:" << std::endl;
            std::cout << sz_sur << std::endl;
        }
        break;
    case 4:
        {
            sx_sur = sx_hor;
            sz_sur = sz_hor2;

            std::cout << "sx:" << std::endl;
            std::cout << sx_sur << std::endl;
            std::cout << "sz:" << std::endl;
            std::cout << sz_sur << std::endl;
        }
        break;
    case 5:
        {
            sz_sur = Eigen::RowVectorXf(sz_hor1.cols() + sz_vert.cols());
            sx_sur = Eigen::RowVectorXf(sx_hor.cols() + sx_vert1.cols());

            sz_sur << sz_hor1, sz_vert;
            sx_sur << sx_hor, sx_vert1;

            std::cout << "sx:" << std::endl;
            std::cout << sx_sur << std::endl;
            std::cout << "sz:" << std::endl;
            std::cout << sz_sur << std::endl;
        }
        break;
    case 6:
        {
            sz_sur = Eigen::RowVectorXf(sz_hor1.cols() + sz_vert.cols());
            sx_sur = Eigen::RowVectorXf(sx_hor.cols() + sx_vert2.cols());

            sz_sur << sz_hor1, sz_vert;
            sx_sur << sx_hor, sx_vert2;

            std::cout << "sx:" << std::endl;
            std::cout << sx_sur << std::endl;
            std::cout << "sz:" << std::endl;
            std::cout << sz_sur << std::endl;
        }
        break;
    case 7:
        {
            sz_sur = Eigen::RowVectorXf(sz_hor1.cols() + sz_hor2.cols());
            sx_sur = Eigen::RowVectorXf(sx_hor.cols() + sx_hor.cols());

            sz_sur << sz_hor1, sz_hor2;
            sx_sur << sx_hor, sx_hor;

            std::cout << "sx:" << std::endl;
            std::cout << sx_sur << std::endl;
            std::cout << "sz:" << std::endl;
            std::cout << sz_sur << std::endl;
        }
        break;
    case 8:
        {
            sz_sur = Eigen::RowVectorXf(sz_vert.cols() + sz_vert.cols());
            sx_sur = Eigen::RowVectorXf(sx_vert1.cols() + sx_vert2.cols());

            sz_sur << sz_vert, sz_vert;
            sx_sur << sx_vert1, sx_vert2;

            std::cout << "sx:" << std::endl;
            std::cout << sx_sur << std::endl;
            std::cout << "sz:" << std::endl;
            std::cout << sz_sur << std::endl;
        }
        break;
    case 9:
        {
            sz_sur = Eigen::RowVectorXf(sz_vert.cols() + sz_vert.cols() + sz_hor1.cols() + sz_hor2.cols());
            sx_sur = Eigen::RowVectorXf(sx_vert1.cols() + sx_vert2.cols() + sx_hor.cols() + sx_hor.cols());

            sz_sur << sz_vert, sz_vert, sz_hor1, sz_hor2;
            sx_sur << sx_vert1, sx_vert2, sx_hor, sx_hor;

            std::cout << "sx:" << std::endl;
            std::cout << sx_sur << std::endl;
            std::cout << "sz:" << std::endl;
            std::cout << sz_sur << std::endl;
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
    case 2:
        {
            rx = rx_vert1;
            rz = rz_vert;
        }
        break;
    case 3:
        {
            rx = rx_vert2;
            rz = rz_vert;
        }
        break;
    case 4:
        {
            rx = rx_hor;
            rz = rz_hor2;
        }
        break;
    case 5:
        {
            rx = Eigen::RowVectorXf(rx_hor.cols() + rx_vert1.cols());
            rz = Eigen::RowVectorXf(rz_hor1.cols() + rz_vert.cols());
            rx << rx_hor, rx_vert1;
            rz << rz_hor1, rz_vert;
        }
        break;
    case 6:
        {
            rx = Eigen::RowVectorXf(rx_hor.cols() + rx_vert2.cols());
            rz = Eigen::RowVectorXf(rz_hor1.cols() + rz_vert.cols());
            rx << rx_hor, rx_vert2;
            rz << rz_hor1, rz_vert;
        }
        break;
    case 7:
        {
            rx = Eigen::RowVectorXf(rx_hor.cols() + rx_hor.cols());
            rz = Eigen::RowVectorXf(rz_hor1.cols() + rz_hor2.cols());
            rx << rx_hor, rx_hor;
            rz << rz_hor1, rz_hor2;
        }
        break;
    case 8:
        {
            rx = Eigen::RowVectorXf(rx_vert1.cols() + rx_vert2.cols());
            rz = Eigen::RowVectorXf(rz_vert.cols() + rz_vert.cols());
            rx << rx_vert1, rx_vert2;
            rz << rz_vert, rz_vert;
        }
        break;
    case 9:
        {
            rx = Eigen::RowVectorXf(rx_vert1.cols() + rx_vert2.cols() + rx_hor.cols() + rx_hor.cols());
            rz = Eigen::RowVectorXf(rz_vert.cols() + rz_vert.cols() + rz_hor1.cols() + rz_hor2.cols());
            rx << rx_vert1, rx_vert2, rx_hor, rx_hor;
            rz << rz_vert, rz_vert, rz_hor1, rz_hor2;
        }
        break;
    }
    sx_hor.resize(0); sx_vert1.resize(0); sx_vert2.resize(0);
    sz_vert.resize(0); sz_hor1.resize(0); sz_hor2.resize(0);
    rx_hor.resize(0); rz_hor1.resize(0); rz_hor2.resize(0);
    rz_vert.resize(0); rx_vert1.resize(0); rx_vert2.resize(0);
    return std::make_tuple(sx_sur, sx_SWD, sz_sur, sz_SWD, rx, rz);
}

