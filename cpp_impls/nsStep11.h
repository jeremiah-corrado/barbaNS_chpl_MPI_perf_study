#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <variant>
#include <unordered_map>
#include <algorithm>

#include <mpi.h>
#include <omp.h>
#include "utils.h"

void runCavityFlowSim(
    vector<vector<double> >& p,
    vector<vector<double> >& u,
    vector<vector<double> >& v,
    vector<vector<double> >& b,
    const int ranges[2][2],
    const int nt,
    const int nit,
    const int world_size,
    const int my_rank,
    const double dx,
    const double dy,
    const double dxy2,
    const double dt,
    const double rho,
    const double nu
);

void comp_b(
    vector<vector<double> >& b,
    vector<vector<double> > const& u,
    vector<vector<double> > const& v,
    const int ranges[2][2],
    const double dx,
    const double dy,
    const double dt,
    const double rho
);

void p_np1(
    vector<vector<double> >& p,
    vector<vector<double> > const& pn,
    vector<vector<double> > const& b,
    const int ranges[2][2],
    const double dx,
    const double dy,
    const double dxy2
);

void p_boundary(vector<vector<double>>& p, int my_rank);

void u_np1(
    vector<vector<double> >& u,
    vector<vector<double> > const& un,
    vector<vector<double> > const& vn,
    vector<vector<double> > const& p,
    const int ranges[2][2],
    const double dx,
    const double dy,
    const double dt,
    const double rho,
    const double nu
);

void v_np1(
    vector<vector<double> >& v,
    vector<vector<double> > const& un,
    vector<vector<double> > const& vn,
    vector<vector<double> > const& p,
    const int ranges[2][2],
    const double dx,
    const double dy,
    const double dt,
    const double rho,
    const double nu
);

void update_halos(
    vector<vector<double> >& a,
    int my_rank, int world_size,
    MPI_Status& status
);

void printDownSampled(
    vector<vector<double>>& a,
    char name,
    int my_rank,
    int world_size,
    int xStride, int yStride,
    int nx, int ny
);

void downSampleAndGather(
    vector<vector<double>>& a,
    vector<double>& a_global,
    const int my_rank,
    int world_size,
    int xStride, int yStride,
    int nx, int ny
);
