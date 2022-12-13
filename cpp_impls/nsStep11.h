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
    const int nt,
    const int nit,
    const int world_size,
    const int my_rank,
    const double dx,
    const double dy,
    const double dt,
    const double rho,
    const double nu
);

void comp_b(
    vector<vector<double> >& b,
    vector<vector<double> > const& u,
    vector<vector<double> > const& v,
    const double dx,
    const double dy,
    const double dt,
    const double rho
);

void p_np1(
    vector<vector<double> >& p,
    vector<vector<double> > const& pn,
    vector<vector<double> > const& b,
    const double dx,
    const double dy
);

void u_np1(
    vector<vector<double> >& u,
    vector<vector<double> > const& un,
    vector<vector<double> > const& vn,
    vector<vector<double> > const& p,
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
