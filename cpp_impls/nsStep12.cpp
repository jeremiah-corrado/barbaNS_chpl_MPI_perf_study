#include "nsStep12.h"

int main(int argc, char *argv[]) {
        MPI_Init(&argc, &argv);

    // define default values of constants
    unordered_map<string, variant<int, double>> default_args = {
            {"xLen", 2.0},
            {"yLen", 2.0},
            {"nx", 41},
            {"ny", 41},
            {"nt", 500},
            {"nit", 50},
            {"dt", 0.001},
            {"rho", 1.0},
            {"nu", 0.1},
            {"F", 1.0},
            {"xStride", 2},
            {"yStride", 2}
    };
    // overwrite defaults with command line arguments if present
    parseArgsWithDefaults(argc, argv, default_args);

    const double xLen = get<double>(default_args.at("xLen")),
                 yLen = get<double>(default_args.at("yLen")),
                 dt   = get<double>(default_args.at("dt")),
                 rho  = get<double>(default_args.at("rho")),
                 nu   = get<double>(default_args.at("nu")),
                 F    = get<double>(default_args.at("F"));
    const int nx  = get<int>(default_args.at("nx")),
              ny  = get<int>(default_args.at("ny")),
              nt  = get<int>(default_args.at("nt")),
              nit = get<int>(default_args.at("nit")),
              xStride = get<int>(default_args.at("xStride")),
              yStride = get<int>(default_args.at("yStride"));

    const double dx = xLen / (nx - 1),
                 dy = yLen / (ny - 1),
                 dxy2 = 2.0 * (pow(dx, 2) + pow(dy, 2));

    // initialize MPI variables for this rank
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // define the sizes of the local sub-arrays - not including halos (giving extra elements to rank 0)
    const int local_sizes[2] = {((nx%size != 0 && rank == 0) ? (nx/size) + (nx%size) : nx / size), ny};

    // define the iteration bounds (handling edge cases - halos on the left and right of the domain are unused)
    const int local_iter_ranges[2][2] = {
        { 1, local_sizes[0] + 1 },
        { 1, local_sizes[1] - 1 }
    };

    vector<vector<double>> p(local_sizes[0] + 2, vector<double>(local_sizes[1], 0.0));
    vector<vector<double>> u(local_sizes[0] + 2, vector<double>(local_sizes[1], 0.0));
    vector<vector<double>> v(local_sizes[0] + 2, vector<double>(local_sizes[1], 0.0));
    vector<vector<double>> b(local_sizes[0] + 2, vector<double>(local_sizes[1], 0.0));

    runChannelFlowSim(p, u, v, b, local_iter_ranges, nt, nit, size, rank, dx, dy, dxy2, dt, rho, nu, F);

    // print results
    #ifdef CREATEPLOTS
        printDownSampled(p, 'p', "results/nsStep12", rank, size, xStride, yStride, nx, ny);
        printDownSampled(u, 'u', "results/nsStep12", rank, size, xStride, yStride, nx, ny);
        printDownSampled(v, 'v', "results/nsStep12", rank, size, xStride, yStride, nx, ny);
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) flowPlot("results/nsStep12", "Channel Flow Solution", nx / xStride, ny / yStride, xLen, yLen);
    #endif

    MPI_Finalize();
    return 0;
}

void runChannelFlowSim(
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
    const double nu,
    const double F
) {
    auto pn = p;
    auto un = u;
    auto vn = v;
    MPI_Status status;

    // run simulation for nt time steps
    for (int i = 0; i < nt; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        u.swap(un);
        v.swap(vn);

        // solve for the component of p that depends solely on u and v
        comp_b(b, un, vn, ranges, dx, dy, dt, rho);
        MPI_Barrier(MPI_COMM_WORLD);
        update_halos(b, my_rank, world_size, status);
        MPI_Barrier(MPI_COMM_WORLD);

        // iteratively solve for pressure
        for (int p_iter = 0; p_iter < nit; p_iter++) {
            p.swap(pn);
            p_np1(p, pn, b, ranges, dx, dy, dxy2);
            p_boundary(p, my_rank);
            MPI_Barrier(MPI_COMM_WORLD);
            update_halos(p, my_rank, world_size, status);
            update_halos(pn, my_rank, world_size, status);
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // solve for u and v using the updated pressure values
        u_np1(u, un, vn, p, ranges, dx, dy, dt, rho, nu, F);
        v_np1(v, un, vn, p, ranges, dx, dy, dt, rho, nu);

        // solve for u and v using the updated pressure values
        MPI_Barrier(MPI_COMM_WORLD);
        update_halos(u, my_rank, world_size, status);
        update_halos(v, my_rank, world_size, status);
        update_halos(un, my_rank, world_size, status);
        update_halos(vn, my_rank, world_size, status);
    }
}

void comp_b(
    vector<vector<double> >& b,
    vector<vector<double> > const& u,
    vector<vector<double> > const& v,
    const int ranges[2][2],
    const double dx,
    const double dy,
    const double dt,
    const double rho
) {
    double du, dv;

    #pragma omp parallel for default(none) shared(b, u, v, ranges) \
        firstprivate(dx, dy, dt, rho) private(du, dv) collapse(2)
    for (int i = ranges[0][0]; i < ranges[0][1]; i++) {
        for (int j = ranges[1][0]; j < ranges[1][1]; j++) {
            du = u[i][j+1] - u[i][j-1];
            dv = v[i+1][j] - v[i-1][j];

            b[i][j] = rho * (1.0 / dt) *
                (du / (2.0 * dx) + dv / (2.0 * dy)) -
                pow((du / (2.0 * dx)), 2) -
                pow((dv / (2.0 * dy)), 2) -
                2.0 * (
                    (u[i+1][j] - u[i-1][j]) / (2.0 * dy) *
                    (v[i][j+1] - v[i][j-1]) / (2.0 * dx)
                );
        }
    }
}

void p_np1(
    vector<vector<double> >& p,
    vector<vector<double> > const& pn,
    vector<vector<double> > const& b,
    const int ranges[2][2],
    const double dx,
    const double dy,
    const double dxy2
) {
    #pragma omp parallel for default(none) shared(p, pn, b, ranges) \
        firstprivate(dx, dy, dxy2) collapse(2)
    for (int i = ranges[0][0]; i < ranges[0][1]; i++) {
        for (int j = ranges[1][0]; j < ranges[1][1]; j++) {
            p[i][j] = (
                    pow(dy, 2) * (pn[i][j+1] + pn[i][j-1]) +
                    pow(dx, 2) * (pn[i+1][j] + pn[i-1][j])
                ) / dxy2 - pow(dx, 2) * pow(dy, 2) / dxy2 * b[i][j];
        }
    }
}

void p_boundary(vector<vector<double>>& p, int my_rank) {
    // top/bottom walls
    #pragma omp parallel for default(none) shared(p)
    for (int i = 0; i < p.size(); i++) {
        p[i][0] = p[i][1];
        p[i][p[0].size()-1] = p[i][p[0].size()-2];
    }
}

void u_np1(
    vector<vector<double> >& u,
    vector<vector<double> > const& un,
    vector<vector<double> > const& vn,
    vector<vector<double> > const& pn,
    const int ranges[2][2],
    const double dx,
    const double dy,
    const double dt,
    const double rho,
    const double nu,
    const double F
) {
    #pragma omp parallel for default(none) shared(u, un, vn, pn, ranges) \
        firstprivate(dx, dy, dt, rho, nu, F) collapse(2)
    for (int i = ranges[0][0]; i < ranges[0][1]; i++) {
        for (int j = ranges[1][0]; j < ranges[1][1]; j++) {
            u[i][j] = un[i][j] -
                un[i][j] * (dt / dx) * (un[i][j] - un[i][j-1]) -
                vn[i][j] * (dt / dy) * (un[i][j] - un[i-1][j]) -
                dt / (2.0 * rho * dx) * (pn[i][j+1] - pn[i][j-1]) +
                nu * (
                    (dt / pow(dx, 2)) * (un[i+1][j] - 2.0 * un[i][j] + un[i-1][j]) +
                    (dt / pow(dy, 2)) * (un[i][j+1] - 2.0 * un[i][j] + un[i][j-1])
                ) + F * dt;
        }
    }
}

void v_np1(
    vector<vector<double> >& v,
    vector<vector<double> > const& un,
    vector<vector<double> > const& vn,
    vector<vector<double> > const& pn,
    const int ranges[2][2],
    const double dx,
    const double dy,
    const double dt,
    const double rho,
    const double nu
) {
    #pragma omp parallel for default(none) shared(v, un, vn, pn, ranges) \
        firstprivate(dx, dy, dt, rho, nu) collapse(2)
    for (int i = ranges[0][0]; i < ranges[0][1]; i++) {
        for (int j = ranges[1][0]; j < ranges[1][1]; j++) {
            v[i][j] = vn[i][j] -
                un[i][j] * (dt / dx) * (vn[i][j] - vn[i][j-1]) -
                vn[i][j] * (dt / dy) * (vn[i][j] - vn[i-1][j]) -
                dt / (2.0 * rho * dy) * (pn[i+1][j] - pn[i-1][j]) +
                nu * (
                    (dt / pow(dx, 2)) * (vn[i+1][j] - 2.0 * vn[i][j] + vn[i-1][j]) +
                    (dt / pow(dy, 2)) * (vn[i][j+1] - 2.0 * vn[i][j] + vn[i][j-1])
                );
        }
    }
}


// copy the edges of each sub-array into the neighboring rank's halos
//  wrap the leftmost and rightmost halos around the domain (implementing the cyclic BC)
void update_halos(
    vector<vector<double> >& a,
    int my_rank, int world_size,
    MPI_Status& status
) {
    const int num_y = a[0].size();

    // send right edge to the right
    MPI_Send(&a[a.size() - 2][0], num_y, MPI_DOUBLE, (my_rank+1) % world_size, 1, MPI_COMM_WORLD);

    // receive left halo from the left
    MPI_Recv(&a[0][0], num_y, MPI_DOUBLE, (my_rank==0) ? world_size - 1 : my_rank - 1, 1, MPI_COMM_WORLD, &status);

    // send left edge to the left
    MPI_Send(&a[1][0], num_y, MPI_DOUBLE, (my_rank==0) ? world_size - 1 : my_rank - 1, 2, MPI_COMM_WORLD);

    // receive right halo from the right
    MPI_Recv(&a[a.size() - 1][0], num_y, MPI_DOUBLE, (my_rank+1) % world_size, 2, MPI_COMM_WORLD, &status);
}
