#include "nsStep11.h"

/*

    Distributed matrix layout:

    | ------------ (nx) ------------|

    |(my-nx)|

    *-------*-------*  ...  *-------*   --
    |       |       |       |       |   |
    |       |       |       |       |   |
    |       |       |       |       |   |
    |       |       |       |       |   |
    |  b0   |  b1   |  ...  |   bn  |   (ny == my_ny)
    |       |       |       |       |   |
    |       |       |       |       |   |
    |       |       |       |       |   |
  ^ |       |       |       |       |   |
  y *-------*-------*  ...  *-------*   --
    x ->

*/

void writeDistArray(vector<vector<double>>& a, int my_rank, int world_size) {
    for (int p = 0; p < world_size; p++) {
        if (my_rank == p) {
            printf("On proc %d:\n", p);
            for (int r = 0; r < a[0].size(); r++) {
                printf("[%2.3f | ", a[0][r]);
                for (int c = 1; c < a.size() - 1; c++) {
                    printf("%2.3f ", a[c][r]);
                }
                printf("| %2.3f]\n", a[a.size()-1][r]);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

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
            {"xStride", 2},
            {"yStride", 2}
    };
    // overwrite defaults with command line arguments if present
    parseArgsWithDefaults(argc, argv, default_args);

    const double xLen = get<double>(default_args.at("xLen")),
                 yLen = get<double>(default_args.at("yLen")),
                 dt   = get<double>(default_args.at("dt")),
                 rho  = get<double>(default_args.at("rho")),
                 nu   = get<double>(default_args.at("nu"));
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

    // define the sizes of various arrays along x and y (giving any extra elements to rank 0)
    const int global_sizes   [2] = {nx, ny},
              local_sizes    [2] = {((nx%size != 0 && rank == 0) ? (nx/size) + (nx%size) : nx / size), ny},
              global_strided [2] = {nx / xStride, ny / yStride},
              local_strided  [2] = {local_sizes[0] / xStride, local_sizes[1] / yStride};

    // define the iteration bounds (handling edge cases - halos on the left and right are unused)
    const int local_iter_ranges[2][2] = {
        { (rank == 0) ? 2 : 1, local_sizes[0] + ((rank == size-1) ? 0 : 1) },
        { 1, local_sizes[1] - 1 }
    };

    printf(
        "From rank %d : local size [%d, %d] \t global size [%d, %d] \t i-range [%d - %d] \t j-range [%d - %d]\n",
        rank,
        local_sizes[0], local_sizes[1],
        global_sizes[0], global_sizes[1],
        local_iter_ranges[0][0], local_iter_ranges[0][1],
        local_iter_ranges[1][0], local_iter_ranges[1][1]
    );

    vector<vector<double>> p(local_sizes[0] + 2, vector<double>(local_sizes[1], 0.0));
    vector<vector<double>> u(local_sizes[0] + 2, vector<double>(local_sizes[1], 0.0));
    vector<vector<double>> v(local_sizes[0] + 2, vector<double>(local_sizes[1], 0.0));
    vector<vector<double>> b(local_sizes[0] + 2, vector<double>(local_sizes[1], 0.0));

    // u-boundary on right wall
    if (rank == size - 1) {
        for(int j = 0; j < local_sizes[1]; j++) {
            // u[u.size()-1] would be inside the unused halo
            u[u.size()-2][j] = 1.0;
        }
    }

    // writeDistArray(p, rank, size);
    runCavityFlowSim(p, u, v, b, local_iter_ranges, nt, nit, size, rank, dx, dy, dxy2, dt, rho, nu);
    writeDistArray(p, rank, size);

    printDownSampled(p, 'p', rank, size, xStride, yStride, nx, ny);
    printDownSampled(u, 'u', rank, size, xStride, yStride, nx, ny);
    printDownSampled(v, 'v', rank, size, xStride, yStride, nx, ny);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) flowPlot("results/nsStep11", "Cavity Flow Solution", global_strided[0], global_strided[1], xLen, yLen);

    MPI_Finalize();
    return 0;
}

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
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // solve for u and v using the updated pressure values
        u_np1(u, un, vn, p, ranges, dx, dy, dt, rho, nu);
        v_np1(v, un, vn, p, ranges, dx, dy, dt, rho, nu);

        // solve for u and v using the updated pressure values
        MPI_Barrier(MPI_COMM_WORLD);
        update_halos(u, my_rank, world_size, status);
        update_halos(v, my_rank, world_size, status);
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

    // #pragma omp parallel for default(none) shared(b, u, v, ranges) \
    //     firstprivate(dx, dy, dt, rho) private(du, dv) collapse(2)
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
    // #pragma omp parallel for default(none) shared(p, pn, b, ranges) \
    //     firstprivate(dx, dy, dxy2) collapse(2)
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
    // left wall
    if (my_rank == 0) {
        // #pragma omp parallel for default(none) shared(p)
        for (int j = 0; j < p[0].size(); j++) {
            p[1][j] = p[2][j];
        }
    }
    // top/bottom walls
    // #pragma omp parallel for default(none) shared(p)
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
    const double nu

) {
    // #pragma omp parallel for default(none) shared(u, un, vn, pn, ranges) \
    //     firstprivate(dx, dy, dt, rho, nu) collapse(2)
    for (int i = ranges[0][0]; i < ranges[0][1]; i++) {
        for (int j = ranges[1][0]; j < ranges[1][1]; j++) {
            u[i][j] = un[i][j] -
                un[i][j] * (dt / dx) * (un[i][j] - un[i][j-1]) -
                vn[i][j] * (dt / dy) * (un[i][j] - un[i-1][j]) -
                dt / (2.0 * rho * dx) * (pn[i][j+1] - pn[i][j-1]) +
                nu * (
                    (dt / pow(dx, 2)) * (un[i+1][j] - 2.0 * un[i][j] + un[i-1][j]) +
                    (dt / pow(dy, 2)) * (un[i][j+1] - 2.0 * un[i][j] + un[i][j-1])
                );
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
    // #pragma omp parallel for default(none) shared(v, un, vn, pn, ranges) \
    //     firstprivate(dx, dy, dt, rho, nu) collapse(2)
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

void update_halos(
    vector<vector<double> >& a,
    int my_rank, int world_size,
    MPI_Status& status
) {
    const int num_y = a[0].size();

    // send right edge to the right
    if (my_rank < world_size - 1) {
        MPI_Send(&a[a.size() - 2][0], num_y, MPI_DOUBLE, my_rank + 1, 1, MPI_COMM_WORLD);
    }
    // receive left halo from the left
    if (my_rank > 0) {
        MPI_Recv(&a[0][0], num_y, MPI_DOUBLE, my_rank -1, 1, MPI_COMM_WORLD, &status);
    }

    // send left edge to the left
    if (my_rank > 0) {
        MPI_Send(&a[1][0], num_y, MPI_DOUBLE, my_rank - 1, 2, MPI_COMM_WORLD);
    }
    // receive right halo from the right
    if (my_rank < world_size - 1) {
        MPI_Recv(&a[a.size() - 1][0], num_y, MPI_DOUBLE, my_rank + 1, 2, MPI_COMM_WORLD, &status);
    }
}

void printDownSampled(
    vector<vector<double>>& a,
    char name,
    int my_rank,
    int world_size,
    int xStride,
    int yStride,
    int nx,
    int ny
) {
    vector<double> global_ptr;

    if (my_rank != 0) {
       vector<double> empty_global;
        downSampleAndGather(a, empty_global, my_rank, world_size, xStride, yStride, nx, ny);
    } else {
        vector<double> globalA((nx / xStride) * (ny / yStride), 0.0);
        downSampleAndGather(a, globalA, my_rank, world_size, xStride, yStride, nx, ny);
        global_ptr.swap(globalA);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (my_rank == 0) {
        string file_path = "./results/nsStep11_"; file_path.push_back(name);
        printForPlot(global_ptr, file_path, ny / yStride);
    }
}

void downSampleAndGather(
    vector<vector<double>>& a,
    vector<double>& a_global,
    int my_rank,
    int world_size,
    int xStride,
    int yStride,
    int nx,
    int ny
) {
    int subsizes[2] = { static_cast<int>((a.size() - 2) / xStride),  static_cast<int>(ny / yStride) };

    // fill the local strided array with the sparse grid of values from 'a'
    // vector<vector<double>> a_strided(subsizes[0], vector<double>(subsizes[1], 0.0));

    // using a flat array s.t. values are stored in contiguous memory (required for MPI_Gather)
    vector<double> a_strided(subsizes[0] * subsizes[1]);
    for (int i = 0; i < subsizes[0]; i++) {
        for (int j = 0; j < subsizes[1]; j++) {
            a_strided[i * subsizes[1] + j] = a[i * xStride][j * yStride];
        }
    }

    MPI_Gather(
        &a_strided[0],
        subsizes[0] * subsizes[1],
        MPI_DOUBLE,
        &a_global[0],
        subsizes[0] * subsizes[1],
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );
}
