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
            printf("On proc %d:\n", my_rank);
            for (const auto& row: a) {
                printf("|");
                for (auto v : row) {
                    printf("%2.3f ", v);
                }
                printf("|\n");
            }
            printf("wut?");
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
            {"nu", 0.1}
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
              nit = get<int>(default_args.at("nit"));

    const double dx = xLen / (nx - 1),
                 dy = yLen / (ny - 1);

    const int xStride = 1, yStride = 1;

    // initialize MPI variables for this rank
    int world_size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int my_nx = nx / world_size;
    if (nx % world_size != 0 && my_rank == world_size - 1) {
        my_nx += (nx % world_size); // handle un-whole division
    }
    int my_ny = ny;

    cout << "proc: " << my_rank <<  " \tglobal sizes ( " << nx << ", " << ny <<
        ")\tlocal sizes ( " << my_nx << ", " << my_ny << " )\n";

    // allocate 2D solution vectors
    vector<vector<double> > p(my_nx + 2, vector<double>(my_ny, 0.0));
    vector<vector<double> > u(my_nx + 2, vector<double>(my_ny, 0.0));
    vector<vector<double> > v(my_nx + 2, vector<double>(my_ny, 0.0));
    vector<vector<double> > b(my_nx + 2, vector<double>(my_ny, 0.0));

    runCavityFlowSim(
        p, u, v, b,
        nt, nit,
        world_size, my_rank,
        dx, dy, dt,
        rho, nu
    );

    printf("P array: \n");
    writeDistArray(p, my_rank, world_size);

    #ifdef CREATEPLOTS
        MPI_Barrier(MPI_COMM_WORLD);
        printDownSampled(p, 'p', my_rank, world_size, xStride, yStride, nx, ny, xLen, yLen);
        MPI_Barrier(MPI_COMM_WORLD);
        printDownSampled(u, 'u', my_rank, world_size, xStride, yStride, nx, ny, xLen, yLen);
        MPI_Barrier(MPI_COMM_WORLD);
        printDownSampled(v, 'v', my_rank, world_size, xStride, yStride, nx, ny, xLen, yLen);
        MPI_Barrier(MPI_COMM_WORLD);
        flowPlot("./results/nsStep11", "Cavity Flow Solution");
    #endif

    MPI_Finalize();
    return 0;
}

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
) {
    auto pn = p;
    auto un = u;
    auto vn = v;
    MPI_Status status;

    // run simulation for nt time steps
    for (int i = 0; i < nt; i++) {
        u.swap(un);
        v.swap(vn);

        // solve for the component of p that depends solely on u and v
        comp_b(b, un, vn, dx, dy, dt, rho);
        MPI_Barrier(MPI_COMM_WORLD);
        update_halos(b, my_rank, world_size, status);

        // iteratively solve for pressure
        for (int p_iter = 0; p_iter < nit; p_iter++) {
            p.swap(pn);
            p_np1(p, pn, b, dx, dy);
            MPI_Barrier(MPI_COMM_WORLD);
            update_halos(p, my_rank, world_size, status);
        }

        // solve for u and v using the updated pressure values
        u_np1(u, un, vn, p, dx, dy, dt, rho, nu);
        v_np1(v, un, vn, p, dx, dy, dt, rho, nu);

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
    const double dx,
    const double dy,
    const double dt,
    const double rho
) {
    double du, dv;

    #pragma omp parallel for default(none) shared(b, u, v) firstprivate(dx, dy, dt, rho) private(du, dv) collapse(2)
    for (int i = 1; i < b.size() - 1; i++) {
        for (int j = 1; j < b[0].size() - 1; j++) {
            du = u[i][j+1] - u[i][j-1];
            dv = u[i+1][j] - u[i-1][j];

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
    const double dx,
    const double dy
) {
    #pragma omp parallel for default(none) shared(p, pn, b) firstprivate(dx, dy) collapse(2)
    for (int i = 1; i < p.size() - 1; i++) {
        for (int j = 1; j < p[0].size() - 1; j++) {
            p[i][j] = ((
                    pow(dy, 2) * (pn[i][j+1] + pn[i][j-1]) +
                    pow(dx, 2) * (pn[i+1][j] + pn[i-1][j])
                ) / (2.0*(pow(dx, 2) + pow(dy, 2))) - pow(dx, 2) * pow(dy, 2) /
                    (2.0*(pow(dx, 2) + pow(dy, 2))) * b[i][j]);
        }
    }
}

void u_np1(
    vector<vector<double> >& u,
    vector<vector<double> > const& un,
    vector<vector<double> > const& vn,
    vector<vector<double> > const& pn,
    const double dx,
    const double dy,
    const double dt,
    const double rho,
    const double nu

) {
    #pragma omp parallel for default(none) shared(u, un, vn, pn) firstprivate(dx, dy, dt, rho, nu) collapse(2)
    for (int i = 1; i < u.size() - 1; i++) {
        for (int j = 1; j < u[0].size() - 1; j++) {
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
    const double dx,
    const double dy,
    const double dt,
    const double rho,
    const double nu
) {
    #pragma omp parallel for default(none) shared(v, un, vn, pn) firstprivate(dx, dy, dt, rho, nu) collapse(2)
    for (int i = 1; i < v.size() - 1; i++) {
        for (int j = 1; j < v[0].size() - 1; j++) {
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
    int ny,
    double xLen,
    double yLen
) {
    vector<vector<double>> global_ptr;

    if (my_rank != 0) {
        vector<vector<double>> empty_global;
        downSampleAndGather(a, empty_global, my_rank, world_size, xStride, yStride, nx, ny);
    } else {
        vector<vector<double>> globalA(nx / xStride, vector<double>(ny / yStride, 0.0));
        downSampleAndGather(a, globalA, my_rank, world_size, xStride, yStride, nx, ny);
        global_ptr.swap(globalA);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (my_rank == 0) {
        string file_path = "./results/nsStep11_"; file_path.push_back(name);
        printForPlot(global_ptr, file_path, xLen, yLen);
    }
}

void downSampleAndGather(
    vector<vector<double>>& a,
    vector<vector<double>>& a_global,
    int my_rank,
    int world_size,
    int xStride,
    int yStride,
    int nx,
    int ny
) {
    // sizes of global and local strided arrays
    int sizes[2] = { static_cast<int>(nx / xStride),  static_cast<int>(ny / yStride) };
    int subsizes[2] = { static_cast<int>((a.size() - 2) / xStride),  static_cast<int>(ny / yStride) };
    int starts[2] = { 0, 0 };
    MPI_Datatype columnSize, columnOffset;

    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &columnSize);
    MPI_Type_create_resized(columnSize, 0, subsizes[0]*sizeof(double), &columnOffset);
    MPI_Type_commit(&columnOffset);

    // fill the local strided array with the sparse grid of values from 'a'
    vector<vector<double>> a_strided(subsizes[0], vector<double>(subsizes[1], 0.0));
    for (int i = 0; i < subsizes[0] - 1; i++) {
        for (int j = 0; j < subsizes[1] - 1; j++) {
            a_strided[i][j] = a[i * xStride][j * yStride];
        }
    }

    int sendcounts[world_size];
    int displs[world_size];
    double *global_ptr = NULL;

    if (my_rank == 0) {
        global_ptr = &(a_global[0][0]);
        for (int i=0; i<world_size; i++) {
            sendcounts[i] = 1;
            displs[i] = i;
        }
    }

    // gather the local strided arrays into the global strided array
    MPI_Gatherv(
        &(a_strided[0][0]),
        subsizes[1],
        MPI_DOUBLE,
        global_ptr,
        sendcounts,
        displs,
        columnOffset,
        0,
        MPI_COMM_WORLD
    );

    MPI_Type_free(&columnOffset);
}
