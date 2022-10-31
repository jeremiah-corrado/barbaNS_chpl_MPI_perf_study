#include "nsStep11.h"


/*

    | ------------ (nx) ------------|

    |(my-nx)|

    *-------*-------*  ...  *-------*   --
    |       |       |       |       |   |
    |       |       |       |       |   |
    |       |       |       |       |   |
    |       |       |       |       |   |
    |  b0   |  b1   |  ...  |   bn  |   (ny) & (my_ny)
    |       |       |       |       |   |
    |       |       |       |       |   |
    |       |       |       |       |   |
  ^ |       |       |       |       |   |
  y *-------*-------*  ...  *-------*   --
    x ->

*/

int main(int argc, const char *argv[]) {
    MPI_Init(&argc, argv);

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

    // initialize MPI variables for this rank
    int world_size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int my_nx = nx / world_size;
    if (nx % world_size != 0 && my_rank == world_size - 1) {
        my_nx += (nx % world_size);
    }
    int my_ny = ny;

    // allocate 2D solution vectors
    vector<vector<double> > p(my_nx + 2, vector<double>(my_ny, 0.0));
    vector<vector<double> > u(my_nx + 2, vector<double>(my_ny, 0.0));
    vector<vector<double> > v(my_nx + 2, vector<double>(my_ny, 0.0));
    vector<vector<double> > b(my_nx + 2, vector<double>(my_ny, 0.0));

    // allocate 1D communication buffers
    vector<double> c_left_recv(my_ny, 0.0);
    vector<double> c_left_send(my_ny, 0.0);
    vector<double> c_right_recv(my_ny, 0.0);
    vector<double> c_right_send(my_ny, 0.0);

    runCavityFlowSim(
        p, u, v, b,
        {c_left_recv, c_left_send},
        {c_right_recv, c_right_send},
        nt, nit,
        world_size, my_rank,
        dx, dy, dt,
        rho, nu
    );

    MPI_Finalize();
    return 0;
}

void runCavityFlowSim(
    vector<vector<double> >& p,
    vector<vector<double> >& u,
    vector<vector<double> >& v,
    vector<vector<double> >& b,
    array<vector<double>, 2>& c_left,
    array<vector<double>, 2>& c_right,
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

    for (int i = 0; i < nt; i++) {
        u.swap(un);
        v.swap(vn);

        comp_b(b, un, vn, dx, dy, dt, rho);
        MPI_Barrier(MPI_COMM_WORLD);
        update_halos(b, c_left, c_right, my_rank, world_size);

        for (int p_iter = 0; p_iter < nit; p_iter++) {
            p.swap(pn);
            p_np1(p, pn, b, dx, dy);
            MPI_Barrier(MPI_COMM_WORLD);
            update_halos(p, c_left, c_right, my_rank, world_size);
        }

        u_np1(u, un, vn, p, dx, dy, dt, rho, nu);
        v_np1(v, un, vn, p, dx, dy, dt, rho, nu);

        MPI_Barrier(MPI_COMM_WORLD);
        update_halos(u, c_left, c_right, my_rank, world_size);
        update_halos(v, c_left, c_right, my_rank, world_size);
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
            p[i][j] = (
                    pow(dy, 2) * (pn[i][j+1] + pn[i][j-1]) +
                    pow(dx, 2) * (pn[i+1][j] + pn[i-1][j])
                ) / (2.0*(pow(dx, 2)+pow(dy, 2))) - pow(dx, 2) * pow(dy, 2) /
                    (2.0*(pow(dx, 2)+pow(dy, 2))) * b[i, j];
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
    #pragma omp parallel for default(none) shared(u, un, vn, p) firstprivate(dx, dy, dt, rho, nu) collapse(2)
    for (int i = 1; i < u.size() - 1; i++) {
        for (int j = 1; j < u[0].size() - 1; j++) {
            u[i][j] = un[i][j] -
                un[i][j] * (dt / dx) * (un[i][j] - un[i][j-1]) -
                vn[i][j] * (dt / dy) * (un[i][j] - un[i-1][j]) -
                dt / (2.0 * rho * dx) * (p[i][j+1] - p[i][j-1]) +
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
    #pragma omp parallel for default(none) shared(v, un, vn, p) firstprivate(dx, dy, dt, rho, nu) collapse(2)
    for (int i = 1; i < v.size() - 1; i++) {
        for (int j = 1; j < v[0].size() - 1; j++) {
            v[i, j] = vn[i, j] -
                un[i][j] * (dt / dx) * (vn[i][j] - vn[i][j-1]) -
                vn[i][j] * (dt / dy) * (vn[i][j] - vn[i-1][j]) -
                dt / (2.0 * rho * dy) * (p[i+1][j] - p[i-1][j]) +
                nu * (
                    (dt / pow(dx, 2)) * (vn[i+1][j] - 2.0 * vn[i][j] + vn[i-1][j]) +
                    (dt / pow(dy, 2)) * (vn[i][j+1] - 2.0 * vn[i][j] + vn[i][j-1])
                );
        }
    }
}

void update_halos(
    vector<vector<double> >& a,
    array<vector<double>, 2>& left_comm_buffers,
    array<vector<double>, 2>& right_comm_buffers,
    int my_rank, int world_size,
) {
    // send the values just inside of my halos to my neighbors
    if (my_rank != 0) {
        left_comm_buffers[0].assign(a[1].begin(), a[1].end());
        MPI_Send(&left_comm_buffers[0][0], left_comm_buffers[0].size(), MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD);
    }
    if (my_rank != world_size - 1) {
        right_comm_buffers[0].assign(a[a.size() - 2].begin(), a[a.size() - 2].end());
        MPI_Send(&right_comm_buffers[0][0], right_comm_buffers[0].size(), MPI_DOUBLE, my_rank - 1, 1, MPI_COMM_WORLD);
    }

    // receive neighboring values and store them in my halos
    if (my_rank != 0) {
        MPI_Recv(&left_comm_buffers[1][0], left_comm_buffers[1].size(), MPI_DOUBLE, my_rank - 1, 1, MPI_COMM_WORLD);
        a[0].assign(left_comm_buffers[1].begin(), left_comm_buffers[1].end());
    }
    if (my_rank != world_size - 1) {
        MPI_Recv(&right_comm_buffers[1][0], right_comm_buffers[1].size(), MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD);
        a.back().assign(right_comm_buffers[1].begin(), right_comm_buffers[1].end());
    }
}
