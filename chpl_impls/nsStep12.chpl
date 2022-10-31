import flowPlot.printAndPlot;
use BlockDist;

config const xLen: real = 2,
             yLen: real = 2,
             nx = 41,
             ny = 41,
             nt = 500,
             nit = 50,
             dt = 0.001,
             rho = 1,
             nu = 0.1,
             F = 1;

const dx = 2.0 / (nx - 1),
      dy = 2.0 / (ny - 1),
      dxy2 = 2.0 * (dx**2 + dy**2);

config const l1_tol = 1e-4,
             max_iters = 2500;

config param createPlots = true,
             termOnTol = true;

// define 2D domain and distribution
const dom = {0..<nx, 0..<ny};
const DOM = dom dmapped Block(boundingBox=dom.expand(-1));
const DOMINNER : subdomain(DOM) = DOM[dom.expand(-1)];

// define distributed arrays
var p : [DOM] real = 0.0, // pressure scalar
    u : [DOM] real = 0.0, // x component of flow
    v : [DOM] real = 0.0; // y component of flow

runChannelFlowSim(p, u, v);

// print results
if createPlots then printAndPlot(
    p, u, v,
    (xLen, yLen),
    "results/nsStep12",
    "Channel Flow Solution"
);

proc runChannelFlowSim(ref u, ref v, ref p) where termOnTol == true {
    var l1_delta = 1.0,
        i = 0;

    var un : [DOM] real = u,
        vn : [DOM] real = v,
        pn : [DOM] real = p;

    var b : [DOM] real;

    while l1_delta > l1_tol && i < max_iters
    {
        u <=> un;
        v <=> vn;

        // compute the portion of p that depends only on u and v
        comp_b(b, u, v);

        // iteratively solve for p
        for p_iter in 0..#nit {
            p <=> pn;
            p_np1(p, pn, b);
            p_boundary(p);
        }

        // compute u and v concurrently
        cobegin {
            {
                u_np1(u, un, vn, p);
                // u_boundary(u);
            }
            {
                v_np1(v, un, vn, p);
                // v_boundary(v);
            }
        }

        // compute the tolerance condition and increment 'i'
        l1_delta = ((+ reduce u) - (+ reduce un)) / (+ reduce u);
        i += 1;
    }
}

proc runChannelFlowSim(ref u, ref v, ref p) where termOnTol == false {
    var un : [DOM] real = u,
        vn : [DOM] real = v,
        pn : [DOM] real = p;

    var b : [DOM] real;

    for i in 0..#max_iters {
        u <=> un;
        v <=> vn;

        // compute the portion of p that depends only on u and v
        comp_b(b, u, v);

        // iteratively solve for p
        for p_iter in 0..#nit {
            p <=> pn;
            p_np1(p, pn, b);
            p_boundary(p);
        }

        // compute u and v concurrently
        cobegin {
            {
                u_np1(u, un, vn, p);
                // u_boundary(u);
            }
            {
                v_np1(v, un, vn, p);
                // v_boundary(v);
            }
        }
    }
}

proc comp_b(ref b : [] real, const ref u, const ref v) {
    forall (i, j) in DOMINNER with (var du: real, var dv: real) {
        du = u[i, j+1] - u[i, j-1];
        dv = v[i+1, j] - v[i-1, j];

        b[i, j] = rho * (1.0 / dt) *
            (du / (2.0 * dx) + dv / (2.0 * dy)) -
            (du / (2.0 * dx))**2 -
            (dv / (2.0 * dy))**2 -
            2.0 * (
                (u[i+1, j] - u[i-1, j]) / (2.0 * dy) *
                (v[i, j+1] - v[i, j-1]) / (2.0 * dx)
            );
    }

    for (j_m, j, j_p) in [
        (ny-1, 0, 1), // bottom wall
        (ny-2, ny-1, 0) // top wall
    ] {
        forall i in DOMINNER.dim(0) with (var du: real, var dv: real) {
            du = u[i, j_p] - u[i, j_m];
            dv = v[i+1, j] - v[i-1, j];

            b[i, j] = rho * (1.0 / dt) *
                (du / (2.0 * dx) + dv / (2.0 * dy)) -
                (du / (2.0 * dx))**2 -
                (dv / (2.0 * dy))**2 -
                2.0 * (
                    (u[i+1, j] - u[i-1, j]) / (2.0 * dy) *
                    (v[i, j_p] - v[i, j_m]) / (2.0 * dx)
                );
        }
    }
}

proc p_np1(ref p : [] real, const ref pn, const ref b) {
    forall (i, j) in DOMINNER {
        p[i, j] = (
                    dy**2 * (pn[i, j+1] + pn[i, j-1]) +
                    dx**2 * (pn[i+1, j] + pn[i-1, j])
                ) / dxy2 - dx**2 * dy**2 / dxy2 * b[i, j];
    }

    for (j_m, j, j_p) in [
        (ny-1, 0, 1), // bottom wall
        (ny-2, ny-1, 0) // top wall
    ] {
        forall i in DOMINNER.dim(0) {
            p[i, j] = (
                    dy**2 * (pn[i, j_p] + pn[i, j_m]) +
                    dx**2 * (pn[i+1, j] + pn[i-1, j])
                ) / dxy2 - dx**2 * dy**2 / dxy2 * b[i, j];
        }
    }
}

proc u_np1(ref u : [] real, const ref un, const ref vn, const ref p) {
    forall (i, j) in DOMINNER {
        u[i, j] = un[i, j] -
            (un[i, j] * (dt / dx) * (un[i, j] - un[i, j-1])) -
            (vn[i, j] * (dt / dy) * (un[i, j] - un[i-1, j])) -
            (dt / (rho**2 * dx) * (p[i+1, j] - p[i-1, j])) +
            nu * (
                (dt / dx**2) * (un[i+1, j] - 2.0 * un[i, j] + un[i-1, j]) +
                (dt / dy**2) * (un[i, j+1] - 2.0 * un[i, j] + un[i, j-1])
            ) +
            F * dt;
    }

    for (j_m, j, j_p) in [
        (ny-1, 0, 1), // bottom wall
        (ny-2, ny-1, 0) // top wall
    ] {
        forall i in DOMINNER.dim(0) {
            u[i, j] = un[i, j] -
            (un[i, j] * (dt / dx) * (un[i, j] - un[i, j_m])) -
            (vn[i, j] * (dt / dy) * (un[i, j] - un[i-1, j])) -
            (dt / (rho**2 * dx) * (p[i+1, j] - p[i-1, j])) +
            nu * (
                (dt / dx**2) * (un[i+1, j] - 2.0 * un[i, j] + un[i-1, j]) +
                (dt / dy**2) * (un[i, j_p] - 2.0 * un[i, j] + un[i, j_m])
            ) +
            F * dt;
        }
    }
}

proc v_np1(ref v : [] real, const ref un, const ref vn, const ref p) {
    forall (i, j) in DOMINNER  {
        v[i, j] = vn[i, j] -
            un[i, j] * (dt / dx) * (vn[i, j] - vn[i, j-1]) -
            vn[i, j] * (dt / dy) * (vn[i, j] - vn[i-1, j]) -
            dt / (rho**2 * dy) * (p[i, j+1] - p[i, j-1]) +
            nu * (
                (dt / dx**2) * (vn[i+1, j] - 2.0 * vn[i, j] + vn[i-1, j]) +
                (dt / dy**2) * (vn[i, j+1] - 2.0 * vn[i, j] + vn[i, j-1])
            );
    }

    for (j_m, j, j_p) in [
        (ny-1, 0, 1), // bottom wall
        (ny-2, ny-1, 0) // top wall
    ] {
        forall i in DOMINNER.dim(0) {
            v[i, j] = vn[i, j] -
            un[i, j] * (dt / dx) * (vn[i, j] - vn[i, j_m]) -
            vn[i, j] * (dt / dy) * (vn[i, j] - vn[i-1, j]) -
            dt / (rho**2 * dy) * (p[i, j_p] - p[i, j_m]) +
            nu * (
                (dt / dx**2) * (vn[i+1, j] - 2.0 * vn[i, j] + vn[i-1, j]) +
                (dt / dy**2) * (vn[i, j_p] - 2.0 * vn[i, j] + vn[i, j_m])
            );
        }
    }
}

// proc u_boundary(ref u) {
//     u[0, ..] = 0.0;
//     u[nx-1, ..] = 0.0;
// }

// proc v_boundary(ref v) {
//     v[0, ..] = 0.0;
//     v[nx-1, ..] = 0.0;
// }

proc p_boundary(ref p) {
    p[0, ..] = p[1, ..];
    p[nx-1, ..] = p[nx-2, ..];
}
