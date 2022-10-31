use IO;
import Subprocess.spawn;

proc printAndPlot(
    const ref p: [?d] real,
    const ref u: [d] real,
    const ref v: [d] real,
    lens: 2*(real),
    path: string,
    plotTitle: string
) throws where d.rank == 2 {
    const nx = d.dim(0).size;
    const ny = d.dim(1).size;

    var metaFile = openwriter(path + ".meta");
    metaFile.writef("%i, %i, %.10dr, %.10dr\n", nx, ny, lens(0), lens(1));
    metaFile.close();

    writeDat(openwriter(path + "_p.dat"), p, nx, ny);
    writeDat(openwriter(path + "_u.dat"), u, nx, ny);
    writeDat(openwriter(path + "_v.dat"), v, nx, ny);

    spawn(["Python3", "flowPlot.py", path, plotTitle]);
}

proc writeDat(in file, const ref a: [?d] real, nx, ny) throws where d.rank == 2 {
    for i in 0..<nx {
        for j in 0..<(ny-1) do file.writef("%.10dr ", a[i, j]);
        file.writef("%.10dr\n", a[i, ny-1]);
    }
    file.close();
}
