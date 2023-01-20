#include "utils.h"

void printDownSampled(
    vector<vector<double>>& a,
    char name,
    string const& path,
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
        // (using a flat array s.t. values are stored in contiguous memory (required for MPI_Gather))
        vector<double> globalA((nx / xStride) * (ny / yStride), 0.0);
        downSampleAndGather(a, globalA, my_rank, world_size, xStride, yStride, nx, ny);
        global_ptr.swap(globalA);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (my_rank == 0) {
        printForPlot(global_ptr, path + '_' + name, ny / yStride);
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
    // (using a flat array s.t. values are stored in contiguous memory (required for MPI_Gather))
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

void flowPlot(
    string const& path,
    string const& title,
    int nx, int ny,
    double xLen, double yLen
) {
    ofstream metaFile;
    metaFile.open(path + ".meta");
    metaFile << setprecision(10);
    metaFile << nx << ", " << ny << ", " << xLen << ", " << yLen << "\n";
    metaFile.close();

    stringstream plotCmd;
    plotCmd << "Python3 ./flowPlot.py " << path << " " << title;
    const string plotCmdString = plotCmd.str();
    const char* plotCmdCString = plotCmdString.c_str();
    std::system(plotCmdCString);
}

void printForPlot(
    vector<double> const& a,
    string const& path,
    int ny
) {
    const int nx = a.size() / ny;
    ofstream dataFile;
    dataFile.open(path + ".dat");
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny - 1; j++) {
            dataFile << a[i * ny + j] << " ";
        }
        dataFile << a[i * ny + ny - 1] << "\n";
    }
    dataFile.close();
}

void parseArgsWithDefaults(
        int argc, char *argv[],
        unordered_map<string, variant<int, double>>& defaults
) {
    for (int i = 0; i < argc; i++) {
        string arg(argv[i]);
        if (arg[0] == '-' && arg[1] == '-' && arg.find('=') != string::npos) {
            size_t eq_pos = arg.find('=');
            string argName = arg.substr(2, eq_pos-2);

            auto search = defaults.find(argName);
            if (search != defaults.end()) {
                if (holds_alternative<int>(search->second)) {
                    try {
                        search->second = stoi(arg.substr(eq_pos+1, arg.size()));
                    } catch (...) {
                        cout << "Warning: invalid value for: '" << argName <<
                            "'; keeping default value (" << get<int>(search->second) << ")\n";
                    }
                } else {
                    try {
                        search->second = stod(arg.substr(eq_pos + 1, arg.size()));
                    } catch (...) {
                        cout << "Warning: invalid value for: '" << argName <<
                             "'; keeping default value (" << get<double>(search->second) << ")\n";
                    }
                }
            } else {
                cout << "Warning: argument: '" << argName << "' doesn't have a default value; ignoring\n";
            }
        }
    }
}
