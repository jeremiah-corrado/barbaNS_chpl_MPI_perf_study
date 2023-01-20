#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <variant>
#include <unordered_map>
#include <iomanip>

#include <mpi.h>

using namespace std;

void flowPlot(
    string const& path,
    string const& title,
    int nx, int ny,
    double xLen, double yLen
);

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
);

void printForPlot(
    vector<double> const& a,
    string const& path,
    int ny
);

void parseArgsWithDefaults(
        int argc, char *argv[],
        unordered_map<string, variant<int, double>>& defaults
);

void downSampleAndGather(
    vector<vector<double>>& a,
    vector<double>& a_global,
    int my_rank,
    int world_size,
    int xStride,
    int yStride,
    int nx,
    int ny
);
