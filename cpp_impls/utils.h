#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <variant>
#include <unordered_map>
#include <iomanip>

using namespace std;

void flowPlot(
    string const& path,
    string const& title,
    int nx, int ny,
    double xLen, double yLen
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
