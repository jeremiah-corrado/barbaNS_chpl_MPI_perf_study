#include "utils.h"

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
