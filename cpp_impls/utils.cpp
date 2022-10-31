#include "utils.h"

// Print 'a' to '{path}.dat' and the domain information to '{path}.meta'. Call python plotting script
void printAndPlot(
        vector<vector<double> > const& a,
        double xLen, double yLen,
        string const& path,
        string const& plotTitle,
        string const& plotKey
) {
    const auto nx = a.size();
    const auto ny = a[0].size();

    ofstream metaFile;
    metaFile.open(path + ".meta");
    metaFile << setprecision(10);
    metaFile << nx << ", " << ny << ", " << xLen << ", " << yLen << "\n";
    metaFile.close();

    ofstream dataFile;
    dataFile.open(path + ".dat");
    dataFile << setprecision(10);
    for (auto const& xlayer : a) {
        for (auto const& val : xlayer) {
            dataFile << val << " ";
        }
        dataFile << "\n";
    }
    dataFile.close();

    // run matplotlib script
    stringstream plotCmd;
    plotCmd << "Python3 ./surfPlot.py " << path << " " << plotTitle << " " << plotKey;
    const string plotCmdString = plotCmd.str();
    const char* plotCmdCString = plotCmdString.c_str();
    std::system(plotCmdCString);
}

void parseArgsWithDefaults(
        int argc, const char *argv[],
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
