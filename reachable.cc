#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <sstream>
#include <vector>
#include <cassert>
#include "compound_analyzer.h"
#include "utils.h"
using namespace std;

void ShowUsageAndExit(char** argv) {
  cerr << "Usage: " << argv[0] << " fwd_ttable" << endl;
  exit(1);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    ShowUsageAndExit(argv);
  }
  adept::Stack stack;
  ttable fwd_ttable;
  fwd_ttable.load(argv[1]);
  compound_analyzer analyzer(&fwd_ttable);

  string line;
  while (getline(cin, line)) {
    stringstream sstream(line);
    vector<string> english;
    string german;
    string temp;
    while (sstream >> temp) {
      temp = to_lower_case(temp);
      english.push_back(temp);
    }
    german = english[english.size() - 1];
    english.pop_back();

    if (analyzer.isReachable(english, german)) {
      cout << line << endl;
    }
  }

  return 0;
}
