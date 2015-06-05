#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <sstream>
#include <vector>
#include <cassert>
#include "compound_analyzer.h"
#include "feature_scorer.h"
#include "utils.h"
#include "derivation.h"
using namespace std;

void process(int line_number, const vector<string>& english, string german,
    compound_analyzer* analyzer, feature_scorer* scorer) {

  for (Derivation& derivation : analyzer->analyze(english, german, false)) {
    vector<string>& translations = derivation.translations;
    vector<string>& suffixes = derivation.suffixes;
    vector<unsigned>& indices = derivation.permutation;

    cout << line_number << " ||| ";

    // Output the translations and suffixes
    for (unsigned i = 0; i < indices.size(); ++i) {
      cout << translations[indices[i]] << "+" << suffixes[i] << " ";
    }
    cout << "||| ";

    // Output the permutation
    for (unsigned i = 0; i < indices.size(); ++i) {
      cout << indices[i] << " ";
    }
    cout << "||| ";

    // Output the features
    map<string, double> features = scorer->score(english, derivation);
    for (auto it = features.begin(); it != features.end(); ++it) {
      cout << it->first << "=" << it->second << " ";
    }
    cout << endl;
  }
}

void ShowUsageAndExit(char** argv) {
  cerr << "Usage: " << argv[0] << " fwd_ttable rev_ttable" << endl;
  exit(1);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    ShowUsageAndExit(argv);
  }
  adept::Stack stack;
  ttable fwd_ttable;
  ttable rev_ttable;
  fwd_ttable.load(argv[1]);
  rev_ttable.load(argv[2]);
  feature_scorer scorer(&fwd_ttable, &rev_ttable);
  compound_analyzer analyzer(&fwd_ttable);

  string line;
  int line_number = 0;
  while (getline(cin, line)) {
    stringstream sstream(line);
    line_number++;
    vector<string> english;
    string german;
    string temp;
    while (sstream >> temp) {
      temp = to_lower_case(temp);
      english.push_back(temp);
    }
    german = english[english.size() - 1];
    english.pop_back();

    process(line_number, english, german, &analyzer, &scorer);
  }

  return 0;
}
