#include <sstream>
#include <iostream>
#include <fstream>
#include "ttable.h"
#include "utils.h"
using namespace std;

bool ttable::getScore(string source, string target, double& score) {
  if (table.find(source) == table.end()) {
    return false;
  }
  else if (table[source].find(target) == table[source].end()) {
    return false;
  }
  else {
    score = table[source][target];
    return true;
  }
}

void ttable::setScore(string source, string target, double score) {
  if (table.find(source) == table.end()) {
    table[source] = map<string, double>();
  }
  table[source][target] = score;
}

map<string, double> ttable::getTranslations(string source) {
  if (table.find(source) == table.end()) {
    map<string, double> empty_map;
    return empty_map;
  }

  return table[source];
}

void ttable::load(string filename) {
  ifstream f(filename);
  if (!f.is_open()) {
    cerr << "ERROR: Unable to open " << filename << "." << endl;
    exit(1);
  }

  string line;
  while (getline(f, line)) {
    string source;
    string target;
    double score;
    stringstream sstream(line);
    sstream >> source;
    sstream >> target;
    sstream >> score;
    source = to_lower_case(source);
    target = to_lower_case(target);
    setScore(source, target, score);
  }
}

