#pragma once
#include <vector>
#include <string>
#include <map>
using namespace std;

class Derivation {
public:
  vector<string> translations;
  vector<string> suffixes;
  vector<unsigned> permutation;

  string toString() const;
  string toLongString() const;
  string toLongString(map<string, double> features) const;
};
