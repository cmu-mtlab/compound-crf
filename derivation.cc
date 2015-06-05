#include <sstream>
#include <cassert>
#include <iostream>
#include "derivation.h"
using namespace std;

string Derivation::toString() const {
  assert (translations.size() == suffixes.size());
  assert (translations.size() >= permutation.size());
  stringstream ss;
  for (unsigned j : permutation) {
    assert (j < translations.size());
    ss << translations[j] << suffixes[j];
  }
  return ss.str();
}

string Derivation::toLongString() const { return "tLS"; }

string Derivation::toLongString(map<string, double> features) const { return "tLS(f)"; }
