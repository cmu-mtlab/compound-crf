#include <vector>
#include <string>
#include "ttable.h"
#include "utils.h"
#include "derivation.h"
using std::string;
using std::vector;

class compound_analyzer {
public:
  compound_analyzer(ttable* fwd_ttable);
  bool decompose(string compound, const vector<string>& pieces,
    vector<unsigned> permutation, vector<string>& suffixes);
  vector<Derivation> analyze(const vector<string>& english, string german, bool verbose = false);
  bool isReachable(const vector<string>& english, string german);
private:
  ttable* fwd_ttable;
};
