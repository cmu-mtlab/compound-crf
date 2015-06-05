#include <vector>
#include <string>
#include <map>
#include "ttable.h"
#include "utils.h"
#include "derivation.h"
#include "adept.h"
using std::map;
using std::vector;
using std::string;
using adept::adouble;

class noise_model {
public:
  noise_model(ttable* fwd_ttable);
  adouble score(const vector<string>& input, const Derivation& derivation);
  Derivation sample(const vector<string>& input);

private:
  ttable* fwd_ttable;
  map<string, double> ttable_marginals;
};
