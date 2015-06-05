#include <cassert>
#include "noise_model.h"
using namespace std;

noise_model::noise_model(ttable* fwd_ttable) {
  this->fwd_ttable = fwd_ttable;
}

adouble noise_model::score(const vector<string>& inputs, const Derivation& derivation) {
  return 1.0;
}

Derivation noise_model::sample(const vector<string>& inputs) {
  vector<string> chosen_translations;

  for (string w : inputs) {
    map<string, double> translations = fwd_ttable->getTranslations(w);
    double score_sum = 0.0;
    for (auto kvp : translations) {
      string translation = kvp.first;
      double score = kvp.second;
      score_sum += exp(score);
    }

    double r = score_sum * (double)rand() / RAND_MAX;
    for (auto kvp : translations) {
      string translation = kvp.first;
      double score = kvp.second;
      assert (translation.size() > 0);
      if (r < exp(score)) {
        chosen_translations.push_back(translation);
        break;
      }
      r -= exp(score);
    }
  }

  vector<string> suffixes (inputs.size(), "");
  vector<unsigned> permutation;
  for (unsigned i = 0; i < inputs.size(); ++i) {
    permutation.push_back(i);
  }

  Derivation r {chosen_translations, suffixes, permutation};
  return r;
}

