#pragma once
#include <map>
#include <vector>
#include <string>
#include <unordered_set>
#include "NeuralLM/neurallm.h"
#include "NeuralLM/vocabulary.h"
#include "ttable.h"
#include "utils.h"
#include "derivation.h"

using std::string;
using std::vector;
using std::unordered_set;
using std::map;

class feature_scorer {
public:
  feature_scorer(ttable* fwd_ttable, ttable* rev_ttable);
  double lexical_score(ttable* table, const string& source,
    const string& target);
  double lexical_score(ttable* table, const vector<string>& source,
    const vector<string>& target, const vector<unsigned>& permutation);
  static vector<string> split_utf8(const string& target);

  map<string, double> score_translation(const string& source,
    const string& target);
  map<string, double> score_suffix(const string& root, const string& suffix);
  map<string, double> score_permutation(const vector<string>& source,
    const vector<unsigned>& permutation);
  map<string, double> score_lm(const string& output);
  map<string, double> score_lm(const Derivation& derivation);

  map<string, double> score(const vector<string>& source,
    const Derivation& derivation);

  double oov_score = -10.0;
//private:
  ttable* fwd_ttable;
  ttable* rev_ttable;
  NeuralLM* lm;
  vocabulary* lm_vocab;
};
