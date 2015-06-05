#include <cassert>
#include <iostream>
#include "feature_scorer.h"
#include "utf8.h"
using namespace std;

feature_scorer::feature_scorer(ttable* fwd, ttable* rev) {
  fwd_ttable = fwd;
  rev_ttable = rev;
  lm = NULL;
  lm_vocab = NULL;
}

double feature_scorer::lexical_score(ttable* table, const string& source,
  const string& target) {
  if (source.size() == 0 || target.size() == 0) {
    return 0.0;
  }

  double score;
  if (table->getScore(source, target, score)) {
    return score;
  }
  else { 
    return oov_score;
  }
}

double feature_scorer::lexical_score(ttable* table, const vector<string>& source,
    const vector<string>& target, const vector<unsigned>& permutation) {
  double total_score = 0.0; 
  for (int i : permutation) {
    total_score += lexical_score(table, source[i], target[i]);
  }
  return total_score;
}

map<string, double> feature_scorer::score_translation(const string& source,
    const string& target) { 
  map<string, double> features;
  if (target.size() == 0) {
    features["tgt_null"] = 1;
    features[source + "_to_null"] = 1;
    features["null_score"] = lexical_score(rev_ttable, "<eps>", source);
  }
  features["fwd_score"] = lexical_score(fwd_ttable, source, target);
  features["rev_score"] = lexical_score(rev_ttable, target, source);
  features["length"] = target.length();
  return features;
}

map<string, double> feature_scorer::score_suffix(const string& root, const string& suffix) {
  map<string, double> features;
  features["suffix_" + suffix] = 1.0;
  features["length"] = suffix.length();
  return features;
}

map<string, double> feature_scorer::score_permutation(const vector<std::string>& source,
    const vector<unsigned>& permutation) {
  map<string, double> features;
  bool monotone = true;
  if (permutation.size() > 0) {
    unsigned last = permutation[0];
    for (unsigned i = 1; i < permutation.size(); ++i) {
      if (permutation[i] < last) {
        monotone = false;
        break;
      }
      last = permutation[i];
    }
  }
  features["monotone"] = monotone ? 1.0 : 0.0;
  return features;
}

vector<string> feature_scorer::split_utf8(const string& target) {
  vector<string> r;
  // From here down to the while loop is all
  // boiler plate code that extracts UTF8
  // letters from a string one at a time
  const char* s = target.c_str();
  char* i = (char*)s;
  char* end = i + target.length() + 1;

  unsigned char symbol[6] = {0, 0, 0, 0, 0, 0};
  do {
    for (int i = 0; i < 5; ++i) {
      symbol[i] = 0;
    }

    uint32_t code = utf8::next(i, end);
    if (code == 0) {
      continue;
    }
    utf8::append(code, symbol);
    string c = (char*)symbol;
    r.push_back(c);
  } while(i < end);
  return r;
}

map<string, double> feature_scorer::score_lm(const string& target) { 
  adouble lm_score = 0.0;
  int lm_oov = 0;
  const unsigned unk = lm_vocab->convert("<unk>");
  const unsigned bos = lm_vocab->convert("<s>");
  const unsigned eos = lm_vocab->convert("</s>");
  //lm->reset_context(bos);

  // From here down to the while loop is all
  // boiler plate code that extracts UTF8
  // letters from a string one at a time
  const char* s = target.c_str();
  char* i = (char*)s;
  char* end = i + target.length() + 1;

  unsigned char symbol[6] = {0, 0, 0, 0, 0, 0};
  do {
    for (int i = 0; i < 5; ++i) {
      symbol[i] = 0;
    }

    uint32_t code = utf8::next(i, end);
    if (code == 0) {
      continue;
    }
    utf8::append(code, symbol);
    string c = (char*)symbol;

    // Now that we have a single UTF8 character,
    // score it, then add it to the context to be
    // re-used for the next character
    /*unsigned cid = lm_vocab->lookup(c, unk);
    if (cid != unk) {
      lm_score += lm->log_prob(cid);
    }
    else {
      lm_oov += 1;
    }
    lm->add_to_context(cid);*/
  } while(i < end);

  //lm_score += lm->log_prob(eos);

  map<string, double> features;
  features["lm_score"] = lm_score.value();
  features["lm_oov"] = lm_oov;
  return features;
}

map<string, double> feature_scorer::score_lm(const Derivation& derivation) {
  return score_lm(derivation.toString());
}

map<string, double> feature_scorer::score(const vector<string>& source,
    const Derivation& derivation) {
  const vector<string> translations = derivation.translations;
  const vector<string> suffixes = derivation.suffixes;
  const vector<unsigned> permutation = derivation.permutation;

  assert (source.size() == translations.size());
  assert (translations.size() == suffixes.size());
  assert (permutation.size() <= translations.size());

  map<string, double> features;

  for (auto& kvp : score_permutation(source, permutation)) {
    features[kvp.first] += kvp.second;
  }

  for (unsigned i = 0; i < translations.size(); ++i) {
    for (auto& kvp : score_translation(source[i], translations[i])) {
      features[kvp.first] += kvp.second;
    }

    for (auto& kvp : score_suffix(translations[i], suffixes[i])) {
      features[kvp.first] += kvp.second;
    }
  }

  if (lm != NULL) {
    for (auto& kvp : score_lm(derivation)) {
      features[kvp.first] += kvp.second;
    }
  }

  return features;
}

