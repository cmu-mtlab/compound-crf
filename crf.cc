#include <cassert>
#include <iostream>
#include <algorithm>
#include <set>
#include <tuple>
#include <unordered_map>
#include <limits>
#include <fstream>
#include "NeuralLM/context.h"
#include "crf.h"
using namespace std;

namespace std {
  template<>
  struct hash<tuple<unsigned int, vector<unsigned>, Context> > {
    hash<unsigned> uint_hash;
    hash<Context> context_hash;
    size_t operator()(const tuple<unsigned int, vector<unsigned>, Context>& t) const {
      size_t seed = 0;
      hash_combine(seed, get<0>(t));
      for (unsigned i : get<1>(t)) {
        hash_combine(seed, i);
      }
      hash_combine(seed, get<2>(t));
      return seed;
    }
  };
}

const bool use_adadelta = true;

// returns log(a * exp(x) + b * exp(y))
adouble log_sum_exp(adouble x, adouble y, adouble a = 1.0, adouble b = 1.0) {
  adouble m = max(x, y);
  return m + log(a * exp(x - m) + b * exp(y - m));
}

crf::crf(adept::Stack* stack, feature_scorer* scorer) {
  this->stack = stack;
  this->scorer = scorer;
}

adouble crf::dot(const map<string, double>& features, const map<string, adouble>& weights) {
  adouble score = 0.0;
  for (auto& kvp : features) {
    if (weights.find(kvp.first) == weights.end()) {
      cerr << "ERROR: Invalid attempt to use unknown feature \"" << kvp.first << "\"." << endl;
    }
    assert(weights.find(kvp.first) != weights.end());
    auto x = weights.find(kvp.first);
    score += x->second * kvp.second;
  }
  return score;
}

adouble crf::score(const vector<string>& x, const Derivation& y) {
  map<string, double> features = scorer->score(x, y);
  return dot(features, weights);
}

adouble crf::lattice_partition_function(const vector<string>& x) {
  // a state is a coverage bitvector, a list of used source indices, and a context
  // the bit vector includes words that translate to NULL
  // but the vector of indices does not
  typedef tuple<unsigned, vector<unsigned>, Context> state;
  const unsigned unk = scorer->lm_vocab->convert("<unk>");
  const unsigned eos = scorer->lm_vocab->convert("</s>");
  const unsigned one = 1;

  unordered_map<state, vector<adouble>> scores;
  unordered_map<unsigned, unordered_set<state> > states_by_step;
  for (unsigned i = 0; i < x.size() + 1; ++i) {
    states_by_step[i] = unordered_set<state>();
  } 

  Context start_context(scorer->lm->context_size());
  start_context.init(scorer->lm_vocab->lookup("<s>", 0));
  vector<unsigned> empty_coverage;
  for (unsigned null_coverage = 0; null_coverage < (one << x.size()); ++null_coverage) { 
    adouble score = 0.0;
    for (unsigned int i = 0; i < x.size(); ++i) {
      if (null_coverage & (one << i)) {
        map<string, double> translation_features = scorer->score_translation(x[i], "");
        map<string, double> suffix_features = scorer->score_suffix("", "");
        adouble local_score = dot(translation_features, weights) + dot(suffix_features, weights);
        score += local_score;
      }
    }
    state start_state = make_tuple(null_coverage, empty_coverage, start_context);
    scores[start_state].push_back(score);
    states_by_step[popCount(null_coverage)].insert(start_state);
  }

  for (unsigned int step = 0; step < x.size(); ++step) {
    for (const state& from_state : states_by_step[step]) {
      adouble from_score = log_sum_exp(scores[from_state]);
      unsigned coverage = get<0>(from_state);
      assert (popCount(coverage) == step);
      for (unsigned int i = 0; i < x.size(); ++i) {
        if (coverage & (one << i)) {
          continue;
        } 
        for (const auto& pair : scorer->fwd_ttable->getTranslations(x[i])) {
          const string& translation = get<0>(pair);
          for (const string& suffix : suffix_list) {
            map<string, double> translation_features = scorer->score_translation(x[i], translation);
            map<string, double> suffix_features = scorer->score_suffix(translation, suffix);
            adouble local_score = dot(translation_features, weights) + dot(suffix_features, weights);

            adouble lm_score = 0.0;
            vector<unsigned> new_coverage = get<1>(from_state);
            new_coverage.push_back(i);
            Context new_context = get<2>(from_state);
            for (const string& letter : feature_scorer::split_utf8(translation + suffix)) {
              const unsigned letter_id = scorer->lm_vocab->lookup(letter, unk);
              lm_score += scorer->lm->log_prob(new_context, letter_id);
              new_context.add(letter_id);
            }
            state new_state = make_tuple(coverage | (one << i), new_coverage, new_context);
            assert (popCount(get<0>(new_state)) == step + 1);
            states_by_step[step + 1].insert(new_state);
            scores[new_state].push_back(from_score + local_score + lm_score * weights["lm_score"]);
          }
        }
      }
    }
  }

  vector<adouble> final_scores;
  for (const state& final_state : states_by_step[x.size()]) {
    unsigned coverage = get<0>(final_state);
    assert (popCount(coverage) == x.size());
    assert (coverage < (one << x.size()));

    vector<unsigned> permutation = get<1>(final_state);
    adouble permutation_score = dot(scorer->score_permutation(x, permutation), weights);

    Context context = get<2>(final_state);
    adouble lm_score = scorer->lm->log_prob(context, eos);

    adouble final_score = log_sum_exp(scores[final_state]);
    final_score += lm_score * weights["lm_score"];
    final_score += permutation_score;
    final_scores.push_back(final_score);
  }
  return log_sum_exp(final_scores);
}

// Computes log of sum_t sum_s exp (score(t|w) + score(s|t))
// where w is the source word
// t is a translation of w
// and s is a suffix on t
// Does NOT handle th ecase where w translates into NULL.
adouble crf::word_partition_function(const string& source) {
  vector<adouble> translation_scores;
  for (auto kvp : scorer->fwd_ttable->getTranslations(source)) {
    string target = kvp.first;
    map<string, double> translation_features = scorer->score_translation(source, target);
    for (auto kvp : scorer->score_lm(target)) {
      translation_features[kvp.first] += kvp.second;
    }

    vector<adouble> suffix_scores;
    for (string suffix : suffix_list) {
      map<string, double> suffix_features = scorer->score_suffix(target, suffix);
      for (auto kvp : scorer->score_lm(suffix)) {
        suffix_features[kvp.first] += kvp.second;
      }
      adouble suffix_score = dot(suffix_features, weights);
      suffix_scores.push_back(suffix_score);
    }
    adouble suffix_scores_sum = log_sum_exp(suffix_scores);
    adouble translation_score = dot(translation_features, weights);

    translation_scores.push_back(translation_score + suffix_scores_sum);
  }
  return log_sum_exp(translation_scores);
}

adouble crf::partition_function(const vector<string>& x) {
  adouble z = 0.0;
  vector<adouble> non_null_scores;
  vector<adouble> null_scores;
  for (unsigned i = 0; i < x.size(); ++i) {
    const string& source = x[i];
    adouble non_null_score = word_partition_function(source);
    adouble null_score;

    // Handle the case where the ith source word translates into NULL
    {
      map<string, double> translation_features = scorer->score_translation(source, "");
      map<string, double> suffix_features = scorer->score_suffix("", "");
      adouble translation_score = dot(translation_features, weights);
      adouble suffix_score = dot(suffix_features, weights);
      null_score = translation_score + suffix_score;
    }

    non_null_scores.push_back(non_null_score);
    null_scores.push_back(null_score);
  }

  assert(null_scores.size() == x.size());
  assert(non_null_scores.size() == x.size());
  assert(x.size() <= 5);

  vector<adouble> final_scores;
  const int factorial[] = {1, 1, 2, 6, 24, 120};
  const unsigned one = 1;
  // Loop over combinations of NULLs and non-NULLs
  for (unsigned i = 0; i < (one << x.size()); ++i) {
    adouble score = 0.0;
    vector<unsigned> indices;
    for (unsigned j = 0; j < x.size(); ++j) {
      if (i & (1 << j)) {
        indices.push_back(j);
      }
    }

    vector<adouble> permutation_scores;
    permutation_scores.reserve(factorial[popCount(i)]);
    do {
      map<string, double> permutation_features = scorer->score_permutation(x, indices);
      adouble permutation_score = dot(permutation_features, weights);
      for (unsigned j = 0; j < x.size(); ++j) {
        if (i & (1 << j)) {
          permutation_score += non_null_scores[j];
        }
        else {
          permutation_score += null_scores[j];
        }
      }
      permutation_scores.push_back(permutation_score);
    } while (next_permutation(indices.begin(), indices.end()));
    assert(popCount(i) <= 5);
    //score += log(factorial[popCount(i)]);
    final_scores.push_back(log_sum_exp(permutation_scores));
  }

  return log_sum_exp(final_scores);
}

adouble crf::slow_partition_function(const vector<string>& x, const map<string, adouble>& weights) {
  vector<vector<string> > candidate_translations;
  for (unsigned i = 0; i < x.size(); ++i) {
    vector<string> translations;
    translations.push_back("");
    for (auto& kvp : scorer->fwd_ttable->getTranslations(x[i])) {
      translations.push_back(kvp.first);
    }
    candidate_translations.push_back(translations);
  }
  assert(candidate_translations.size() == x.size());


  vector<Derivation> derivations;
  // Loop over the cross product of possible translations
  for (vector<string> translations : cross(candidate_translations)) {
    // This variable will hold a permutation of the integers [0, |G|)
    // Note that we remove indices coresponding to NULL translations
    // since their ordering does not affect the output.
    vector<unsigned> indices;
    for (unsigned i = 0; i < translations.size(); ++i) {
      if (translations[i].size() != 0) {
        indices.push_back(i);
      }
    }

    // Don't allow all the pieces to translate as NULL
    /*if (indices.size() == 0) {
      continue;
    }*/

    vector<vector<string> > candidate_suffixes;
    for (unsigned i = 0; indices.size() > 0 && i < indices.size() - 1; ++i) {
      vector<string> suffixes;
      for (string suffix : suffix_list) {
        suffixes.push_back(suffix);
      }
      candidate_suffixes.push_back(suffixes);
    }

    // The last iteration is split since eventually the last suffix
    // will be drawn from a different table
    if (indices.size() > 0)
    {
      vector<string> suffixes; 
      for (string suffix : suffix_list) {
        suffixes.push_back(suffix);
      }
      candidate_suffixes.push_back(suffixes);
    }
    assert(candidate_suffixes.size() == indices.size());

    // Loop over all possible permutations.
    do { 
      for (vector<string> chosen_suffixes : cross(candidate_suffixes)) {
        vector<string> suffixes(translations.size(), string(""));
        for (unsigned i = 0; i < indices.size(); ++i) {
          suffixes[indices[i]] = chosen_suffixes[i];
        } 

        assert(suffixes.size() == translations.size());
        Derivation derivation { translations, suffixes, indices };
        derivations.push_back(derivation);
      }
    } while (next_permutation(indices.begin(), indices.end()));
  }

  vector<adouble> scores;
  for (Derivation d : derivations) {
    map<string, double> features = scorer->score(x, d);
    scores.push_back(dot(features, weights)); 
  }

  return log_sum_exp(scores);
}

adouble crf::score_noise(const vector<string>& x, const Derivation& y) {
  adouble score = 0.0;
  assert(x.size() == y.translations.size());
  for (unsigned i = 0; i < x.size(); ++i) {
    double s;
    bool found = scorer->fwd_ttable->getScore(x[i], y.translations[i], s);
    s = found ? s : -10.0;
    score += s;
  }
  return score;
}

adouble crf::nce_loss(const vector<string>& x, const Derivation& y, const vector<Derivation>& n) {
  int k = n.size();
  adouble loss = 0.0;

  adouble py = score(x, y);       // log u_model(x, y)
  adouble ny = score_noise(x, y); // log u_noise(x, y)
  assert(isfinite(py.value()));
  assert(isfinite(ny.value()));

  adouble pd1y = py - log_sum_exp(py, ny, 1.0, k);          // log p(D = 1 | x, y)
  adouble pd0y = log(k) + ny - log_sum_exp(py, ny, 1.0, k); // log p(D = 0 | x, y)
  assert(exp(pd1y) >= 0.0 && exp(pd1y) <= 1.0);             // Ensure p(D = 0 | x, y) is between 0 and 1
  assert(abs(exp(pd0y) + exp(pd1y) - 1.0) < 0.0001);        // Ensure p(D = 0 | .) + p(D + 1 | .) = 1

  loss += pd1y;

  for (int i = 0; i < k; ++i) {
    const Derivation& z = n[i];                               // z is a noise sample
    adouble pz = score(x, z);                                 // log u_model(x, z);
    adouble nz = score_noise(x, z);                           // log u_noise(x, z);
    adouble pd0z = log(k) + nz - log_sum_exp(pz, nz, 1.0, k); // p(D = 0 | x, y);
    adouble pd1z = pz - log_sum_exp(pz, nz, 1.0, k);          // p(D = 1 | x, y);
    assert(exp(pd0z) >= 0.0 && exp(pd0z) <= 1.0);             // Ensure p(D = 0 | x, y) is between 0 and 1
    assert(abs(exp(pd0z) + exp(pd1z) - 1.0) < 0.0001);        // Ensure p(D = 0 | .) + p(D + 1 | .) = 1
    loss += pd0z;
  }

  return loss;
}

adouble crf::l2penalty(const double lambda) {
  adouble res = 0.0;
  for (auto& kvp : weights) {
    res += lambda * kvp.second * kvp.second;
  }
  return res;
}

adouble crf::train_nobatch(const vector<vector<string> >& x, const vector<vector<Derivation> >& y,
    double learning_rate, double l2_strength) {
  assert(x.size() == y.size());
  adouble total_log_loss = 0.0;
  for (unsigned i = 0; i < x.size(); ++i) {
    cerr << i << "/" << x.size() << "\r";
    cerr.flush();
    adouble log_loss = 0.0;
    stack->new_recording();
    adouble d = lattice_partition_function(x[i]);
    /*vector<adouble> scores;
    for (unsigned int j = 0; j < y[i].size(); ++j) {
      scores.push_back(score(x[i], y[i][j]));
    }
    adouble log_total = log_sum_exp(scores);
    log_loss -= log_total - d;*/

    adouble score_sum = 0.0;
    for (unsigned int j = 0; j < y[i].size(); ++j) {
      score_sum += exp(score(x[i], y[i][j]));
    }
    log_loss -= log(score_sum) - d;
    log_loss += l2penalty(l2_strength);

    log_loss.set_gradient(1.0);
    stack->compute_adjoint();
    for (auto& fv : weights) {
      const double g = fv.second.get_gradient();
      double delta;
      if (use_adadelta) {
        historical_gradients[fv.first] = rho * historical_gradients[fv.first] + (1 - rho) * g * g;
        delta = -g * sqrt(historical_deltas[fv.first]) / sqrt(historical_gradients[fv.first]);
        historical_deltas[fv.first] = rho * historical_deltas[fv.first] + (1 - rho) * delta * delta;
      }
      else {
        delta = -g * learning_rate;
      }
      weights[fv.first] += delta;
    }
    total_log_loss += log_loss;
  }
  cerr << x.size() << "/" << x.size() << endl;

  return total_log_loss;
}

adouble crf::train(const vector<vector<string> >& x, const vector<vector<Derivation> >& y,
    double learning_rate, double l2_strength) {
  assert(x.size() == y.size());
  adouble log_loss = 0.0;
  stack->new_recording();
  for (unsigned i = 0; i < x.size(); ++i) {
    cerr << i << "/" << x.size() << "\r";
    adouble d = partition_function(x[i]);
    /*vector<adouble> scores;
    for (unsigned int j = 0; j < y[i].size(); ++j) {
      scores.push_back(score(x[i], y[i][j]));
    }
    adouble log_total = log_sum_exp(scores);
    log_loss -= log_total - d;*/

    adouble score_sum = 0.0;
    for (unsigned int j = 0; j < y[i].size(); ++j) {
      score_sum += exp(score(x[i], y[i][j]));
    }
    log_loss -= log(score_sum) - d;
  }
  cerr << x.size() << "/" << x.size() << "\r";
   
  log_loss += l2penalty(l2_strength);

  log_loss.set_gradient(1.0);
  stack->compute_adjoint();
  for (auto& fv : weights) {
    const double g = fv.second.get_gradient();
    double delta;
    if (use_adadelta) {
      historical_gradients[fv.first] = rho * historical_gradients[fv.first] + (1 - rho) * g * g;
      delta = -g * sqrt(historical_deltas[fv.first]) / sqrt(historical_gradients[fv.first]);
      historical_deltas[fv.first] = rho * historical_deltas[fv.first] + (1 - rho) * delta * delta;
    }
    else {
      delta = -g * learning_rate;
    }
    weights[fv.first] += delta;
  }

  return log_loss;
}

adouble crf::train(const vector<vector<string> >& x, const vector<Derivation>& y,
    double learning_rate, double l2_strength) {
  assert(x.size() == y.size());
  adouble log_loss = 0.0;
  stack->new_recording();
  for (unsigned i = 0; i < x.size(); ++i) {
    adouble n = score(x[i], y[i]);
    adouble d = partition_function(x[i]);
    //adouble slow = slow_partition_function(x[i], weights);
    //assert(abs(d - slow) < 0.0001);
    assert(n < d);
    adouble log_prob = n - d;
    log_loss -= log_prob;
  }
  log_loss += l2penalty(l2_strength);

  log_loss.set_gradient(1.0);
  stack->compute_adjoint();
  for (auto& fv : weights) {
    const double g = fv.second.get_gradient();
    double delta;
    if (use_adadelta) {
      historical_gradients[fv.first] = rho * historical_gradients[fv.first] + (1 - rho) * g * g;
      delta = -g * sqrt(historical_deltas[fv.first]) / sqrt(historical_gradients[fv.first]);
      historical_deltas[fv.first] = rho * historical_deltas[fv.first] + (1 - rho) * delta * delta;
    }
    else {
      delta = -g * learning_rate;
    }
    weights[fv.first] += delta;
  }

  return log_loss;
}

adouble crf::train(const vector<vector<string> >& x, const vector<Derivation>& y,
    const vector<vector<Derivation> >& noise_samples, double learning_rate, double l2_strength) {
  assert(x.size() == y.size());
  stack->new_recording();
  adouble log_loss = 0.0;
  for (unsigned i = 0; i < x.size(); ++i) {
    log_loss += nce_loss(x[i], y[i], noise_samples[i]);
  }
  log_loss += l2penalty(l2_strength);

  log_loss.set_gradient(1.0);
  stack->compute_adjoint();
  for (auto& fv : weights) {
    const double g = fv.second.get_gradient();
    double delta;
    if (use_adadelta) {
      historical_gradients[fv.first] = rho * historical_gradients[fv.first] + (1 - rho) * g * g;
      delta = -g * sqrt(historical_deltas[fv.first] + epsilon) / sqrt(historical_gradients[fv.first] + epsilon);
      historical_deltas[fv.first] = rho * historical_deltas[fv.first] + (1 - rho) * delta * delta;
    }
    else {
      delta = -g * learning_rate;
    }
    weights[fv.first] += delta;
  }

  return log_loss;
}

void crf::add_feature(string name) {
  if (weights.find(name) == weights.end()) {
    weights[name] = 0.0;
    historical_deltas[name] = 1.0;
    historical_gradients[name] = 1.0;
  }
}

Derivation crf::combine(const vector<string>& x, const vector<unsigned>& indices, const vector<vector<tuple<adouble, string, string> > >& best_pieces, const vector<unsigned>& permutation) {
  Derivation d;
  assert(d.translations.size() == d.suffixes.size());
  for (unsigned i = 0; i < indices.size(); ++i) {
    auto& piece = best_pieces[i][indices[i]];
    const string& translation = get<1>(piece);
    const string& suffix = get<2>(piece);
    d.translations.push_back(translation);
    d.suffixes.push_back(suffix);
    assert(d.translations.size() == d.suffixes.size());
    if (translation.size() > 0) {
      d.permutation.push_back(i);
    }
  }
  assert(d.translations.size() == d.suffixes.size());
  assert(d.translations.size() == x.size());
  return d;
}

vector<tuple<double, Derivation> > crf::predict(const vector<string>& x, unsigned k) {
  bool verbose = false;
  suffix_list.insert("");
  // First we find the k-best (translation, suffix) pairs for each index in x
  vector<vector<tuple<adouble, string, string> > > best_pieces;
  for (unsigned i = 0; i < x.size(); ++i) {
    vector<tuple<adouble, string, string> > local_best_pieces;
    local_best_pieces.reserve(k + 1);

    string source = x[i];
    vector<string> translation_list;
    translation_list.push_back("");
    for (auto kvp : scorer->fwd_ttable->getTranslations(source)) {
      translation_list.push_back(kvp.first);
    }

    for (string target : translation_list) {
      map<string, double> translation_features = scorer->score_translation(source, target);
      adouble translation_score = dot(translation_features, weights);
      for (string suffix : suffix_list) {
        if (target.size() == 0 && suffix.size() != 0) {
          continue;
        }
        map<string, double> suffix_features = scorer->score_suffix(target, suffix);
        adouble suffix_score = dot(suffix_features, weights);
        adouble total_score = suffix_score + translation_score;
        if (local_best_pieces.size() < k ||
            total_score > get<0>(local_best_pieces[local_best_pieces.size() - 1])) {
          local_best_pieces.push_back(make_tuple(total_score, target, suffix));
          std::sort(local_best_pieces.rbegin(), local_best_pieces.rend());
          assert(local_best_pieces.size() <= k + 1);
          if (local_best_pieces.size() == k + 1) {
            local_best_pieces.pop_back();
          }
        }
      }
    }
    if (verbose) {
      cerr << "Best " << k << " candidates for word " << i << " (" << x[i] << "):" << endl;
      int i = 0;
      for (auto tup : local_best_pieces) {
        cerr << "\t" << i++ << " ||| " << get<1>(tup) << "+" << get<2>(tup) << " ||| " << get<0>(tup) << endl;
      }
    }
    best_pieces.push_back(local_best_pieces);
  }
  assert(best_pieces.size() == x.size());

  vector<tuple<double, Derivation> > kbest;
   
  // Now that we have the k-best (translation, suffix) pairs for each index,
  // run cube pruning to find our final k-best
  set<tuple<adouble, vector<unsigned> > > candidates;
  set<vector<unsigned> > used_index_sets;

  vector<unsigned> start(x.size(), 0);
  assert(start.size() == x.size()); 
  adouble start_score = 0.0;
  for (unsigned i = 0; i < x.size(); ++i) {
    assert(best_pieces[i].size() > 0);
    start_score += get<0>(best_pieces[i][0]);
  }
  candidates.insert(make_tuple(start_score, start));
 
  // TODO: Right now the list of candidates may be substantially bigger
  // than it needs to be.
  while (candidates.size() > 0 && kbest.size() < k) {
    // Pop the best candidate from the list and unpack it
    auto best_candidate = *(candidates.rbegin());
    vector<unsigned> indices;
    adouble score;
    do {
      best_candidate = *(candidates.rbegin());
      candidates.erase(best_candidate);
      score = get<0>(best_candidate);
      indices = get<1>(best_candidate);
      assert(indices.size() == x.size());
    } while(used_index_sets.find(indices) != used_index_sets.end() && candidates.size() > 0);
    if (used_index_sets.find(indices) != used_index_sets.end() && candidates.size() == 0) {
      break;
    }
    used_index_sets.insert(indices);

    // Make a derivation structure from the pieces and add it to kbest list
    Derivation d = combine(x, indices, best_pieces, vector<unsigned>());
    kbest.push_back(make_tuple(score.value(), d));
    if (verbose) {
      cerr << "next best derivation: " << d.toLongString() << " ||| " << score.value() << endl;
    }

    // Add any new candidates to the candidate list
    assert (indices.size() == x.size());
    for (unsigned i = 0; i < indices.size(); ++i) {
      vector<unsigned> new_indices(indices.begin(), indices.end());
      assert (new_indices.size() == indices.size());
      if (new_indices[i] + 1 < best_pieces[i].size()) {
        new_indices[i]++;
        assert (new_indices.size() == x.size());
        adouble new_score = score - get<0>(best_pieces[i][indices[i]])
            + get<0>(best_pieces[i][new_indices[i]]);
        candidates.insert(make_tuple(new_score, new_indices));
      }
    }
    //std::sort(candidates.rbegin(), candidates.rend());
    //candidates.erase(unique(candidates.begin(), candidates.end()), candidates.end());

    if (verbose) {
      cerr << "Current candidate list:" << endl;
      int i = 0;
      for (auto candidate : candidates) {
        adouble score = get<0>(candidate);
        vector<unsigned> indices = get<1>(candidate);
        assert(indices.size() == x.size());
        Derivation d = combine(x, indices, best_pieces, vector<unsigned>());
        cerr << "\t" << i++ << " ||| " << d.toLongString() << " ||| " << score << endl;
      }
    }
  }
  return kbest;
}

crf::crf() {}

crf crf::ReadFromFile(adept::Stack* stack, feature_scorer* scorer, const string& filename) {
  crf model;
  ifstream ifs(filename);
  boost::archive::text_iarchive ia(ifs);
  ia & model;

  model.stack = stack;
  model.scorer = scorer;

  return model;
}

void crf::WriteToFile(const string& filename) const {
  ofstream ofs(filename);
  boost::archive::text_oarchive oa(ofs);
  oa & *this;
}
