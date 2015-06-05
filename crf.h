#pragma once
#include <vector>
#include <string>
#include <map>
#include <tuple>
#include <boost/serialization/map.hpp>
#include <boost/serialization/unordered_set.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "adept.h"
#include "derivation.h"
#include "utils.h"
#include "feature_scorer.h"
using std::string;
using std::vector;
using std::map;
using std::tuple;
using adept::adouble;

class crf {
public:
  crf(adept::Stack* stack, feature_scorer* scorer);
  adouble dot(const map<string, double>& features, const map<string, adouble>& weights);
  adouble score(const vector<string>& x, const Derivation& y);
  adouble word_partition_function(const string& source);
  adouble partition_function(const vector<string>& x);
  adouble lattice_partition_function(const vector<string>& x);
  adouble slow_partition_function(const vector<string>& x,
    const map<string, adouble>& weights);
  adouble score_noise(const vector<string>& x, const Derivation& y);
  adouble nce_loss(const vector<string>& x, const Derivation& y, const vector<Derivation>& n);

  adouble l2penalty(const double lambda);
  adouble train_nobatch(const vector<vector<string>>& x, const vector<vector<Derivation> >& z, double learning_rate, double l2_strength);
  adouble train(const vector<vector<string>>& x, const vector<vector<Derivation> >& z, double learning_rate, double l2_strength);
  adouble train(const vector<vector<string>>& x, const vector<Derivation>& z, double learning_rate, double l2_strength);
  adouble train(const vector<vector<string>>& x, const vector<Derivation>& z, const vector<vector<Derivation> >& noise_samples, double learning_rate, double l2_strength);
  void add_feature(string name);

  Derivation combine(const vector<string>& x, const vector<unsigned>& indices,
    const vector<vector<tuple<adouble, string, string> > >& best_pieces,
    const vector<unsigned>& permutation);
  vector<tuple<double, Derivation> > predict(const vector<string>& x, unsigned k=1);

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & weights;
    ar & suffix_list;
  }
  static crf ReadFromFile(adept::Stack* stack, feature_scorer* scorer, const string& filename);
  void WriteToFile(const string& filename) const;
//private:
  map<string, adouble> weights;
  unordered_set<string> suffix_list;
private:
  crf();
  adept::Stack* stack;
  feature_scorer* scorer;
  map<string, double> historical_deltas;
  map<string, double> historical_gradients;
  const double rho = 0.95;
  const double epsilon = 1.0e-6;
};

