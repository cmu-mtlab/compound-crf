#include <set>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <cassert>
#include <fstream>
#include <sstream>

#include <execinfo.h>
#include <signal.h>
#include <unistd.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "adept.h"
#include "crf.h"
#include "feature_scorer.h"
#include "compound_analyzer.h"
#include "noise_model.h"

using namespace std;
using adept::adouble;

const double eta = 0.01;
const double lambda = 0.0;
const int num_noise_samples = 100;

void read_input_file(string filename, vector<vector<string> >& X, vector<string>& Y) {
  ifstream f(filename);
  if (!f.is_open()) {
    cerr << "Unable to read from file " << filename << "." << endl;
    exit(1);
  }

  string line;
  vector<string> source;
  while (getline(f, line)) {
    stringstream sstream(line);
    string word;
    while (sstream >> word) {
      source.push_back(to_lower_case(word));
    }
    if (source.size() > 0) {
      string target = source[source.size() - 1];
      source.pop_back();
      X.push_back(source);
      Y.push_back(target);
      source.clear();
    }
  }
}

Derivation sample_derivation(crf* model, const vector<string>& input, const vector<Derivation>& derivations) {
  vector<adouble> scores(derivations.size(), 0.0);
  adouble sum = 0.0;
  for (unsigned i = 0; i < derivations.size(); ++i) {
    adouble score = model->score(input, derivations[i]);
    score = exp(score);
    scores[i] = score;
    sum += score; 
  }

  adouble r = sum * (double)rand() / RAND_MAX;
  for (unsigned i = 0; i < derivations.size(); ++i) {
    if (r < scores[i]) {
      return derivations[i];
    }
    r -= scores[i];
  }

  assert(false);
}

vector<Derivation> sample_derivations(crf* model, const vector<vector<string> >& inputs, const vector<vector<Derivation> >& derivations) {
  assert (inputs.size() == derivations.size());

  vector<Derivation> sampled_derivations;
  for (unsigned i = 0; i < inputs.size(); ++i) { 
    const Derivation& sample = sample_derivation(model, inputs[i], derivations[i]);
    sampled_derivations.push_back(sample);
  }
  assert (sampled_derivations.size() == inputs.size());
  return sampled_derivations;
}

void exception_handler(int sig) {
  void *array[10];
  int size = backtrace(array, 10);

  cerr << "Error: signal " << sig << ":" << endl;
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

void test(int argc, char** argv) {
  adept::Stack stack;

  ttable fwd_ttable;
  ttable rev_ttable;
  fwd_ttable.load(argv[2]);
  rev_ttable.load(argv[3]);
  vocabulary lm_vocab = vocabulary::ReadFromFile(argv[4]);
  NeuralLM lm = NeuralLM::ReadFromFile(argv[5]);
  feature_scorer scorer(&fwd_ttable, &rev_ttable);
  compound_analyzer analyzer(&fwd_ttable);
  scorer.lm_vocab = &lm_vocab;
  scorer.lm = &lm;

  crf model(&stack, &scorer);
  model.add_feature("length");
  model.add_feature("fwd_score");
  model.add_feature("rev_score");
  model.add_feature("tgt_null");
  model.add_feature("null_score");
  model.add_feature("lm_score");
  model.add_feature("lm_oov");
  model.add_feature("monotone");
  model.add_feature("suffix_n"); 
  model.add_feature("suffix_");
  model.add_feature("tomato_to_null");
  model.add_feature("processing_to_null");
  model.suffix_list.insert("");
  model.suffix_list.insert("n");

  vector<string> input {"tomato", "processing"}; 

  cerr << "Computing fast partition function..." << endl;
  adouble fast = model.partition_function(input);
  cerr << "Fast partition function: " << fast << endl;

  cerr << "Computing slow partition function..." << endl;
  adouble slow = model.slow_partition_function(input, model.weights);
  cerr << "Slow partition function: " << slow << endl;

  double diff = abs(fast.value() - slow.value());
  assert (diff < 1.0e-6);
}

int main(int argc, char** argv) {
  signal(SIGSEGV, exception_handler);
  if (argc != 6) {
    cerr << "Usage: " << argv[0] << " train.txt fwd_ttable rev_ttable target.vcb target.nlm" << endl;
    cerr << "where target.vcb is a character level vocabulary file" << endl;
    return 1;
  }

  // Quick sanity check
  //test(argc, argv); 

  // read training data
  vector<vector<string> > train_source;
  vector<string> train_target;
  vector<vector<Derivation> > train_derivations;
  read_input_file(argv[1], train_source, train_target);

  cerr << "Successfully read " << train_source.size() << " training instances." << endl;
  assert (train_source.size() == train_target.size());

  // Read in the ttables
  cerr << "Loading ttables..." << endl;
  ttable fwd_ttable;
  ttable rev_ttable;
  fwd_ttable.load(argv[2]);
  rev_ttable.load(argv[3]);

  cerr << "Loading LM..." << endl;
  adept::Stack stack;
  vocabulary lm_vocab = vocabulary::ReadFromFile(argv[4]);
  NeuralLM lm = NeuralLM::ReadFromFile(argv[5]);

  feature_scorer scorer(&fwd_ttable, &rev_ttable);
  compound_analyzer analyzer(&fwd_ttable);

  scorer.lm_vocab = &lm_vocab;
  scorer.lm = &lm;

  // Analyze the target side of the training corpus into lists of possible derivations
  cerr << "Analyzing training data..." << endl;
  for (unsigned i = 0; i < train_source.size(); ++i) {
    cerr << i << "/" << train_source.size() << "\r";
    vector<Derivation> derivations = analyzer.analyze(train_source[i], train_target[i]);
    for (Derivation& d : derivations) {
      if (d.toString() != train_target[i]) {
        cerr << "source: ";
        for (unsigned j = 0; j < train_source[i].size(); ++j) {
          cerr << train_source[i][j] << " ";
        }
        cerr << endl;
        cerr << d.toLongString() << endl;
        cerr << "derivation: " << d.toString() << endl;
        cerr << "target: " << train_target[i] << endl;
      }
      assert (d.toString() == train_target[i]);
    }
    train_derivations.push_back(derivations);
  }
  cerr << train_source.size() << "/" << train_source.size() << "\n";

  cerr << "Removing unreachable compounds..." << endl;
  // Remove any unreachable references from the training data
  for (unsigned i = 0; i < train_source.size(); ++i) {
    cerr << i << "/" << train_source.size() << "\r";
    if (train_derivations[i].size() == 0) { 
      train_source.erase(train_source.begin() + i);
      train_target.erase(train_target.begin() + i); 
      train_derivations.erase(train_derivations.begin() + i);
      i--;
    }
  }
  cerr << train_source.size() << "/" << train_source.size() << "\n";
 
  cerr << "Initializing model..." << endl;
  crf model(&stack, &scorer);
  //noise_model noise_generator(&fwd_ttable);

  // Preload features into the CRF to avoid adept errors
  cerr << "Preloading features..." << endl;
  model.add_feature("length");
  model.add_feature("null_score");
  model.add_feature("fwd_score");
  model.add_feature("rev_score");
  model.add_feature("tgt_null");
  model.add_feature("monotone");
  model.add_feature("lm_score");
  model.add_feature("lm_oov");
  for (unsigned i = 0; i < train_source.size(); ++i) {
    cerr << i << "/" << train_source.size() << "\r";
    for (unsigned j = 0; j < train_source[i].size(); ++j) {
      model.add_feature(train_source[i][j] + "_to_null");
    }
    for (Derivation& derivation : train_derivations[i]) {
      for (string suffix : derivation.suffixes) {
        model.suffix_list.insert(suffix);
        model.add_feature("suffix_" + suffix);
      }
    }
  }
  cerr << train_source.size() << "/" << train_source.size() << "\n";

  /*vector<vector<Derivation> > noise_samples; 
  for (int i = 0; i < train_source.size(); ++i) {
    vector<Derivation> samples;
    for (int j = 0; j < num_noise_samples; ++j) {
      Derivation sample = noise_generator.sample(train_source[i]);
      samples.push_back(sample);
    }
    noise_samples.push_back(samples);
  }*/

  assert (train_source.size() == train_target.size());
  assert (train_source.size() == train_derivations.size());
  cerr << train_source.size() << " reachable examples remain." << endl;

  /*train_derivations[0].erase(train_derivations[0].begin());
  train_derivations[0].erase(train_derivations[0].begin());
  train_derivations[1].erase(train_derivations[1].begin());
  train_derivations[1].erase(train_derivations[1].begin() + 1);*/
  /*train_derivations[2].erase(train_derivations[2].begin());
  train_derivations[2].erase(train_derivations[2].begin());
  train_derivations[2].erase(train_derivations[2].begin());
  train_derivations[4].erase(train_derivations[4].begin());*/

  adouble loss;
  /*vector<Derivation> chosen_derivations = sample_derivations(&model, train_source, train_derivations);
  assert (chosen_derivations.size() == train_source.size());
  for (int i = 0; i < train_source.size(); ++i) {
    for (int j = 0; j < train_source[i].size(); ++j) {
      //cerr << train_source[i][j] << " "; 
    }
    //cerr << "||| " << chosen_derivations[i].toLongString() << endl;
  }*/

  loss = 0.0;
  /*for (unsigned i = 0; i < train_source.size(); ++i) {
    loss += model.nce_loss(train_source[i], chosen_derivations[i], noise_samples[i]);
  }
  loss += model.l2penalty(lambda);
  cerr << "Iteration " << 0 << " loss: " << loss << endl;*/

  cerr << "Training..." << endl;
  for (unsigned iter = 0; iter < 0; ++iter) {
    //loss = model.train(train_source, chosen_derivations, noise_samples, eta, lambda);
    //loss = model.train(train_source, chosen_derivations, eta, lambda);
    //loss = model.train(train_source, train_derivations, eta, lambda);
    loss = model.train_nobatch(train_source, train_derivations, eta, lambda);
    cerr << "Iteration " << iter + 1 << " loss: " << loss << endl;
    cerr.flush();
  }

  cerr << "Final loss: " << loss << endl;
  cerr << "Final weights: " << endl;
  for (auto kvp : model.weights) {
    if (abs(kvp.second) > 0.0) {
      cerr << "  " << kvp.first << ": " << kvp.second << endl;
    }
  }
  cerr.flush();
  cerr << "Dumping model..." << endl;
  model.WriteToFile("model.crf");

  for (unsigned j = 0; j < train_source.size(); ++j) {
    vector<string>& input = train_source[j];
    double z = model.partition_function(input).value();
    cout << j << " ||| ";
    for (unsigned k = 0; k < train_source[j].size(); ++k) {
      cout << train_source[j][k] << " ";
    }
    cout << "||| " << train_target[j];
    cout << " ||| partition function: " << z << endl;

    for (Derivation& gold : train_derivations[j]) {
      map<string, double> features = scorer.score(input, gold);
      double score = model.dot(features, model.weights).value();
      cout << j << " ||| G ||| " << gold.toLongString(features) << "||| " << score << endl;
    }

    vector<tuple<double, Derivation> > kbest = model.predict(input, 10); 
    for (unsigned i = 0; i < kbest.size(); ++i) {
      double score = get<0>(kbest[i]);
      Derivation& derivation = get<1>(kbest[i]);
      map<string, double> features = scorer.score(input, derivation);
      score = model.dot(features, model.weights).value();
      cout << j << " ||| " <<  i << " ||| ";
      cout << derivation.toLongString(features) << "||| " << score << endl;
    }
    cout.flush();
  }
}
