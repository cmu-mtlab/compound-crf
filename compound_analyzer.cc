#include <iostream>
#include <algorithm>
#include <cassert>
#include "compound_analyzer.h"
using namespace std;

compound_analyzer::compound_analyzer(ttable* fwd_ttable) {
  this->fwd_ttable = fwd_ttable;
}

// Takes in a set of translations and an ordering
// and finds the suffixes necessary to create the compound word given
// out of the pieces given, in the order given.
// If this is not possible, returns false. If possible, returns true.
bool compound_analyzer::decompose(string compound, const vector<string>& pieces,
    vector<unsigned> permutation, vector<string>& suffixes) {
  suffixes.clear();
  suffixes.resize(pieces.size());
  for (unsigned i = 0; i < suffixes.size(); ++i) {
    suffixes[i] = "";
  }

  string remainder = compound;
  for (unsigned i = 0; i < permutation.size(); ++i) {
    int j = permutation[i];
    size_t location = remainder.find(pieces[j]);
    if (location == string::npos) {
      return false;
    }

    string prefix = remainder.substr(0, location);
    if (i == 0 || pieces[i - 1].size() == 0) {
      if (prefix.size() > 0) {
        return false;
      }
    }
    else {
      suffixes[permutation[i - 1]] = prefix;
    }

    assert (location + pieces[j].size() <= remainder.size());
    remainder = remainder.substr(location + pieces[j].size());
  }

  suffixes[permutation[permutation.size() - 1]] = remainder;
  assert (suffixes.size() == pieces.size());
  return true;
}

vector<Derivation> compound_analyzer::analyze(const vector<string>& english, string german, bool verbose) {
  vector<Derivation> derivations;

  // Output the source and target just for clarity
  if (verbose) {
    cout << "Source: ";
    for (string w : english) {
      cout << w << " ";
    }
    cout << endl;
    cout << "Target: " << german << endl;
  }

  // For each english word, look up its list of possible translations,
  // and filter the list down to just the translations that actually
  // appear in the german word we were given
  vector<vector<string> > candidate_translations;
  for (string w : english) {
    map<string, double> translations = fwd_ttable->getTranslations(w);
    vector<string> matching_translations;
    matching_translations.push_back("");
    for (auto kvp : translations) {
      string t = kvp.first;
      if (german.find(t) != string::npos) {
        matching_translations.push_back(t);
      }
    }
    candidate_translations.push_back(matching_translations);
  }

  // Output the list of candidate translations for each english word
  // just for funsies
  if (verbose) {
    for (unsigned i = 0; i < english.size(); ++i) {
      cout << english[i] << endl;
      for (string t : candidate_translations[i]) {
        cout << "\t" << (t.size() > 0 ? t : "(null)") << endl;
      }
    }
  }

  // Loop over the cross product of possible translations
  for (vector<string> translations : cross(candidate_translations)) {
    // This variable will hold a permutation of the integers [0, |G|)
    // Note that we remove indices coresponding to NULL translations
    // since their ordering does not affect the output.
    vector<unsigned> indices;
    for (unsigned i = 0; i < translations.size(); ++i) {
      if (translations[i] != "") {
        indices.push_back(i);
      }
    }

    // Don't allow all the pieces to translate as NULL
    if (indices.size() == 0) {
      continue;
    }

    // Loop over all possible permutations.
    // Since we limit the English span length to 5, this
    // loop will run at most 5! = 120 times.
    do {
      vector<string> suffixes;
      if (!decompose(german, translations, indices, suffixes)) {
        continue;
      }

      bool valid = true;
      // Only allow derivations with at least two non-NULL pieces
      // If it only has one non-NULL piece, then it's a WORD not a COMPOUND
      valid &= (indices.size() >= 2);
      // Only allow derivations where suffixes are <= 2 characters in length
      /*for (string suffix : suffixes) {
        if (suffix.size() > 2) {
          valid = false;
          break;
        }
      }*/

      assert (translations.size() <= 5);
      assert (suffixes.size() == translations.size());
      assert (indices.size() <=  translations.size());
      if (valid) {
        Derivation derivation { translations, suffixes, indices };
        derivations.push_back(derivation);
      }

    } while (next_permutation(indices.begin(), indices.end()));
  }
  return derivations;
}

bool compound_analyzer::isReachable(const vector<string>& english, string german) {
  // For each english word, look up its list of possible translations,
  // and filter the list down to just the translations that actually
  // appear in the german word we were given
  vector<vector<string> > candidate_translations;
  for (string w : english) {
    map<string, double> translations = fwd_ttable->getTranslations(w);
    vector<string> matching_translations;
    matching_translations.push_back("");
    for (auto kvp : translations) {
      string t = kvp.first;
      if (t.size() >= 3 && german.find(t) != string::npos) {
        matching_translations.push_back(t);
      }
    }
    candidate_translations.push_back(matching_translations);
  }

  // Loop over the cross product of possible translations
  for (vector<string> translations : cross(candidate_translations)) {

    // This variable will hold a permutation of the integers [0, |G|)
    // Note that we remove indices coresponding to NULL translations
    // since their ordering does not affect the output.
    vector<unsigned> indices;
    for (unsigned i = 0; i < translations.size(); ++i) {
      if (translations[i] != "") {
        indices.push_back(i);
      }
    }

    // Don't allow all the pieces to translate as NULL
    if (indices.size() == 0) {
      continue;
    }

    // Loop over all possible permutations.
    // Since we limit the English span length to 5, this
    // loop will run at most 5! = 120 times.
    do {
      vector<string> suffixes;
      if (!decompose(german, translations, indices, suffixes)) {
        continue;
      }

      bool valid = true;
      // Only allow derivations with at least two non-NULL pieces
      // If it only has one non-NULL piece, then it's a WORD not a COMPOUND
      valid &= (indices.size() >= 2);
      assert (suffixes.size() == translations.size());
      if (valid) {
        return true;
      }

    } while (next_permutation(indices.begin(), indices.end()));
  }

  return false;
}

