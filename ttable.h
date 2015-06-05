#pragma once
#include <string>
#include <map>
#include <vector>

class ttable {
public:
  bool getScore(std::string source, std::string target, double& score);
  void setScore(std::string source, std::string target, double score);
  std::map<std::string, double> getTranslations(std::string source);
  void load(std::string filename);
private:
  std::map<std::string, std::map<std::string, double> > table;
};

