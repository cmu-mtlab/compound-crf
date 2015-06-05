CC = g++
DEBUG = -g -O3
CFLAGS = -Wall -Wextra -pedantic -Wno-unused-parameter -Wno-unused-variable -std=c++11 -c $(DEBUG) -I/Users/austinma/git/cpyp
LFLAGS = -Wall -Wextra -pedantic -Wno-unused-variable -Wno-unused-parameter -std=c++11 -ladept -lboost_serialization $(DEBUG)

all: crf split score reachable decoder

CRF_OBJECTS = main.o crf.o utils.o ttable.o feature_scorer.o compound_analyzer.o noise_model.o derivation.o NeuralLM/vocabulary.o NeuralLM/neurallm.cc
crf: $(CRF_OBJECTS)
	$(CC) $(CRF_OBJECTS) $(LFLAGS) -o crf

DECODER_OBJECTS = decoder.o utils.o ttable.o feature_scorer.o compound_analyzer.o derivation.o NeuralLM/vocabulary.o NeuralLM/neurallm.cc
decoder: $(DECODER_OBJECTS)
	$(CC) $(DECODER_OBJECTS) $(LFLAGS) -o decoder

SPLIT_OBJECTS = split.o ttable.o utils.o feature_scorer.o compound_analyzer.o derivation.o NeuralLM/vocabulary.o NeuralLM/neurallm.o
split: $(SPLIT_OBJECTS) 
	$(CC) $(SPLIT_OBJECTS) $(LFLAGS) -o split

SCORE_OBJECTS = score.o ttable.o utils.o feature_scorer.o crf.o derivation.o NeuralLM/vocabulary.o NeuralLM/neurallm.o
score: $(SCORE_OBJECTS)
	$(CC) $(SCORE_OBJECTS) $(LFLAGS) -o score

REACHABLE_OBJECTS = reachable.o crf.o utils.o ttable.o feature_scorer.o compound_analyzer.o noise_model.o derivation.o NeuralLM/vocabulary.o NeuralLM/neurallm.cc
reachable: $(REACHABLE_OBJECTS)
	$(CC) $(REACHABLE_OBJECTS) $(LFLAGS) -o reachable

reachable.o: reachable.cc crf.h utils.h feature_scorer.h compound_analyzer.h noise_model.h derivation.h
	$(CC) $(CFLAGS) reachable.cc

NeuralLM/vocabulary.o: NeuralLM/vocabulary.cc NeuralLM/vocabulary.h
	$(CC) $(CFLAGS) NeuralLM/vocabulary.cc -o NeuralLM/vocabulary.o

NeuralLM/neurallm.o: NeuralLM/neurallm.cc NeuralLM/neurallm.h NeuralLM/context.h NeuralLM/utils.h NeuralLM/param.h
	$(CC) $(CFLAGS) NeuralLM/neurallm.cc -o NeuralLM/neurallm.o

split.o: split.cc utils.h ttable.h feature_scorer.h compound_analyzer.h derivation.h
	$(CC) $(CFLAGS) split.cc

score.o: score.cc crf.h utils.h feature_scorer.h derivation.h
	$(CC) $(CFLAGS) score.cc

noise_model.o: noise_model.cc noise_model.h ttable.h utils.h derivation.h
	$(CC) $(CFLAGS) noise_model.cc

ttable.o: ttable.cc ttable.h utils.h
	$(CC) $(CFLAGS) ttable.cc

utils.o: utils.cc utils.h
	$(CC) $(CFLAGS) utils.cc

derivation.o: derivation.cc derivation.h
	$(CC) $(CFLAGS) derivation.cc

feature_scorer.o: feature_scorer.cc ttable.h feature_scorer.h derivation.h NeuralLM/neurallm.h NeuralLM/context.h
	$(CC) $(CFLAGS) feature_scorer.cc

compound_analyzer.o: compound_analyzer.cc compound_analyzer.h utils.h ttable.h derivation.h
	$(CC) $(CFLAGS) compound_analyzer.cc

main.o: main.cc crf.h utils.h feature_scorer.h compound_analyzer.h noise_model.h derivation.h
	$(CC) $(CFLAGS) main.cc

crf.o: crf.cc crf.h utils.h feature_scorer.h derivation.h
	$(CC) $(CFLAGS) crf.cc

decoder.o: decoder.cc utils.h feature_scorer.h derivation.h
	$(CC) $(CFLAGS) decoder.cc

clean:
	rm -f ./crf
	rm -f ./split
	rm -f ./score
	rm *.o
	rm -f NeuralLM/*.o
