/*==========================================================================
 * Copyright (c) 2001 Carnegie Mellon University.  All Rights Reserved.
 *
 * Use of the Lemur Toolkit for Language Modeling and Information Retrieval
 * is subject to the terms of the software license set forth in the LICENSE
 * file included with this software, and also available at
 * http://www.lemurproject.org/license.html
 *
 *==========================================================================
*/

/*! \page  Text Categorization Evaluation Application within light Lemur toolkit


Usage: TCEval parameter_file

Please refor to the namespace LocalParameter for setting the parameters within the parameter_file

 */


#include "common_headers.hpp"
#include "IndexManager.hpp"
#include "BasicDocStream.hpp"
#include "Param.hpp"
#include "String.hpp"
#include "IndexedReal.hpp"
#include "ScoreAccumulator.hpp"
#include "ResultFile.hpp"
#include "TextQueryRep.hpp"

using namespace lemur::api;

namespace LocalParameter{
  std::string databaseIndex; // the index of the documents
  std::string trainDocs;   // the file of query stream
  std::string testDocs;    // the name of the result file
  std::string resultFile;  // the weighting scheme
  void get() {
    // the string with quotes are the actual variable names to use for specifying the parameters
    databaseIndex    = ParamGetString("index"); 
    trainDocs      = ParamGetString("trainDocs");
    testDocs      = ParamGetString("testDocs");
    resultFile       = ParamGetString("result","res");
  }    
};


void GetAppParam() 
{
  LocalParameter::get();
}

void training(double *legiModel, double *spamModel, double &pSpam, 
	      Index &ind, ifstream &trainIFS)
{}

void estTrainModel(ifstream &trainIDFile, double *pWRelModel, double *pWIrrelModel, double &pRel, Index &ind){
  //estimate the naive bayes model from the training data
  int vocabSize=ind.termCountUnique();
  //initiate the value of two models
  for (int t=0; t<=vocabSize; t++){
    pWRelModel[t]=0;
    pWIrrelModel[t]=0;
  }
  
  int numTrainDocs=0;         //number of training documents
  int numRelTrainDocs=0;      //number of relevant (i.e.,spam) training documents
  int numWordRelTrainDocs=0;  //number of words in relevant training documents
  int numWordIrrelTrainDocs=0;//number of words in irrelevant training documents

  while (!trainIDFile.eof()){
    int Rel;
    char docIDStr[1000];
    trainIDFile>>docIDStr>>Rel;

    int docID=ind.document(docIDStr);
    numTrainDocs++;

    if (Rel==1){
      /*!!!!!! Implement the code to accumulate the number of relevant training documents !!!!!!*/     
	numRelTrainDocs++;
    }
    

    //go through every document to generate the count of words in each type of document
    TermInfoList *docTermList=ind.termInfoList(docID);
    docTermList->startIteration();
    while (docTermList->hasMore()){
      TermInfo *info=docTermList->nextEntry();
      int termFreq=info->count();
      int termID=info->termID();
      
      if (Rel==1){
	//this is a relevant document
        /*!!!!!!!!!! Implement the code to accumulate term counts for relevant model !!!!!!*/
	pWRelModel[termID] += termFreq;
	numWordRelTrainDocs += termFreq;
      }else{
	//this is not a relevant document
        /*!!!!!!!!!! Implement the code to accumulate term counts for irrelevant model !!!!!!*/
	pWIrrelModel[termID] += termFreq;
	numWordIrrelTrainDocs += termFreq;

      }      
    }   
    delete docTermList;
  }



  for (int t=0; t<=vocabSize; t++){
    /*!!!!!! Implement the code to normlize the relevant and irrelevant models (i.e. Sum_wP(w)=1 )  !!!!!!*/
    /*!!!!!! Please use smoothing method !!!!!!*/
	pWRelModel[t] = (1 + pWRelModel[t])/(vocabSize+1 + numWordRelTrainDocs );

	pWIrrelModel[t] = (1 + pWIrrelModel[t])/(vocabSize+1 +  numWordIrrelTrainDocs);

  }

  
  /*obtain prior for relevant model (i.e., spam)*/
  pRel=(double)numRelTrainDocs/numTrainDocs;

}


void  printTrainModel(double* pWRelModel, double* pWIrrelModel, double pRel, Index &ind){
  //print out the naive bayes model
  int vocabSize=ind.termCountUnique();
  IndexedRealVector wordVec;
  IndexedRealVector::iterator it;  
  

  cout<<"*****For Model Prior"<<endl;
  cout<<"Relevant Model:"<<pRel<<"    "<<"Irrelvant Model:"<<1-pRel<<endl;


  double pSumRel=0;
  double pSumIrrel=0;

  for (int t=0; t<=vocabSize; t++){
    pSumRel+=pWRelModel[t];
    pSumIrrel+=pWIrrelModel[t];
  }
  cout<<"Prob Sum is: "<<pSumRel<<" and "<< pSumIrrel<<endl;
  cout <<"Deok are you sure?" << endl;
  wordVec.clear();
  for (int t=0; t<=vocabSize; t++){
    wordVec.PushValue(t,pWRelModel[t]);
  }
  wordVec.Sort();


  cout<<"Top Words for the Relevant Mode"<<endl;
  int nTopWord=0;
  for (it=wordVec.begin();it!=wordVec.end();it++){
    nTopWord++;
    if (nTopWord>30){
      break;
    }
    cout<<"Top "<<nTopWord<<" "<<ind.term((*it).ind)<<" "<<(*it).val<<endl;
  }

  wordVec.clear();
  for (int t=0; t<=vocabSize; t++){
    wordVec.PushValue(t,pWIrrelModel[t]);
  }
  wordVec.Sort();

  cout<<"Top Words for the Irrelevant Mode"<<endl;
  nTopWord=0;
  for (it=wordVec.begin();it!=wordVec.end();it++){
    nTopWord++;
    if (nTopWord>30){
      break;
    }
    cout<<"Top "<<nTopWord<<" "<<ind.term((*it).ind)<<" "<<(*it).val<<endl;
  }

}



void  getTestRst(ifstream &testIDFile, double* pWRelModel, double* pWIrrelModel, double pRel, IndexedRealVector &results, Index &ind){
  //generate the test results
  int vocabSize=ind.termCountUnique();



  int numTestDoc=0;
  while (!testIDFile.eof()){
    char docIDStr[1000];
    testIDFile>>docIDStr;

    int docID=ind.document(docIDStr);

    double logRelProb=0;  //log probability (i.e., log-likelihood) given relevant model (i.e., spam)
    double logIrrelProb=0;//log probability (i.e., log-likelihood) given irrelvant model (i.e., non-spam)

    TermInfoList *docTermList=ind.termInfoList(docID);
    docTermList->startIteration();
    while (docTermList->hasMore()){
      TermInfo *info=docTermList->nextEntry();
      int termFreq=info->count();
      int termID=info->termID();

      /*!!!!!! Implement the code to accumuate log probability (i.e., log-likelihood) give relevant model and irrelevant model !!!!!!*/
  	logRelProb += termFreq * log(pWRelModel[termID]/pWIrrelModel[termID]);
//	logIrrelProb += termFreq * log(pWIrrelModel[termID]);
	

    }

    /*Calculate the probability of a document being relevant (i.e. outProb)*/
    /*!!!!!! Please use Bayes Rule; please incoporate the prior probability (i.e., pRel) into the calculating of factor in the next line!!!!!!*/
    double outProb;
    double logProb;
  //  double irrelProb;

//	logRelProb  = log(pRel) + logRelProb;
//	logIrrelProb = log(1-pRel) + logIrrelProb;

//	relProb = exp(logRelProb);
//	irrelProb = exp(logIrrelProb);

	logProb =  log(pRel /(1-pRel)) +  logRelProb;

	if (logProb > 700) {
		logProb = 700;
	}

	outProb = exp(logProb) / (1+exp(logProb));

	if(isnan(outProb)){

		cout << "Hello" <<endl;
	}
    results.PushValue(docID,outProb);
    numTestDoc++;
  }
}

void printTestRst(ofstream &rstFile, IndexedRealVector &results, Index &ind){
  //print out the test results

  IndexedRealVector::iterator it;  
  for (it=results.begin();it!=results.end();it++){
    rstFile<<ind.document((*it).ind)<<" "<<(*it).val<<endl;
  }

}

/// A retrieval evaluation program
int AppMain(int argc, char *argv[]) {
  

  //Step 1: Open the index file
  Index  *ind;

  try {
    ind  = IndexManager::openIndex(LocalParameter::databaseIndex);
  } 
  catch (Exception &ex) {
    ex.writeMessage();
    throw Exception("RelEval", "Can't open index, check parameter index");
  }

  //Step 2: Open the id file to get training and test documents
  ifstream trainIDFile;
  try {
    trainIDFile.open(LocalParameter::trainDocs.c_str());
  } 
  catch (Exception &ex) {
    ex.writeMessage(cerr);
    throw Exception("NBClassify", 
                    "Can't open train Document Files");
  }

  ifstream testIDFile;
  try {
    testIDFile.open(LocalParameter::testDocs.c_str());
  } 
  catch (Exception &ex) {
    ex.writeMessage(cerr);
    throw Exception("NBClassify", 
                    "Can't open test Document Files");
  }

  ofstream rstFile;
  try {
    rstFile.open(LocalParameter::resultFile.c_str());
  } 
  catch (Exception &ex) {
    ex.writeMessage(cerr);
    throw Exception("NBClassify", 
                    "Can't open result Files");
  }

  //Step 3: Training process to generate model parameters
  int vocabSize=ind->termCountUnique();
  double pWRelModel[vocabSize+1]; //p(W|Relevant Docs); for documents with "1" (spam)
  double pWIrrelModel[vocabSize+1]; //p(W|Irrelevant Docs); for documents with "0" (non-spam)
  double pRel;       //probability of relevant model (i.e., spam model)       


  estTrainModel(trainIDFile, pWRelModel, pWIrrelModel, pRel, *ind);
  printTrainModel(pWRelModel, pWIrrelModel, pRel, *ind);


  //Step 4: Test the performance
  IndexedRealVector results;

  results.clear();
  getTestRst(testIDFile, pWRelModel, pWIrrelModel, pRel, results, *ind); 

  printTestRst(rstFile, results, *ind);

  delete ind;
  return 0;
}

