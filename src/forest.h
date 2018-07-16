/*   Code handling fern ensembles -- creation, prediction, OOB, accuracy...

     Copyright 2011-2018 Miron B. Kursa

     This file is part of rFerns R package.

 rFerns is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 rFerns is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 You should have received a copy of the GNU General Public License along with rFerns. If not, see http://www.gnu.org/licenses/.
*/

void killModel(model *x);

model *makeModel(DATASET_,ferns *ferns,params *P,R_){
 uint scalarExecution=(P->threads==1),nt=P->threads;
 uint numC=P->numClasses;
 uint D=P->D;
 assert(D<=MAX_D);
 uint twoToD=P->twoToD;
 uint multi=P->multilabel;

 //=Allocations=//
 //Internal objects
 ALLOCN(_curPreds,score_t,numC*N*nt);
 ALLOCN(_bag,uint,N*nt);
 ALLOCN(_idx,uint,N*nt);

 //Output objects
 ALLOCN(ans,model,1);

 //OOB prediction stuff
 ALLOCZ(ans->oobPreds,score_t,numC*N*nt);
 score_t *_oobPredsAcc=ans->oobPreds;
 ALLOCZ(ans->oobOutOfBagC,uint,N*nt);
 uint *_oobPredsC=ans->oobOutOfBagC;

 //Stuff for importance
 uint *_buf_idxPerm=NULL;
 //Actual importance result
 ans->imp=NULL;
 ans->shimp=NULL;
 ans->try=NULL;
 //Allocate if needed only
 if(P->calcImp){
  ALLOCZ(ans->imp,double,nX);
  ALLOCZ(ans->shimp,double,nX);
  ALLOCZ(ans->try,double,nX);
  if(P->calcImp==1){
   ALLOC(_buf_idxPerm,uint,N*nt);
  }else if(P->calcImp==2){
   ALLOC(_buf_idxPerm,uint,N*nt*2);
  }else error("Somehow invalid importance flag; several internal logic breach. Please report.");
 }

 ans->forest=ferns;
 uint modelSeed=RINTEGER;

 //=Building model=//
 #pragma omp parallel for num_threads(nt)
 for(uint e=0;e<(P->numFerns);e++){
  uint tn=omp_get_thread_num(); //Number of the thread we're in
  uint *bag=_bag+(N*tn),*idx=_idx+(N*tn);
  score_t *curPreds=_curPreds+(N*numC*tn);
  score_t *oobPredsAcc=_oobPredsAcc+(N*numC*tn);
  uint *oobPredsC=_oobPredsC+(N*tn);
  int fernLoc=(P->holdForest)?e:tn;
  if(scalarExecution){
   CHECK_INTERRUPT; //Place to go though event loop, if such is present
  }

  rng_t _curFernRng,*curFernRng=&_curFernRng;
  SETSEEDEX(curFernRng,e+1,modelSeed);
  makeBagMask(bag,N,curFernRng);
  makeFern(_DATASET,_thFERN(fernLoc),bag,curPreds,idx,_SIMP,curFernRng);

  //Accumulating OOB errors, independently per thread
  for(uint ee=0;ee<N;ee++){
   oobPredsC[ee]+=!(bag[ee]);
   for(uint eee=0;eee<numC;eee++)
    oobPredsAcc[eee+numC*ee]+=((double)(!(bag[ee])))*curPreds[eee+numC*ee];
  }

  //Importance
  if(P->calcImp){
   /*
    For importance, we want to know which unique attributes were used to build it.
    Their number will be placed in numAC, and attC[0..(numAC-1)] will contain their indices.
   */
   uint attC[MAX_D];
   attC[0]=(ferns->splitAtts)[fernLoc*D];
   uint numAC=1;
   for(uint ee=1;ee<D;ee++){
    for(uint eee=0;eee<numAC;eee++)
     if((ferns->splitAtts)[fernLoc*D+ee]==attC[eee]) goto isDuplicate;
    attC[numAC]=(ferns->splitAtts)[fernLoc*D+ee]; numAC++;
    isDuplicate:
    continue;
   }

   if(P->calcImp==1){
    uint *buf_idxPermA=_buf_idxPerm+(tn*N);
    for(uint ee=0;ee<numAC;ee++){
     accLoss loss=calcAccLoss(_DATASET,attC[ee],_thFERN(fernLoc),bag,idx,curPreds,numC,D,curFernRng,buf_idxPermA);
     #pragma omp critical
     {
      ans->imp[attC[ee]]+=loss.direct;
      ans->try[attC[ee]]++;
     }
    }
   }else{
    uint *buf_idxPermA=_buf_idxPerm+(tn*N*2);
    uint *buf_idxPermB=_buf_idxPerm+(tn*N*2+N);
    for(uint ee=0;ee<numAC;ee++){
     accLoss loss=calcAccLossConsistent(_DATASET,attC[ee],_thFERN(fernLoc),bag,idx,curPreds,numC,D,curFernRng,P->consSeed,buf_idxPermA,buf_idxPermB);
     #pragma omp critical
     {
      ans->imp[attC[ee]]+=loss.direct;
      ans->shimp[attC[ee]]+=loss.shadow;
      ans->try[attC[ee]]++;
     }
    }
   }
  }
 }

 //=Finishing up=//
 //Finishing importance
 if(P->calcImp) for(uint e=0;e<nX;e++){
  if(ans->try[e]==0){
   ans->imp[e]=0.;
   ans->try[e]=0.;
   ans->shimp[e]=0.;
  }else{
   ans->imp[e]/=ans->try[e];
   if(P->calcImp==2){
    ans->shimp[e]/=ans->try[e];
   }else{
    //This is probably redundant
    ans->shimp[e]=0.;
   }
  }
 }

 //Collecting OOB in parallel case
 if(nt!=1) for(int e=0;e<N;e++){
  //Loop over threads; we accumulate to tn 0, so from 1
  for(int tn=1;tn<nt;tn++){
   //Rprintf("%d/%d %d/%d\n",e,N,tn,nt);
   for(int ee=0;ee<numC;ee++)
    _oobPredsAcc[e*numC+ee]+=_oobPredsAcc[tn*N*numC+e*numC+ee];
   _oobPredsC[e]+=_oobPredsC[tn*N+e];
  }
 }
 //Releasing memory
 FREE(_bag); FREE(_curPreds); FREE(_idx);
 FREE(_buf_idxPerm);
 return(ans);

 #ifndef IN_R
  allocFailed:
  killModel(ans);
  IFFREE(_bag); IFFREE(_curPreds); IFFREE(_idx);
  IFFREE(_buf_idxPerm);
  return(NULL);
 #endif
}

void predictWithModelSimple(PREDSET_,ferns *x,uint *ans,SIMPP_,double *sans,R_){
 ferns *ferns=x;
 for(uint e=0;e<numC*N;e++)
  sans[e]=0.;
 //Use ans memory as idx buffer
 uint *idx=ans;
 for(uint e=0;e<numFerns;e++){
  predictFernAdd(
   _PREDSET,
   _thFERN(e),
   sans,
   idx,
   _SIMP);
 }
 if(!multi){
  for(uint e=0;e<N;e++)
   ans[e]=whichMaxTieAware(&(sans[e*numC]),numC,e);
 }else{
  for(uint e=0;e<numC;e++)
   for(uint ee=0;ee<N;ee++)
    ans[e*N+ee]=sans[ee*numC+e]>0.;
 }
}

void predictWithModelScores(PREDSET_,ferns *x,double *ans,SIMPP_,uint *idx){
 ferns *ferns=x;
 for(uint e=0;e<numC*N;e++)
  ans[e]=0.;
 for(uint e=0;e<numFerns;e++)
  predictFernAdd(
   _PREDSET,
   _thFERN(e),
   ans,
   idx,
   _SIMP);
}

void killModel(model *x){
 if(x){
  IFFREE(x->oobPreds);
  IFFREE(x->oobOutOfBagC);
  IFFREE(x->oobErr);
  IFFREE(x->imp);
  IFFREE(x->shimp);
  IFFREE(x->try);
  FREE(x);
 }
}
