/*   Shared C code

     Copyright 2011-2020 Miron B. Kursa

     This file is part of rFerns R package.

 rFerns is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 rFerns is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 You should have received a copy of the GNU General Public License along with rFerns. If not, see http://www.gnu.org/licenses/.
*/

#include <stdint.h>
#include <limits.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

#ifdef IN_R
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/Utils.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>
#define PRINT Rprintf
#endif

typedef uint32_t uint;
typedef uint32_t mask;
typedef int32_t sint;
typedef double score_t;

#define MAX_D 16

struct parameters{
 uint numClasses;
 uint D;
 uint twoToD;
 uint numFerns;
 uint calcImp;
 uint holdForest;
 uint multilabel;
 uint consSeed;
 uint threads;
};
typedef struct parameters params;
#define PARAMS_ params *P
#define SIMP_ uint numC,uint D,uint twoToD,uint multi
#define SIMPP_ SIMP_,uint numFerns
#define _SIMP numC,D,twoToD,multi
#define _SIMPP _SIMP,numFerns
#define _SIMPPQ(x) (x).numClasses,(x).D,(x).twoToD,(x).multilabel,(x).numFerns


struct accLoss{
 double direct;
 double shadow;
};
typedef struct accLoss accLoss;

union threshold{
 double value;
 sint intValue;
 mask selection;
};
typedef union threshold thresh;

struct ferns{
 //All in series of D elements representing single fern
 int *splitAtts;
 thresh *thresholds;
 score_t *scores;
};
typedef struct ferns ferns;
#define FERN_ int *restrict splitAtts,thresh *restrict thresholds,score_t *restrict scores
#define _FERN splitAtts,thresholds,scores
#define _thFERN(e) &((ferns->splitAtts)[(e)*D]),&((ferns->thresholds)[(e)*D]),&((ferns->scores)[(e)*twoToD*numC])

struct attribute{
 void *x;
 sint numCat; // =      0 --> x is numerical and double*
              // =     -1 --> x is numerical and sint*
              //otherwise --> x is sint* and max(x)=numCat-1
};
typedef struct attribute att;
#define DATASET_ att *X,uint nX,uint *Y,uint N
#define _DATASET X,nX,Y,N

#define PREDSET_ att *X,uint nX,uint N
#define _PREDSET X,nX,N

struct model{
 ferns *forest;
 score_t *oobPreds;
 uint *oobOutOfBagC;
 double *imp;
 double *shimp;
 double *try;
};
typedef struct model model;

/*
 For speed, rFerns uses its own PRNG, PCG, which is seeded from R's PRNG each time an rFerns model is built.
 The generator is PCG32 by M.E. O'Neill https://www.pcg-random.org
*/

struct rng{
 uint64_t state;
 uint32_t stream;
};
typedef struct rng rng_t;
uint32_t __rintegerf(rng_t *rng){
 rng->state=rng->state*6364136223846793005+rng->stream;
 uint32_t rot=(rng->state)>>59;
 uint32_t s=(((rng->state)>>18)^(rng->state))>>27;
 return((s<<((-rot)&31))|(s>>rot));
}
void __setrng(rng_t *rng,uint64_t seed,uint64_t stream){
 rng->state=rng->stream=stream*2+1;
 rng->state+=seed;
 __rintegerf(rng);
}

#define RINTEGER __rintegerf(rng)
#define SETRNG(r,sta,str) __setrng((r),(sta),(str))
#define FETCH_SEED(r) ((r)->state)

//Fast & unbiased algorithm by Daniel Lemire https://arxiv.org/pdf/1805.10941.pdf
uint32_t __rindex(rng_t *rng,uint32_t upto){
 uint32_t x=__rintegerf(rng);
 uint64_t m=((uint64_t)x)*((uint64_t)upto);
 uint32_t l=((uint32_t)m);
 if(l<upto){
  uint32_t t=(-upto)%upto;
  while(l<t){
   x=__rintegerf(rng);
   m=((uint64_t)x)*((uint64_t)upto);
   l=((uint32_t)m);
  }
 }
 return(m>>32);
}
#define RINDEX(upTo) __rindex(rng,(upTo))
uint32_t __rmask(rng_t *rng,uint32_t numCat){
 if(numCat<3) return(1);
 return(__rindex(rng,(1<<(numCat-1))-1)+1);
}
#define RMASK(numCat) __rmask(rng,(numCat))

#define R_ rng_t *rng
#define _R rng

//Some bit fun
#define SET_BIT(where,which,to) (((where)&(~(1<<(which))))|((to)<<(which)))
#define GET_BIT(where,which) (((where)&(1<<(which)))>0)

//Sync PRNG with R; R will feel only two numbers were generated
#define EMERGE_R_FROM_R \
 GetRNGstate(); \
 uint64_t a=(uint32_t)(((double)(~((uint32_t)0)))*unif_rand()); \
 uint64_t b=(uint32_t)(((double)(~((uint32_t)0)))*unif_rand()); \
 PutRNGstate(); \
 a=(a<<32)+b; \
 rng_t rngdata; \
 rng_t *rng=&rngdata; \
 SETRNG(rng,a,1)

void makeBagMask(uint *bMask,uint N,R_){
 for(uint e=0;e<N;e++) bMask[e]=0;
 for(uint e=0;e<N;e++){
  bMask[RINDEX(N)]++;
 }
}

//The algorithm used to use random values; now, it just
//jumps one class further per object, so equal scores will lead to predictions as shown:
// 0 0 0 0 -> a
// 0 0 0 0 -> b
// 0 0 0 0 -> c
// 0 0 0 0 -> d
// 0 0 0 0 -> a etc.
uint whichMaxTieAware(score_t *where,uint N,uint jmp){
 score_t curMax=-INFINITY;
 uint b[N];
 uint be=UINT_MAX;
 for(uint e=0;e<N;e++)
  if(where[e]>curMax){
   be=0;
   b[be]=e;
   curMax=where[e];
  } else if(where[e]==curMax){
   be++;
   b[be]=e;
  }
 if(!be) return(b[0]);
 return(b[jmp%(be+1)]);
}

//Memory stuff
#ifndef FERNS_DEFINES
#define ALLOCN(what,oftype,howmany) oftype* what=(oftype*)R_alloc(sizeof(oftype),(howmany))
#define ALLOCNZ(what,oftype,howmany) oftype* what=(oftype*)R_alloc(sizeof(oftype),(howmany)); for(uint e_=0;e_<(howmany);e_++) what[e_]=(oftype)0
#define ALLOC(what,oftype,howmany) what=(oftype*)R_alloc(sizeof(oftype),(howmany))
#define ALLOCZ(what,oftype,howmany) {what=(oftype*)R_alloc(sizeof(oftype),(howmany));for(uint e_=0;e_<(howmany);e_++) what[e_]=(oftype)0;}
#define IFFREE(x) //Nothing
#define FREE(x) //Nothing
#define CHECK_INTERRUPT R_CheckUserInterrupt()
#endif

