/* mex-C: 
 * compute SUM2 maps for a *single* image, and find the local maximum
 *
 * Usage:
 *    S2Map = mexc_ComputeSUM2( numGaborOri, M1Map, S2Template, subsampleS2, locationPerturbationFraction, templateSize, templateLoc  );
 *
 * S2Template is a matlab struct with the following fields:
 *	selectedRow: single array
 *	selectedCol: single array
 *	selectedOri: single array
 *	selectedScale: single array
 *	selectedLambda: single array
 *	selectedLogZ: single array
 *	
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mex.h"        /* the algorithm is connected to matlab */
#include "math.h"
#include "matrix.h"
#define ABS(x) ((x)>0? (x):(-(x)))
#define MAX(x, y) ((x)>(y)? (x):(y))
#define MIN(x, y) ((x)<(y)? (x):(y))
#define NEGMAX -1e10

/* variable declaration */
const float **M1Map;            /* MAX 1 maps */
const mxArray* S2Template;      /* SUM2 templates */
int nT;                         /* number of S2 templates */
int* nElement;                  /* number of elements for each template */
int subsampleS2;                /* step length when scanning the SUM2 template over MAX1 map */
int heightS2Map, widthS2Map;    /* (down-sampled) size of SUM2 maps */
int heightM1Map, widthM1Map;    /* size of MAX1 maps */
int nGaborScale;
int nGaborOri;
float locationPerturbRadius;    /* allowed location perturbation, relative to the template size */
int *nNeighborTemplate;         /* number of neighbor templates per template */
int nTemplate;                  /* number of S2 templates */
const int* templateSize;		/* size of the template */
const int* templateLoc;			/* location of the template */
int bestS2Row, bestS2Col, bestS2Template; /* the activation vector */
float bestS2Score;

void compute()
{
    int iT, iF;
    const mxArray* pA, *pA2;
    /* Note: we assume that iOri and iS starts from 0. */
    /* Note: we also assume that iRow and iCol is relative to (0,0). */
    int iRowS2, iColS2;
    int iOriM1, iRowM1, iColM1, iScaleM1;
    const float *selectedRow, *selectedCol;
    const float *selectedOri, *selectedScale;
    const float *selectedLambda, *selectedLogZ;
    int rowOffset, colOffset;
    mxClassID datatype;

	srand(time(NULL)); /* intialize random number generator */
	int nstep;	/* compute the search interior */
	nstep = (int)floor(templateSize[0]*locationPerturbRadius/2.0/(float)subsampleS2);
	int top = templateLoc[0] - nstep * subsampleS2;
	int bottom = templateLoc[0] + nstep * subsampleS2;
	nstep = (int)floor(templateSize[1]*locationPerturbRadius/2.0/(float)subsampleS2);
	int left = templateLoc[1] - nstep * subsampleS2;
	int right = templateLoc[1] + nstep * subsampleS2;

    /*
     * compute number of elements for the templates
     */
    nElement = (int*)mxCalloc( nT, sizeof(*nElement) );
    for( iT = 0; iT < nT; ++iT )
    {
        pA = mxGetCell( S2Template, iT );
        pA = mxGetField( pA, 0, "selectedRow" );
        nElement[iT] = mxGetM( pA ) * mxGetN( pA );
    }
    
    /* About the visiting order in the FOR loop:
     *      The scan over M1 map positions should be inner-most.
     */
	bestS2Score = NEGMAX;
    for( iT = 0; iT < nT; ++iT )
    {
        pA = mxGetCell( S2Template, iT );
		pA2 = mxGetField( pA, 0, "selectedRow" );
        selectedRow = (const float*)mxGetPr(pA2);
		pA2 = mxGetField( pA, 0, "selectedCol" );
        selectedCol = (const float*)mxGetPr(pA2);
        pA2 = ( mxGetField( pA, 0, "selectedOri" ) );
        selectedOri = (const float*)mxGetPr(pA2);
        pA2 = ( mxGetField( pA, 0, "selectedScale" ) );
        selectedScale = (const float*)mxGetPr(pA2);
        pA2 = ( mxGetField( pA, 0, "selectedLambda" ) );
        selectedLambda = (const float*)mxGetPr(pA2);
        pA2 = ( mxGetField( pA, 0, "selectedLogZ" ) );
        selectedLogZ = (const float*)mxGetPr(pA2);
           
		for( iColS2 = left; iColS2 <= right; iColS2 += subsampleS2 )
		{
			iColM1 = iColS2 * subsampleS2 + (int)selectedCol[iF];
			for( iRowS2 = top; iRowS2 <= bottom; iRowS2 += subsampleS2 )
			{
				float S2Score = (float)( rand() % 1000 ) / 1e10f;
				for( iF = 0; iF < nElement[iT]; ++iF )
				{
					iOriM1 = (int)selectedOri[iF];
					iScaleM1 = (int)selectedScale[iF];
					iRowM1 = iRowS2 + selectedRow[iF];
					iColM1 = iColS2 + selectedCol[iF];
					if( iRowM1 >= 0 && iRowM1 < heightM1Map && iColM1 >= 0 && iColM1 < widthM1Map )
					{
						S2Score += -selectedLogZ[iF] + selectedLambda[iF] * M1Map[iOriM1+iScaleM1*nGaborOri][iRowM1+iColM1*heightM1Map];
					}
					else
					{
						S2Score += -selectedLogZ[iF] + selectedLambda[iF] * (-10.0);
					}
				}
				if( S2Score > bestS2Score )
				{
					bestS2Score = S2Score;
					bestS2Row = iRowS2;
					bestS2Col = iColS2;
					bestS2Template = iT;
				}
			}
		}
    }
}

/* mex function is used to pass on the pointers and scalars from matlab, 
   so that heavy computation can be done by C, which puts the results into 
   some of the pointers. After that, matlab can then use these results. 
   
   So matlab is very much like a managing platform for organizing the 
   experiments, and mex C is like a work enginee for fast computation. */

void mexFunction(int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray *prhs[])                
{
    int ind, i, x, y, o, dataDim, j, bytes_to_copy, nGaborFilter;
    const mxArray *f;
    const mxArray *pAM1Map;
	mxArray *pA;
    mwSize ndim;
    const mwSize* dims;
    mwSize dimsOutput[2];
    mxClassID datatype;
    int iT;

	/*
	 * input variable 0: nGaborOri
	 */
	nGaborOri = (int)mxGetScalar(prhs[0]);
 
    /*
	 * input variable 1: M1 maps
	 */
    pAM1Map = prhs[1];
    dims = mxGetDimensions(pAM1Map);
    nGaborFilter = dims[0] * dims[1];
    nGaborScale = nGaborFilter / nGaborOri;
 
    M1Map = (const float**)mxCalloc( nGaborFilter, sizeof(*M1Map) );   /* MAX1 maps */
    for (i=0; i<nGaborFilter; ++i)
    {
        f = mxGetCell(pAM1Map, i);
        datatype = mxGetClassID(f);
        if (datatype != mxSINGLE_CLASS)
            mexErrMsgTxt("warning !! single precision required.");
        M1Map[i] = (const float*)mxGetPr(f);    /* get the pointer to cell content */
        heightM1Map = mxGetM(f);    /* overwriting is ok, since it is constant */
        widthM1Map = mxGetN(f);
    }

    /*
     * input variable 2: S2 templates
     */
    S2Template = prhs[2];
    nT = mxGetM(S2Template) * mxGetN(S2Template);
    
    /*
     * input variable 3: subsampleS2
     */
    subsampleS2 = (int)mxGetScalar(prhs[3]);
    
	/*
     * input variable 4: location shift radius
     */
    locationPerturbRadius = (float)mxGetScalar(prhs[4]);
	
	/*
     * input variable 5: size of the template ([height, width])
     */
    datatype = mxGetClassID(prhs[5]);
    if( datatype != mxINT32_CLASS )
        mexErrMsgTxt("warning !! int32 data type required for templateSize");
    templateSize = (const int*)mxGetPr(prhs[5]);
	
	/*
     * input variable 6: size of the template ([height, width])
     */
    datatype = mxGetClassID(prhs[6]);
    if( datatype != mxINT32_CLASS )
        mexErrMsgTxt("warning !! int32 data type required for templateLoc");
    templateLoc = (const int*)mxGetPr(prhs[6]);

    compute();
    
    
    /* =============================================
     * Handle output variables.
     * ============================================= 
     */
    /*
     * output variable 0: activation ([x,y,template,score])
     */
    dimsOutput[0] = 4; dimsOutput[1] = 1;
	pA = mxCreateNumericArray( 2, dimsOutput, mxSINGLE_CLASS, mxREAL );
	float* start_of_pr = (float*)mxGetData(pA);
	start_of_pr[0] = bestS2Row;
	start_of_pr[1] = bestS2Col;
	start_of_pr[2] = bestS2Template;
	start_of_pr[3] = bestS2Score;
	plhs[0] = pA;
}

