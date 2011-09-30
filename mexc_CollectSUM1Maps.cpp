/*
	Collect SUM1 maps for learning. Deal with a single image.
	
	usage:
		[alignedS1 sampleSize] = mexc_CollectSUM1Maps( S1, M1, initializeOrNot, posteriorMap, searchInterior, numCluster, windowSize );
		
	S1 is the SUM1 map, and posteriorMap is the associated posterior cluster probabilities over scanned locations.
	searchInterior contains the information for the scanned positions inside the M1 map.
	
	Main steps: (this is done separately for each cluster)
	1) Compute the sum of posterior probabilities (subject to thresholding);
	2) Using the random seeds, find the locations by scanning over the posterior map again.
	3) Collect both SUM1 and MAX1.
	
*/

# include <cstdlib>
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include "mex.h"
# include "math.h"
# define PI 3.1415926
# define ABS(x) ((x)>0? (x):(-(x)))
# define MAX(x, y) ((x)>(y)? (x):(y))
# define MIN(x, y) ((x)<(y)? (x):(y))
# define ROUND(x) (floor((x)+.5))
# define NEGMAX -1e10

/* Generating float vector */
float *float_vector(int n)
{
    float *v; 
    v = (float*) mxCalloc (n, sizeof(float));    
    return v; 
}
/* Generating integer vector */
int *int_vector(int n)
{
    int *v; 
    v = (int*) mxCalloc (n, sizeof(int));
    return v; 
}
/* Generating float matrix */
float **float_matrix(int m, int n)
{
    float **mat; 
    int i; 
    mat = (float**) mxCalloc(m, sizeof(float*));
    for (i=0; i<m; i++)
        mat[i] = float_vector(n); 
    return mat; 
}
/* Generating integer matrix */
int **int_matrix(int m, int n)
{
    int **mat; 
    int i; 
    mat = (int**) mxCalloc(m, sizeof(int*));
    for (i=0; i<m; i++)
        mat[i] = int_vector(n); 
    return mat; 
}
/* Free matrix space */
void free_matrix(void **mat, int m, int n)
{
        int i;
        for (i=0; i<m; i++)
              mxFree(mat[i]);
        mxFree(mat);
}
/* Compute pixel index in the vector that stores image */
inline int px(int x, int y, int lengthx, int lengthy)  /* the image is lengthx*lengthy */
{
   return (x + (y-1)*lengthx - 1); 
}

/* key input and output variables */
int numOrient, nGaborFilter, nGaborScale;
int numCluster;
const float **SUM1map, **MAX1map;	/* MAX1 maps */
int halfFilterSize; /* filter size = 2*halfFilterSize + 1 */ 
int width, height; /* width and height of the input MAX1 map for the input image */
int verticalMargin, horizontalMargin, stepSize;	/* parameters for the search interior */
int winWidth, winHeight;	/* size of scanning window (template window size) */
int sampleSize;				/* number of scanned locations */
float** posteriorMap;		/* map of posterior probabilities at scanned positions, normalized per position */
int subsampledWidth, subsampledHeight; 		/* size of the posterior maps */
float* searchInterior;
const float backgroundPrior = 0.5;
float membershipThreshold;
// const float membershipThreshold = 1e-1; // examples with posterior probability smaller than this will not be considered
bool display = false;

const mxArray* p_allS1, *p_allM1, *p_weights;
const mxArray* p_patches;
float* srcIm;
int *currentIndPerCluster;
int *numExamplePerCluster;
float* currentMixingProb;
float** seedsPerCluster; /* require that seeds are ranked from low to high in each cluster */
	/* Its numerical range is determined by the sum of posterior probability maps, which is computed before calling this function. */

void Compute()
{

    // ========================================================================
    // Compute the sum of posterior probabilities (subject to thresholding) in this image
    // ========================================================================
    /*
    for( int c = 0; c < numCluster; ++c )
	{

        // the 2D size of the search interior:
        subsampledWidth = (int)floor( (double)( width - horizontalMargin * 2 ) / stepSize );
        subsampledHeight = (int)floor( (double)( height - verticalMargin * 2 ) / stepSize );
        for( int iColPost = 0; iColPost < subsampledWidth; ++iColPost ) // position in the posterior map
        {
            int iColM1 = horizontalMargin + iColPost * stepSize;
            if( iColM1 < 0 || iColM1 >= width )
            {
                mexErrMsgTxt( "iColM1 out of bound" );
            }
            for( int iRowPost = 0; iRowPost < subsampledHeight; ++iRowPost )
            {
                int iRowM1 = verticalMargin + iRowPost * stepSize;
                if( iRowM1 < 0 || iRowM1 >= width )
                {
                    mexErrMsgTxt( "iRowM1 out of bound" );
                }
                
                for( int cc = 0; cc < numCluster; ++cc )
                {
                    float val = posteriorMap[cc][iRowPost+iColPost*subsampledHeight]; 
                    if( val >= membershipThreshold )
                    {
                        currentMixingProb[cc] += val;
                    }
                }
            }
        } // up till now, mixingProb[] is not normalized
	}
    */
    
    // ========================================================
    // Scan the posterior map and collect examples
    // ========================================================
    
	// allocate space for the output aligned S1, M1 maps for all clusters
	
	mwSize dimsOutput[2];
	for( int c = 0; c < numCluster; ++c )
	{
		if(display)
		{
			mexPrintf("cluster=%d, %d\n",c,p_allM1);
		}
		mxArray* p_M1maps = mxGetCell(p_allM1,c);
		mxArray* p_S1maps = mxGetCell(p_allS1,c);
		
		float* weights = (float*)mxGetPr( mxGetCell(p_weights,c) );
		
		if( currentIndPerCluster[c] >= numExamplePerCluster[c] )
		{
			if( display )
				mexPrintf("cluster %d is full with %d examples.",c,numExamplePerCluster[c]);
			continue;
		}
		float currentSeed = seedsPerCluster[c][currentIndPerCluster[c]];
		bool allSeedsAreFound = false;

		/* scan over the posterior maps */
		
		// the 2D size of the search interior:
		subsampledWidth = (int)floor( (double)( width - horizontalMargin * 2 ) / stepSize );
		subsampledHeight = (int)floor( (double)( height - verticalMargin * 2 ) / stepSize );
		for( int iColPost = 0; iColPost < subsampledWidth && !allSeedsAreFound; ++iColPost ) // position in the posterior map
		{
			int iColM1 = horizontalMargin + iColPost * stepSize;
			if( iColM1 < 0 || iColM1 >= width )
			{
			    mexErrMsgTxt( "iColM1 out of bound" );
			}
			for( int iRowPost = 0; iRowPost < subsampledHeight && !allSeedsAreFound; ++iRowPost )
			{
				int iRowM1 = verticalMargin + iRowPost * stepSize;
			    if( iRowM1 < 0 || iRowM1 >= height )
			    {
			        mexErrMsgTxt( "iRowM1 out of bound" );
			    }
				
			    float thisWeight = posteriorMap[c][iRowPost+iColPost*subsampledHeight];

			    if( thisWeight >= membershipThreshold )
			    {
			    	if( currentMixingProb[c] < currentSeed )
			    	{
						float previousMixing = currentMixingProb[c];
						currentMixingProb[c] += thisWeight;
						
						while( currentMixingProb[c] >= currentSeed && !allSeedsAreFound)
						{
							if(display)
								mexPrintf("1\n");

							/* found the bin ! */
							/* collect this example */

							weights[currentIndPerCluster[c]] = thisWeight;
							
							if( thisWeight < 1e-3 )
							{
								mexPrintf("weight too small\n");
							}
							
							dimsOutput[0] = winHeight; dimsOutput[1] = winWidth;
							mxArray* p_patch = mxCreateNumericArray( 2, dimsOutput, mxSINGLE_CLASS, mxREAL );
							mxSetCell( mxGetCell(p_patches,c), currentIndPerCluster[c], p_patch );
							float* patch = (float*)mxGetPr(p_patch);
							
							for( int ori3 = 0; ori3 < numOrient; ++ori3 ) // ori3, row3 and col3 is the index inside the window
							{
								mxArray* p_s1 = mxCreateNumericArray( 2, dimsOutput, mxSINGLE_CLASS, mxREAL );
								mxSetCell( p_S1maps, ori3*numExamplePerCluster[c] + currentIndPerCluster[c], p_s1 );
								mxArray* p_m1 = mxCreateNumericArray( 2, dimsOutput, mxSINGLE_CLASS, mxREAL );
								mxSetCell( p_M1maps, ori3*numExamplePerCluster[c] + currentIndPerCluster[c], p_m1 );
								
								float* s1 = (float*)mxGetPr(p_s1);
								float* m1 = (float*)mxGetPr(p_m1);

								for( int col3 = 0; col3 < winWidth; ++col3 )
								{
									int ind3 = iRowM1 + 0 - floor(winHeight/2.0) +
											height * ( iColM1 + col3 - floor(winWidth/2.0) );
									for( int row3 = 0; row3 < winHeight; ++row3 )
									{
										if( ind3 < 0 || ind3 >= width * height )
							            {
							                mexErrMsgTxt( "ind3 out of bound" );
							            }
										
										int jj = row3+col3*winHeight;
							            s1[jj] = SUM1map[ori3][ind3];
							            m1[jj] = MAX1map[ori3][ind3];
							            if( ori3 == 0 )
							            {
							            	patch[jj] = srcIm[ind3];
							            }
							            ++ind3;
									}
								}
							}
							
							/* go on to the next seed */
							float oldSeed = currentSeed;
							currentIndPerCluster[c]++;
							if( currentIndPerCluster[c] >= numExamplePerCluster[c] )
							{
								allSeedsAreFound = true;
								break;
							}
							
							
							currentSeed = seedsPerCluster[c][currentIndPerCluster[c]];
							if (display)
								mexPrintf("newSeed: %.3f, oldSeed: %.3f, previousMixing: %.3f, thisWeight: %.3f\n",currentSeed,oldSeed,previousMixing,thisWeight);
							if( currentSeed < oldSeed)
							{
								mexPrintf("The random seeds should be sorted from low to high!\n");
								mexPrintf("previous: %.3f, current: %.3f, ind: %d, cluster: %d\n",oldSeed,currentSeed,currentIndPerCluster[c], c);
								mexErrMsgTxt("Error !");
							}
						}
					}
					
				}
			}
		}
	}
    
}


/* read in input variables and run the algorithm */
void mexFunction(int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray *prhs[])            
{
	 mxArray *f;
	 mxClassID datatype;
	 const mwSize* dims;
	 mwSize dimsOutput[2];
     void* start_of_pr;
     mxArray* pA;
     int bytes_to_copy;
 
 	/*
	 * input variable 0: numOrient
	 */
	numOrient = (int)mxGetScalar(prhs[0]);
 
 	/*
	 * input variable 1: S1 maps
	 */
    const mxArray* pAS1Map = prhs[1];
    dims = mxGetDimensions(pAS1Map);
    
    nGaborFilter = dims[0] * dims[1];
    nGaborScale = nGaborFilter / numOrient;
    SUM1map = (const float**)mxCalloc( nGaborFilter, sizeof(*SUM1map) );   /* SUM1 maps */
    
    for ( int i=0; i<nGaborFilter; ++i )
    {
        f = mxGetCell(pAS1Map, i);
        datatype = mxGetClassID(f);
        if ( datatype != mxSINGLE_CLASS )
            mexErrMsgTxt("warning !! single precision required for MAX1map");
        SUM1map[i] = (const float*)mxGetPr(f);    /* get the pointer to cell content */
        height = mxGetM(f);    /* overwriting is ok, since it is constant  over Gabor orientations and scales */
        width = mxGetN(f);
    }

    /*
	 * input variable 2: M1 maps
	 */
    const mxArray* pAM1Map = prhs[2];
    dims = mxGetDimensions(pAM1Map);
    int nGaborFilter = dims[0] * dims[1];
    int nGaborScale = nGaborFilter / numOrient;

    MAX1map = (const float**)mxCalloc( nGaborFilter, sizeof(*MAX1map) );   /* MAX1 maps */
    for ( int i=0; i<nGaborFilter; ++i )
    {
        f = mxGetCell(pAM1Map, i);
        datatype = mxGetClassID(f);
        if ( datatype != mxSINGLE_CLASS )
            mexErrMsgTxt("warning !! single precision required for MAX1map");
        MAX1map[i] = (const float*)mxGetPr(f);    /* get the pointer to cell content */
    }
    
 	/*
     * input variable 3: posteriorMap: for multiple images (scales, rotations are marginalized out as latent variables)
     */
    numCluster = mxGetM( prhs[3] ) * mxGetN( prhs[3] );
	posteriorMap = (float**) mxCalloc( numCluster, sizeof(*posteriorMap) );
    for( int j = 0; j < numCluster; ++j )
    {
        f = mxGetCell(prhs[3], j);
        datatype = mxGetClassID( f );
        if (datatype != mxSINGLE_CLASS)
            mexErrMsgTxt("warning !! single precision required for posteriorMap.");
        posteriorMap[j] = (float*)mxGetPr( f );
    }
	
	/*
	 * input variable 4: ARGMAX2 maps
	 */
	// currently not in use
	
	/*
	 * input variable 5: searchInterior
	 */
	searchInterior = (float*) mxGetPr( prhs[5] );
	datatype = mxGetClassID( prhs[5] );
	if (datatype != mxSINGLE_CLASS)
        mexErrMsgTxt("warning !! single precision required for search interior.");
    /* parse searchInterior */
    verticalMargin = (int)searchInterior[0];
    horizontalMargin = (int)searchInterior[1];
    stepSize = (int)searchInterior[2];
	
	/*
	 * input variable 6: windowSize
	 */
    datatype = mxGetClassID( prhs[6] );
    if (datatype != mxSINGLE_CLASS)
        mexErrMsgTxt("warning !! single precision required for windowSize.");
	float* windowSize = (float*) mxGetPr( prhs[6] );
	winWidth = windowSize[1];
	winHeight = windowSize[0];
	
	if (display)
    	mexPrintf("numCluster=%d\n",numCluster);
	
	/*
	 * input variable 7: random seeds
	 */
	seedsPerCluster = (float**)mxCalloc(numCluster,sizeof(*seedsPerCluster));
	numExamplePerCluster = (int*)mxCalloc(numCluster,sizeof(*numExamplePerCluster));
	for( int c = 0; c < numCluster; ++c )
    {
    	f = mxGetCell( prhs[7], c );
    	numExamplePerCluster[c] = mxGetM(f) * mxGetN(f);
    	seedsPerCluster[c] = (float*)mxGetPr(f);
    }
	
	/*
	 * input variable 8: currentIndPerCluster[c] (also as output)
	 */
	currentIndPerCluster = (int*)mxGetPr(prhs[8]);

	/*
	 * input variable 9: currentMixingProb (also as output)
	 */
	currentMixingProb = (float*)mxGetPr(prhs[9]);
	
	/*
	 * input variable 10: data weights (also as output)
	 */
    p_weights = prhs[10];
    
    /*
     * input variable 11: collected S1 maps (also as output)
     */
    p_allS1 = prhs[11];
    
    /*
     * input variable 12: collected M1 maps (also as output)
     */
    p_allM1 = prhs[12];
    
    /*
     * input variable 13: input image (gray)
     */
    srcIm = (float*)mxGetPr( prhs[13] );

    /*
     * input variable 14: collected image patches (gray, also as output)
     */
    p_patches = prhs[14];
	
	/*
     * input variable 15: membershipThreshold
     */
	membershipThreshold = (float)mxGetScalar(prhs[15]);
    
    Compute();

}

