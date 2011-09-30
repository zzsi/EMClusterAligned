/* 
 * Crops a patch on image or feature map. The patch can be 4-dimensional (width, height, orientation, scale). 
 *
 * Usage:
 *    dest = mexc_CropInstanceNew(src,outRow,outCol,outOri,outScale,destWidth,destHeight,nOri,nScale);
 *
 *      When cropping a colored image, set nOri = 3, nScale = 1.
 *
 *
 */
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

float **src, **dest;        /* source and destination images or feature maps */
int nPixel;               /* number of pixels in destination image patch */
int destWidth, destHeight;
int srcWidth, srcHeight;
float rshift, cshift, rotation, scaling, reflection;  /* rigid geometric transformation of the template */
/* The following two parameters are of the destination map. Source feature map may have a different number of scales.  */
int nOri;                   /* number of quantized levels of orientation within 0~PI */
int nScale;                 /* number of feature scales */

void crop(int indDest,int indSrc)
{
    int i, r, c;
	r = rshift - (int)floor(destHeight/2.0); c = cshift - (int)floor(destWidth/2.0);
	int rMIN = r, cMIN = c;
	int rMAX = r + destHeight - 1;
	int cMAX = c + destWidth - 1;
    for( i = 0; i < nPixel; ++i ) /* location */
    {
        if( r < 0 || r >= srcHeight || c < 0 || c >= srcWidth )
        {
            /* do nothing */
        }
		else
		{
			if( reflection > 0 ) /* reflection = 1: no reflection, -1: horizontal flip */
			{
				dest[indDest][i] = src[indSrc][c*srcHeight+r];
			}
			else
			{
				int c2; /* reflected */
				c2 = cMAX + cMIN - c;
                if( c2 >= 0 && c2 < srcWidth )
                {
                    dest[indDest][i] = src[indSrc][c2*srcHeight+r]; /* only horizontal flip */
                }
			}
		}
		/* increment for (r,c) */
		++r;
		if( r > rMAX )
		{
			r = rMIN;
			++c;
		}
    }
}

void compute()
{
    int i, s, o; /* destination image: scale, orientation */
    int srcS, srcO; /* src image: scale, orientation */
    int indSrc, indDest;
	
    for( s = 0; s < nScale; ++s )
    {
         for( o = 0; o < nOri; ++o )
         {
             indDest = s*nOri + o;
             for( i = 0; i < nPixel; ++i ) /* location */
             {
                 dest[indDest][i] = 0;
             }
         }
    }

    
    /* The following two FOR loops go over destimation images/feature maps
     * at different scales and orientations.
     */	
    for( s = 0; s < nScale; ++s )
    {
        srcS = s + scaling;
		if( srcS < 0 || srcS >= nScale )
        {
            continue;
        }

        for( o = 0; o < nOri; ++o )
        {
            srcO = o + rotation;
			while( srcO < 0 )
            {
                srcO += nOri;
            }
            while( srcO >= nOri )
            {
                srcO -= nOri;
            }
			if( reflection < 0 )
			{
				srcO = nOri - srcO;
			}
			while( srcO < 0 )
            {
                srcO += nOri;
            }
            while( srcO >= nOri )
            {
                srcO -= nOri;
            }
            indDest = s*nOri + o;
            indSrc = srcS*nOri + srcO;
            crop(indDest,indSrc);
        }
    }
}


/* entry point: load input variables and run the algorithm */
void mexFunction(int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray *prhs[])   
{
    const mxArray *pA;
	mxArray *pA2;
    mwSize dimsOutput[2];
    void* start_of_pr;
    mxClassID datatype;
    int nMap, i, bytes_to_copy;
	
    /* =============================================
     * Handle input variables.
     * ============================================= 
     */
    /*
	 * input variable 0: src. A list of images or 2D feature maps.
	 */
    pA = prhs[0];
    nMap = mxGetM(pA) * mxGetN(pA);
    src = (float**)mxCalloc(nMap, sizeof(*src));
	for( i = 0; i < nMap; ++i )
    {
        srcHeight = mxGetM( mxGetCell(pA,i) );
        srcWidth = mxGetN( mxGetCell(pA,i) );
        src[i] = (float*)mxGetPr( mxGetCell(pA,i) );
        datatype = mxGetClassID( mxGetCell(pA,i) );
        if( datatype != mxSINGLE_CLASS )
        {
            mexErrMsgTxt("warning !! single precision required. src");
        }
    }
	
    /*
	 * input variable 1: row shift
	 */
	rshift = (float)mxGetScalar(prhs[1]);
	/*
	 * input variable 2: col shift
	 */
	cshift = (float)mxGetScalar(prhs[2]);
	/*
	 * input variable 3: rotation
	 */
	rotation = (float)mxGetScalar(prhs[3]);
	/*
	 * input variable 4: scaling
	 */
	scaling = (float)mxGetScalar(prhs[4]);
	/*
	 * input variable 5: (horizontal) reflection
	 */
	reflection = (float)mxGetScalar(prhs[5]);
	
    /*
	 * input variable 6: destWidth
	 */
	destWidth = (int)mxGetScalar(prhs[6]);
    /*
	 * input variable 7: destHeight
	 */
	destHeight = (int)mxGetScalar(prhs[7]);
	nPixel = destWidth * destHeight;
	
    /*
	 * input variable 8: nOri (number of orientation levels within 0 ~ pi) 
     *                   This is a parameter of the destimation map.
	 */
	nOri = (int)mxGetScalar(prhs[8]);
    
    /*
	 * input variable 9: nScale
     *                  This is a parameter of the destimation map.
	 */
	nScale = (float)mxGetScalar(prhs[9]);
    
    /* =============================================
     * Computation.
     * ============================================= 
     */
    dest = (float**)mxCalloc(nOri*nScale,sizeof(*dest));
    for( i = 0; i < nOri*nScale; ++i )
    {
        dest[i] = (float*)mxCalloc( nPixel, sizeof(**dest) );
    }
    
    compute();
    
    /* =============================================
     * Handle output variables.
     * ============================================= 
     */
    
    /*
     * output variable 0: dest
     */
    dimsOutput[0] = nOri; dimsOutput[1] = nScale;
	plhs[0] = mxCreateCellArray( 2, dimsOutput );
    dimsOutput[0] = destHeight; dimsOutput[1] = destWidth;
    for( i = 0; i < nOri*nScale; ++i )
    {
        pA2 = mxCreateNumericArray( 2, dimsOutput, mxSINGLE_CLASS, mxREAL );
        /* populate the real part of the created array */
        start_of_pr = (float*)mxGetData(pA2);
        bytes_to_copy = dimsOutput[0] * dimsOutput[1] * mxGetElementSize(pA2);
        memcpy( start_of_pr, dest[i], bytes_to_copy );
        mxSetCell( plhs[0], i, pA2 );
    }
}



