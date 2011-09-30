# include <stdio.h>
# include <stdlib.h>
# include "mex.h"        
# include "math.h"
# define ROUND(x) (floor((x)+.5))
/* Compute pixel index in the vector that stores image */
int px(int x, int y, int lengthx, int lengthy)  /* the image is lengthx*lengthy */
{            
   return (x + (y-1)*lengthx - 1); 
 }
 /* variables */
float **SUM1map;    
int numImage, numOrient; 
float saturation; /* saturation level for sigmoid transformation */
float *allSizex, *allSizey;  /* sizes of images at multiple resolutions */        
int sizex, sizey; /* MAX1 maps are smaller than SUM1 maps by subsample */ 
/* compute sigmoid transformation */
void SigmoidTransform(int img)
{
   int x, y, here, orient, i; 
   for (x=1; x<=sizex; x++)
       for (y=1; y<=sizey; y++)
       {
        here = px(x, y, sizex, sizey); 
        for (orient=0; orient<numOrient; orient++)
           {    
                i = orient*numImage+img; 
                SUM1map[i][here] = saturation*(2./(1.+exp(-2.*SUM1map[i][here]/saturation))-1.); 
            }
       }
}
/* load in variables */
void mexFunction(int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray *prhs[])                
{
 int img, orient, c; 
 mxArray *f;   
 
 c = 0; 
 numImage = ROUND(mxGetScalar(prhs[c++]));
 allSizex = (float*)mxGetPr(prhs[c++]);
 allSizey = (float*)mxGetPr(prhs[c++]);
 numOrient = ROUND(mxGetScalar(prhs[c++])); 
 saturation = mxGetScalar(prhs[c++]);
 SUM1map = (float**)mxCalloc(numImage*numOrient, sizeof(float*));   
 for (img=0; img<numImage; img++)
     for (orient=0; orient<numOrient; orient++)
      {  
       f = mxGetCell(prhs[c], orient*numImage+img); 
       SUM1map[orient*numImage+img] = (float*)mxGetPr(f);       
      }
 c++;
 for (img=0; img<numImage; img++)
 {
   sizex = ROUND(allSizex[img]);
   sizey = ROUND(allSizey[img]);
   SigmoidTransform(img); 
 }
}


                    