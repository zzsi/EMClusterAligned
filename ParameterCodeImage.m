
disp(['===============> Getting parameters, mex-c code, and training images']);
mkdir('working'); % working directory to store .mat files. 
mkdir('output'); % directory that contains output from the code. 

%% parameters for the outer square bounding box
category = 'Social';
outerBBsize = 200;
rectStep = 40;
%% parameters for EM clustering
locationPerturbationFraction = .25; % the size of neighborhood for MAX2 pooling, as well as surround supression
rotationRange = -0:2:0; % allowed global rotation of the template
subsampleS2 = 1; % subsampling step size for computing SUM2 maps
maxNumClusterMember = 30; % maximum number of training examples in each cluster used in re-learning the template
S2Thres = 0; % cut-off value for detected instances of the template
numEMIter = 3; % number of EM iterations
%% parameters for active basis
numCluster = 10; % number of data clusters
numElement = 30; % maximum number of Gabors in active basis
S1Thres = .3;        % cut-off for selected Gabors
epsilon = .1; % allowed correlation between selected Gabors 
subsample = 1; subsampleM1 = 1; % subsample in computing MAX1 maps
locationShiftLimit = 3; % shift in normal direction = locationShiftLimit*subsample pixels
orientShiftLimit = 1; % shift in orientation
%% parameters for Gabor filters
numScale = 1;
scales = 0.7; % scales of Gabor wavelets 
numOrient = 16;  % number of orientations
saturation = 6.; % saturation level for sigmoid transformation
doubleOrNot = -1;
%% parameters for exponential model
binSize = .2;  % binsize for computing histogram of q()
numStoredPoint = 50; % number of stored lambda values
spacing = .1; % spacing between two adjacent points
%% parameters for normalization
localOrNot = 1; % if we use local normalization or not. If not, set it to -1 
localHalfx1 = 15; localHalfy1 = 15; % the half range for local normalization, has to be quite large
windowNormalizeOrNot = -1; % whether normalize within the scanning window in detection 
if (localOrNot>0)
    windowNormalizeOrNot = -1;
end % if we use local normalization, we should not use window normalization in detection
thresholdFactor = .01;  % divide the response by max(average, maxAverage*thresholdFactor)

%% parameters for detection
inhibitFind = -1;  % whether to apply inhibition after detection for re-computing MAX2 score
resolutionGap = .5; % gap between consecutive resolutions in detection
numExtend = 0; % number of gaps extended both below and above zero
numResolution = numExtend*2 + 1;  % number of resolutions to search for in detection stage
originalResolution = numExtend + 1; % original resolution is the one at which the imresize factor = 1
allResolution = (-numExtend : numExtend)*resolutionGap + 1.;
%% read in positive images
imageFolder = 'positiveImage'; % folder of training images  
imageName = dir([imageFolder '/*.jpg']);
numImage = size(imageName, 1); % number of training images 
imageClass = zeros(1, numImage);
Ioriginal = cell(1, numImage);
for img = 1 : numImage
    tmpIm = imread([imageFolder '/' imageName(img).name]); 
    if size(tmpIm,3) == 3
        tmpIm = rgb2gray(tmpIm);
    end
    
    imageClass(img) = str2num(imageName(img).name(2:3));

    sx = size(tmpIm,1); sy = size(tmpIm,2);
    tmpIm = imresize( tmpIm, outerBBsize/sqrt(sx*sy), 'bilinear' );
    Ioriginal{img} = single(tmpIm);
    J0 = Ioriginal{img};
    J = cell(1, numResolution);
    for r=1:numResolution
       J{r} = imresize(J0, allResolution(r), 'nearest');  % scaled images
    end
    multipleResolutionImageName = ['working/multipleResolutionImage' num2str(img)]; 
    save(multipleResolutionImageName, 'J');
end

count = 0;
% generate the set of geometric transformations for each template
for templateScaleInd = 0:0 % large scale change
    for rotation = rotationRange 
        for rowScale = 2.^[0] % small scale change and aspect ratio change
            for colScale = 2.^[0]
                count = count + 1;
                templateTransform{count} = [templateScaleInd rowScale colScale rotation];
                if rotation == 0
                	NO_TRANSFORM = count - 1; % starts from 0
                end
            end
        end
    end
end
nTransform = count;

save('partLocConfig.mat','outerBBsize',...
    'category','numOrient','localOrNot','subsample','saturation','locationShiftLimit','orientShiftLimit',...
    'numElement','thresholdFactor','doubleOrNot', 'numCluster', 'numEMIter', 'subsampleS2', 'locationPerturbationFraction',...
    'rotationRange','nTransform','templateTransform');
clear Ioriginal

