clear all
close all
clc

gpuDevice(1)
%%
% Load training data.
% data = load('Training_For_Horizontal_Regression_Network_Yasser.mat', 'RotatedCoordinates_plus_Angle', 'imageFileName');
data = load('Training_For_Horizontal_Regression_Network_Yasser_4228images_Update2020.mat');
% stopSigns2 = struct2table(data.TrainingDataForRegression);
% stopSigns2 = (data.TrainingDataForRegression);
stopSigns2 = data.s3;

%data = load('Training_For_Horizontal_Regression_Network_Yasser_4228images_Update2020.mat');
%stopSigns2 = (data.s3)

% rng('default');
% Used_Model='Built_in';
Used_Model='squeezenet';
% Used_Model='vgg16';    % memory Error
% Used_Model='resnet50';
% Used_Model='alexnet';
% Used_Model='googlenet';
% Used_Model='inceptionv3';
% Used_Model='vgg19';     % Error Nan-Values
% Used_Model='resnet50';
% %     'resnet101'
% %     'inceptionv3'
% %     'inceptionresnetv2'


%             ---->>>      ------>   Trained on :::   4212-images  <<-----
%%
figure,
b=[];
% Add fullpath to image files.
for kYasser=1:size(stopSigns2,1)
%     stopSigns2(kYasser).imageFileName = fullfile(pwd,(stopSigns2(kYasser).imageFileName));
%     FY_name=stopSigns2(kYasser).imageFileName;
%     [ drive, New_FY ,ext ]=fileparts(FY_name);
    %%
    % Uncomment following 5-lines only first time if Folder::'Ratio_Preserved_CLE_DataSet_V00' is not having ratio preserve images
%     RatioPreservedImage=YsrNetCopiedCode_RatioPreserve(imread(FY_name),320);
%     imshow(RatioPreservedImage);
%     pause(0.1);
%     Ratio_Right_image_Path=[fullfile(pwd,'Ratio_Preserved_CLE_DataSet_V00') '\' New_FY '.jpg' ];
%     imwrite(RatioPreservedImage,Ratio_Right_image_Path);
%     stopSigns2(kYasser).imageFileName=Ratio_Right_image_Path;
%     temp=stopSigns2(kYasser).RotatedCoordinates_plus_Angle;
%     stopSigns2(kYasser).RotatedCoordinates_plus_Angle=temp;
    kYasser
end

% s3=struct2table(stopSigns2);
% for kYasser=1:size(s3,1)
%     temp=cell2mat(s3.RotatedCoordinates_plus_Angle(kYasser));
%     s3.RotatedCoordinates_plus_Angle{kYasser}=str2num(temp);
% end
% save('Training_For_Horizontal_Regression_Network_Yasser_4228images_Update2020.mat','s3');

%% Testing Rectangles on Original images
% % % % for kYasser=1:size(s3,1)/100
% % % %     imshow(imread(s3.imageFileName{kYasser}))
% % % %     rectangle('Position',s3.RotatedCoordinates_plus_Angle{kYasser});
% % % %     pause(0.5);
% % % % end
% disp(s3);

% return



%%



%%
% Set random seed to ensure example training reproducibility.
% rng(0);
rng('default');
% s3=s3(1:200,:);
% Randomly split data into a training and test set.
shuffledIndices = randperm(height(stopSigns2));
idx = floor(0.9 * length(shuffledIndices) );
trainingData = stopSigns2(shuffledIndices(1:idx),:);
testData = stopSigns2(shuffledIndices(idx+1:end),:);

%%
% Set network training options:
%   directory. Change this to another location if required.
YasserEpochs=20;
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 1, ...
    'ExecutionEnvironment','gpu', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs',YasserEpochs, ...
    'CheckpointPath', [pwd '\temp_FastRCNN_Trainings']);

%%
% Loading Base Squeeze Network (Modified by me : Syed YASSER Arafat
load('Base_Squeeze_Network_Ysr_v1.mat');

%%
% trainingData=trainingData(1:50,:);

%%5
tic
% Train the Fast R-CNN detector. Training can take a few minutes to complete.
% frcnn = trainFastRCNNObjectDetector(trainingData, Used_Model , options, ...
%     'NegativeOverlapRange', [0 0.1], ...
%     'PositiveOverlapRange', [0.5 1], ...
%     'SmallestImageDimension', 320);
% % frcnn = trainFastRCNNObjectDetector(trainingData, lgraph_1 , options, ...
% %     'NegativeOverlapRange', [0 0.1], ...
% %     'PositiveOverlapRange', [0.5 1], ...
% %     'SmallestImageDimension', 320);
frcnn = trainFastRCNNObjectDetector(trainingData, frcnn , options, ...
    'NegativeOverlapRange', [0 0.1], ...
    'PositiveOverlapRange', [0.5 1], ...
    'SmallestImageDimension', 320);
 Y_endTime=toc;
 Y_TrainTime=Y_endTime;

YsrModel_Name=['Yasser_HorizontalX_Urdu_FRCNN_Trained_On_' num2str(size(trainingData,1)) '_Tested_On_' num2str(size(testData,1)) '-images_n_Model-Name_' Used_Model '_Ep' num2str(YasserEpochs) '_Time-' num2str(Y_TrainTime) '_.mat'];
save(YsrModel_Name,'frcnn','Y_TrainTime');
% % % % % % % 
% % % % % % % %%

%//////////////////////////////////////////////////////////////////
%//////////////////////  Training Accuracy ////////////////////////////////////////////
%/////////////////////////////////////////////////////////////////////////////////////
%/////////////////////////////////////////////////////////////////////////////////////
results=[];
numImages = size(trainingData,1);
results= struct('Boxes',[],'Scores',[]);
GroundTruth=table((trainingData.RotatedCoordinates_plus_Angle));
hold ,
for i = 1:numImages
%                 I = (imread(stopSigns2(i).imageFileName));
                I = imread(trainingData.imageFileName{i});
            %     RatioPreservedImage=YsrNetCopiedCode_RatioPreserve(YourImage,EqualDimenstion)
            %     Following function 'YsrNetCopiedCode_RatioPreserve' is only necessary for
            %     InceptionV3. Others Alexnet+Googlenet+Squeeznet automatically adjusts for
            %     the image input size.
             I=YsrNetCopiedCode_RatioPreserve(I,320);
%///////////////////////////////////////////////////////////////////////////            
% % % % %                 I=YsrNetCopiedCode_RatioPreserve(I,299);
% % % % %                 GroundTruthCoords=cell2mat(GroundTruth.Var1(i));
% % % % %                 GroundTruth.Var1{i,1}(1)=GroundTruth.Var1{i,1}(1)-10;   
%///////////////////////////////////////////////////////////////////////////

            %     imshow(I);
                [bboxes,scores] = detect(frcnn,I,'ExecutionEnvironment','gpu');
                detectedImg = insertShape(I, 'Rectangle', bboxes,'Color','red');
                

% % % %                 GroundTruthCoords(2)=GroundTruthCoords(2)-10;
% % % %                 GroundTruthCoords(1)=GroundTruthCoords(1)-11;   % changing column value
% %                 detectedImg = insertShape(detectedImg, 'Rectangle',GroundTruthCoords ,'Color','green');
%                 imshow(detectedImg)
%                 drawnow 
%                 pause(0.01);
                results(i).Boxes = bboxes;
                results(i).Scores = scores;
                disp(['Tr-' num2str(i)]);
end
results = struct2table(results);

% GroundTruth=table((s3.RotatedCoordinates_plus_Angle));
% [ap,recall,precision] = evaluateDetectionPrecision(results(1:20,:),GroundTruth(1:20,:));
[ap_Train,recall,precision] = evaluateDetectionPrecision(results,GroundTruth);
% figure
plot(recall,precision)
grid on
title(sprintf('Train-Set Average Precision = %.4f',ap_Train));
TrResults={ap_Train,recall,precision};
TrGroundTruth=GroundTruth;
%/////////////////////////////////////////////////////////////////////////////////////
%/////////////////////////////////////////////////////////////////////////////////////


%//////////////////////////////////////////////////////////////////
%//////////////////////////// Testing Accuracy//////////////////////////////////////
%/////////////////////////////////////////////////////////////////////////////////////
%/////////////////////////////////////////////////////////////////////////////////////
results=[];
numImages = size(testData,1);
results= struct('Boxes',[],'Scores',[]);
GroundTruth=table((testData.RotatedCoordinates_plus_Angle));
hold ,
for i = 1:numImages
%                 I = (imread(stopSigns2(i).imageFileName));
                I = imread(testData.imageFileName{i});
            %     RatioPreservedImage=YsrNetCopiedCode_RatioPreserve(YourImage,EqualDimenstion)
            %     Following function 'YsrNetCopiedCode_RatioPreserve' is only necessary for
            %     InceptionV3. Others Alexnet+Googlenet+Squeeznet automatically adjusts for
            %     the image input size.
             I=YsrNetCopiedCode_RatioPreserve(I,320);
%///////////////////////////////////////////////////////////////////////////            
% % % % %                 I=YsrNetCopiedCode_RatioPreserve(I,299);
% % % % %                 GroundTruthCoords=cell2mat(GroundTruth.Var1(i));
% % % % %                 GroundTruth.Var1{i,1}(1)=GroundTruth.Var1{i,1}(1)-10;   
%///////////////////////////////////////////////////////////////////////////

            %     imshow(I);
                [bboxes,scores] = detect(frcnn,I,'ExecutionEnvironment','gpu');
                detectedImg = insertShape(I, 'Rectangle', bboxes,'Color','red');
                

% % % %                 GroundTruthCoords(2)=GroundTruthCoords(2)-10;
% % % %                 GroundTruthCoords(1)=GroundTruthCoords(1)-11;   % changing column value
% %                 detectedImg = insertShape(detectedImg, 'Rectangle',GroundTruthCoords ,'Color','green');
                imshow(detectedImg)
                drawnow 
                pause(0.01);
                results(i).Boxes = bboxes;
                results(i).Scores = scores;
                disp(['Ts-' num2str(i)]);
end
results = struct2table(results);

% GroundTruth=table((s3.RotatedCoordinates_plus_Angle));
% [ap,recall,precision] = evaluateDetectionPrecision(results(1:20,:),GroundTruth(1:20,:));
[ap_Test,recall,precision] = evaluateDetectionPrecision(results,GroundTruth);
figure
plot(recall,precision)
grid on
title(sprintf('Test-Set Average Precision = %.4f',ap_Test))
%/////////////////////////////////////////////////////////////////////////////////////
%/////////////////////////////////////////////////////////////////////////////////////
TsResults={ap_Test,recall,precision};
TsGroundTruth=GroundTruth;

YsrModel_Name=['Yasser_HorizontalX_Urdu_FRCNN_Trained_On_' num2str(size(trainingData,1)) '_Tested_On_' num2str(size(testData,1)) '-images_n_Model-Name_' Used_Model '_Ep' num2str(YasserEpochs) '_Tr_ap-' num2str(ap_Train) '_Ts_ap-' num2str(ap_Test) '_Time-' num2str(Y_TrainTime) '_.mat'];
save(YsrModel_Name,'frcnn','TrResults','YasserEpochs','TsResults','TrGroundTruth','TsGroundTruth','ap_Train','ap_Test','Y_TrainTime');
% % % % % % % % % 
% % % % % % % % % 
% % % % % % % % % 
% % % % % % % % % %
% % load(YsrModel_Name);
% % % % % % % % % %
% % % % % % % % % % % Test the Fast R-CNN detector on a test image.
% % img = imread('Ytest_Urdu_localization.jpg');
% % % img = imcomplement(imread('17.jpg'));
% % % img = (imread('17.jpg'));
% % img = (imread('18.jpg'));
% % % 
% % % % Run the detector.
% % [bbox, score, label] = detect(frcnn, img,'ExecutionEnvironment','cpu');
% % 
% % %
% % % % Display detection results.
% % detectedImg = insertShape(img, 'Rectangle', bbox,'Color','green');
% % figure
% % imshow(detectedImg)


%%

% % function RatioPreservedImage=YsrNetCopiedCode_RatioPreserve(YourImage,EqualDimenstion)
% %             %figure out the pad value to pad to white
% %             if isinteger(YourImage)
% %                pad = intmax(class(YourImage));
% %             else
% %                pad = 1;   %white for floating point is 1.0
% %             end
% %             %figure out which dimension is longer and rescale that to be the 256
% %             %and pad the shorter one to 256
% % %             EqualDimenstion=256
% %             [r, c, ~] = size(YourImage)
% %             if r > c
% %               newImage = imresize(YourImage, EqualDimenstion / r);
% %               NewImage(:, end+1 : EqualDimenstion, :) = pad;
% %             elseif c > r
% %               newImage = imresize(YourImage, EqualDimenstion / c);
% %               NewImage(end+1 : EqualDimenstion, :, :) = pad;
% %             else
% %               newImage = imresize(YourImage, [EqualDimenstion, EqualDimenstion]);
% %             end
% %             RatioPreservedImage=newImage;
% % end
