clear all
close all
clc
% return;
Yassers_Model='Yasser_HoriX_v2_Urdu_FasterRCNN_Trained_On_';
%%
% Load training data.
% data = load('Training_For_Horizontal_Regression_Network_Yasser.mat', 'RotatedCoordinates_plus_Angle', 'imageFileName');
data = load('Training_For_Horizontal_Regression_Network_Yasser_4228images.mat');
% stopSigns2 = struct2table(data.TrainingDataForRegression);
stopSigns2 = (data.TrainingDataForRegression);
% data2 = load('rcnnStopSigns.mat','stopSigns','fastRCNNLayers');
% fastRCNNLayers = data2.fastRCNNLayers;

% rng('default');
% Used_Model='Built_in';
% Used_Model='squeezenet';
% Used_Model='vgg16';    % memory Error
Used_Model='resnet50';
% Used_Model='alexnet';
% Used_Model='googlenet';
% Used_Model='inceptionv3';
% Used_Model='vgg19';     % Error Nan-Values
% Used_Model='resnet18';
% Used_Model='inceptionresnetv2';  

% %     'alexnet'
% %     'vgg16'
% %     'vgg19'
% %     'resnet50'
% %     'resnet101'
% %     'inceptionv3'
% %     'googlenet'
% %     'inceptionresnetv2'
% %     'squeezenet'

%             ---->>>      ------>   Trained on :::   4212-images  <<-----
%%
b=[];
% Add fullpath to image files.
for kYasser=1:size(stopSigns2,2)
    stopSigns2(kYasser).imageFileName = fullfile(pwd,(stopSigns2(kYasser).imageFileName));
    temp=stopSigns2(kYasser).RotatedCoordinates_plus_Angle;

    stopSigns2(kYasser).RotatedCoordinates_plus_Angle=temp;

end

s3=struct2table(stopSigns2);
for kYasser=1:size(s3,1)
    temp=cell2mat(s3.RotatedCoordinates_plus_Angle(kYasser));
    s3.RotatedCoordinates_plus_Angle{kYasser}=str2num(temp);
end


%% Testing Rectangles on Original images

% % % % for kYasser=1:size(s3,1)/100
% % % %     imshow(imread(s3.imageFileName{kYasser}))
% % % %     rectangle('Position',s3.RotatedCoordinates_plus_Angle{kYasser});
% % % %     pause(0.5);
% % % % end
% disp(s3);
% return

%%
% % % % % % %%
% % % % % % imageAugmenter = imageDataAugmenter( ...
% % % % % %     'RandRotation',[-20,20], ...
% % % % % %     'RandScale',[0.5 1.5])
% % % % % % 
% % % % % % imageSize = [320 240 3];
% % % % % % augimds = augmentedImageDatastore(imageSize,s3,'DataAugmentation',imageAugmenter);
% % % % % % 
% % % % % % minibatch = preview(augimds);
% % % % % % imshow(imtile(minibatch.input));

% s3=s3(1:200,:);
%%
% Set random seed to ensure example training reproducibility.
% rng(0);
rng('default');

% Randomly split data into a training and test set.
shuffledIndices = randperm(height(s3));
idx = floor(0.9 * length(shuffledIndices) );
trainingData = s3(shuffledIndices(1:idx),:);
testData = s3(shuffledIndices(idx+1:end),:);

%%
% disp('Estimating Anchorboxes values from data..... (wait a little)');
% trainingData_estimate = boxLabelDatastore(trainingData(:,2:end));
% numAnchors = 5;
% [anchorBoxes,meanIoU] = estimateAnchorBoxes(trainingData_estimate,numAnchors);
% anchorBoxes
% % % % % % % % maxNumAnchors = 8;
% % % % % % % % meanIoU = zeros([maxNumAnchors,1]);
% % % % % % % % anchorBoxes = cell(maxNumAnchors, 1);
% % % % % % % % for k = 1:maxNumAnchors
% % % % % % % %     % Estimate anchors and mean IoU.
% % % % % % % %     [anchorBoxes{k},meanIoU(k)] = estimateAnchorBoxes(trainingData_estimate,k);    
% % % % % % % % end
% % % % % % % % figure
% % % % % % % % plot(1:maxNumAnchors,meanIoU,'-o');
% % % % % % % % ylabel("Mean IoU");
% % % % % % % % xlabel("Number of Anchors");
% % % % % % % % title("Number of Anchors vs. Mean IoU");

%%
% Set network training options:
%
% * Set the CheckpointPath to save detector checkpoints to a temporary
%   directory. Change this to another location if required.

% %   SelRange=20;
% %   trainingData=trainingData(1:SelRange,:);
% %   testData=testData(1:SelRange,:);
  
 
YasserEpochs=10;
for i=1:6    %//  4x5=20 epochs
   
        options = trainingOptions('sgdm', ...
            'MiniBatchSize', 15, ...
            'ExecutionEnvironment','gpu', ...
            'InitialLearnRate', 1e-3, ...
            'MaxEpochs',YasserEpochs, ...
            'CheckpointPath', tempdir);
        %  return;
        %%
        if i==1
            Used_Model_U=Used_Model; 
        else
            load(YsrModel_Name);
            Used_Model_U=frcnn;
        end
        
             
        tic;
        % Train the Fast R-CNN detector. Training can take a few minutes to complete.
        [frcnn,info]= trainFasterRCNNObjectDetector(trainingData, Used_Model_U , options, ...
            'NegativeOverlapRange', [0 0.1], ...
            'PositiveOverlapRange', [0.5 1], ...
            'SmallestImageDimension', 300);
        Y_endTime=toc;
        Y_TrainTime=Y_endTime;

        old_Epochs=YasserEpochs*i;
        YsrModel_Name=[Yassers_Model num2str(size(trainingData,1)) '_Tested_On_' num2str(size(testData,1)) '-images_n_Model-Name_' Used_Model '_Ep' num2str(old_Epochs) '_TrainTime_' num2str(Y_TrainTime) 'seconds_.mat'];
        save(YsrModel_Name,'frcnn','info');

                % % % % % % % 
                % % % % % % % %%

                %% Retraining of Detector
                % load('fast_rcnn_checkpoint__14520__2019_04_26__02_02_34.mat')
                % % % % % %  load('faster_rcnn_stage_3_checkpoint__7582__2019_05_30__11_59_38.mat')
                % % % % % tic;
                % % % % % frcnn = trainFasterRCNNObjectDetector(trainingData, detector , options, ...
                            % % % % % % YasserEpochs=25;
                            % % % % tic
                            % % % % 
                            % % % % 
                            % % % % %//////////////////////////////////////////////////////////////////
                            % % % % %//////////////////////  Training Accuracy ////////////////////////////////////////////
                            % % % % %////////////frcnn = trainFasterRCNNObjectDetector(trainingData, frcnn , options, ...
                            % % % %     'NegativeOverlapRange', [0 0.1], ...
                            % % % %     'PositiveOverlapRange', [0.5 1], ...
                            % % % %     'SmallestImageDimension', 300);
                            % % % % Y_endTime=toc;
                            % % % % Y_TrainTime=Y_endTime;/////////////////////////////////////////////////////////////////////////
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

                %///////////////////////////////////////////////////////////////////////////            
                % % % % %                 I=YsrNetCopiedCode_RatioPreserve(I,299);
                % % % % %                 GroundTruthCoords=cell2mat(GroundTruth.Var1(i));
                % % % % %                 GroundTruth.Var1{i,1}(1)=GroundTruth.Var1{i,1}(1)-10;   
                %///////////////////////////////////////////////////////////////////////////

                            %     imshow(I);
                                [bboxes,scores] = detect(frcnn,I,'ExecutionEnvironment','gpu');
%                                 detectedImg = insertShape(I, 'Rectangle', bboxes,'Color','red');


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
                figure
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
                % % % %                 detectedImg = insertShape(detectedImg, 'Rectangle',GroundTruthCoords ,'Color','green');
                                imshow(detectedImg)
                                drawnow 
                                pause(0.1);
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
                xlabel('Recall');
                ylabel('Precision');
                %/////////////////////////////////////////////////////////////////////////////////////
                %/////////////////////////////////////////////////////////////////////////////////////
                TsResults={ap_Test,recall,precision};
                TsGroundTruth=GroundTruth;
                YsrModel_Name=[Yassers_Model num2str(size(trainingData,1)) '_Tested_On_' num2str(size(testData,1)) '-images_n_Model-Name_' Used_Model '_Ep' num2str(old_Epochs) '_Tr_ap-' num2str(ap_Train) '_Ts_ap-' num2str(ap_Test) '_' num2str(Y_TrainTime) '_.mat'];
                save(YsrModel_Name,'frcnn','info','TrResults','YasserEpochs','TsResults','TrGroundTruth','TsGroundTruth','ap_Train','ap_Test','Y_TrainTime','YasserEpochs');
end
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
