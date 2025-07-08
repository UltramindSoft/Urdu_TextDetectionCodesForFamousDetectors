clc
clear all
close all
% return
gpuDevice(1);

DetectorModelSelector_SSD=1;         %   <---------------   SSD==1  Else  FasterRCNN==0
ReduceSize4th_Selector=0;            %   <---------------   1== 320x320   or 0==620x620     
YasserEpochs=30;                %  <---- Epochs
NoOfTimesExperiments=1;         %  <---- For Multiple experiments
ReTrainOption=1;                %  <--- 1==True
Y_TrainTime=-1;
Volume_No='volume2b_';
load('Volume_#2b_Final_Info__Urdu-Text-images-573_Mixed-Text-336_Signboards-400_V2_.mat')


if DetectorModelSelector_SSD==1
            Yassers_Model=[Volume_No 'Yasser_Nat_DataSet_v2_Urdu_SSD_Trained_On_'];
            if ReduceSize4th_Selector==1
                 Yassers_Model=[Volume_No 'Yasser_Nat_DataSet_v2_Urdu_SSD_320x320_Trained_On_'];
            else
                 Yassers_Model=[Volume_No 'Yasser_Nat_DataSet_v2_Urdu_SSD_620x620_Trained_On_'];
            end
else
            Yassers_Model=[Volume_No 'Yasser_Nat_DataSet_v2_Urdu_FasterRCNN_Trained_On_'];
            if ReduceSize4th_Selector==1
                 Yassers_Model=[Volume_No 'Yasser_Nat_DataSet_v2_Urdu_FasterRCNN_320x320_Trained_On_'];
            else
                 Yassers_Model=[Volume_No 'Yasser_Nat_DataSet_v2_Urdu_FasterRCNN_620x620_Trained_On_'];
            end
end
%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        
if ReduceSize4th_Selector==1
                imgRows=320;
                imgCols=320;
                imgDimension=3;
else
                imgRows=620;
                imgCols=620;
                imgDimension=3;
end
% Used_Model='Built_in';
% Used_Model='squeezenet';
% Used_Model='vgg16';    % memory Error
% Used_Model='resnet18';
Used_Model='resnet50';
% Used_Model='alexnet';
% Used_Model='googlenet';
% Used_Model='inceptionv3';
% Used_Model='vgg19';     % Error Nan-Values

% Used_Model='resnet18';
% Used_Model='inceptionresnetv2';  
% load('Volume_#2_Final_Merged_text_n_mixText_Info__Urdu-Text-images-239_Mixed-Text-286_Signboards-302_.mat');
% load('Volume_#2b_Final_Info__Urdu-Text-images-573_Mixed-Text-336_Signboards-400_V2_.mat')

stopSigns2=struct2table(Yasser_OutDoor_Urdu_Text_Detection_structure);
% stopSigns2b = stopSigns2(1:250,[1,5]);    % trains on First & Last column
% stopSigns2b = stopSigns2(1:250,[1:4]);    % trains on First,to 4th column
stopSigns2b = stopSigns2(1:504,[1,5]);    % trains on First & Last column
% return


%             ---->>>      ------>   Trained on :::   4212-images  <<-----
%%
b=[];
New_Images_Path='E:\PhD2_Yasser_v2\Detection_Results_For_Papers\Yasser_SSD_Detection_Natural_DataSet\Yasser_Urdu_DataSet_Part1_vol2';
% New_Images_Path='E:\Yasser\Yasser_Chandio2000_Word_Recognition\Yasser_SSD_Detection_Natural_DataSet_v1\Volume_2b_images';
% Add fullpath to image files.
temp=[];
for kYasser=1:size(stopSigns2b,1)
    [filepath,name,ext] = fileparts((stopSigns2b.Original_Outodoor_Image_Path{kYasser}));
    
%%%%%%%Activate following line if want to copy source images to current Diectory    
    stopSigns2b.Original_Outodoor_Image_Path{kYasser} = [fullfile(New_Images_Path,name) '.jpg'];

    temp=stopSigns2b.Merged_Text_n_MixText_Rects_CoOrdinates{kYasser};
%     temp
         if ReduceSize4th_Selector==1
             bboxB = bboxresize(uint16(temp),1/4)
         else
             bboxB = bboxresize(uint16(temp),1/2);
         end
    stopSigns2b.Merged_Text_n_MixText_Rects_CoOrdinates{kYasser}=single(bboxB);
end
s3=stopSigns2b;
% s3=struct2table(stopSigns2b);
% for kYasser=1:size(s3,1)
%     temp=cell2mat(s3.RotatedCoordinates_plus_Angle(kYasser));
%     s3.RotatedCoordinates_plus_Angle{kYasser}=str2num(temp);
% end

%%//////////////////////////////////////////////////////////////////////////
% Set random seed to ensure example training reproducibility.
% rng(0);
rng('default');
% rng(7);
% Randomly split data into a training and test set.
shuffledIndices = randperm(height(s3));
idx = floor(0.95 * length(shuffledIndices) );
trainingData = s3(shuffledIndices(1:idx),:);
testData = s3(shuffledIndices(idx+1:end),:);

trainingData2=trainingData;
% % imgRows=233;
% % imgCols=310;
ColExceeded=0
RowExceeded=0
BoundaryFlag=0

% return;
for LoopSize=1:size(trainingData2,1)
            imP=trainingData2.Original_Outodoor_Image_Path{LoopSize};
            imG=imread(imP);
            imshow(imG);
            hold on
            CoOrs=[];
            CoOrs=trainingData2.Merged_Text_n_MixText_Rects_CoOrdinates{LoopSize};
            if ~isempty(CoOrs)
% % % % % %                     if CoOrs(1)+CoOrs(4) > imgCols 
% % % % % %                         disp(' Col Range Exceeded ...');
% % % % % %                         maxRows=CoOrs(1)+CoOrs(4)
% % % % % %                         plot(maxRows,CoOrs(2),'r*');
% % % % % %                         disp(' Col-Ended-Range Exceeded ...');
% % % % % %                         ColExceeded=ColExceeded+1;
% % % % % %                         BoundaryFlag=1;
% % % % % %                         trainingData2.Merged_Text_n_MixText_Rects_CoOrdinates{LoopSize}=[];
% % % % % %                     end
% % % % % %                        if  CoOrs(2)+CoOrs(3) > imgRows
% % % % % %                         disp(' Row Range Exceeded ...');
% % % % % %                         maxCols=CoOrs(2)+CoOrs(3)
% % % % % %                         plot(maxRows,CoOrs(2),'b*');
% % % % % %                         disp(' Row-Ended-Range Exceeded ...');
% % % % % %                         RowExceeded=RowExceeded+1;
% % % % % %                         BoundaryFlag=1;
% % % % % %                         trainingData2.Merged_Text_n_MixText_Rects_CoOrdinates{LoopSize}=[];
% % % % % %                        end
% % % % % %                     if CoOrs(1) <1
% % % % % %                         disp(['Less than 1 row found at-->  ' num2str(LoopSize)]);
% % % % % %                         trainingData2.Merged_Text_n_MixText_Rects_CoOrdinates{LoopSize}=[];
% % % % % %                     end
% % % % % %                     if CoOrs(2) <1
% % % % % %                         disp(['Less than 1 col found at-->  ' num2str(LoopSize)]);
% % % % % %                         trainingData2.Merged_Text_n_MixText_Rects_CoOrdinates{LoopSize}=[];
% % % % % %                     end
% % % % % % % % % %                     rectangle('Position',CoOrs);
% % % % % %                       imG = insertObjectAnnotation(imG,'Rectangle',CoOrs,'- -');
% % % % % %                       imshow(imG);
% % % % % %                       pause(0.3);
% % % % % %                       drawnow
% % % % % %                     if BoundaryFlag==1
% % % % % %                         pause(0.3)
% % % % % %                        BoundaryFlag=0;
% % % % % %                     end
            end
            LoopSize
            hold off
end
ColExceeded
RowExceeded

%Remove empty rows, acoording to 2nd empty column
idx=all(cellfun(@isempty,trainingData2{:,2}),2);
trainingData2(idx,:)=[];
%///////////////////////////////////////////////////////////////////////////
startI=1;
imageRange=size(trainingData2,1);
% imageRange=157;
% imageRange=200;
% imageRange=3766;
trainingData3=[];
trainingData3=trainingData2(startI:imageRange,:);

%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
if ReduceSize4th_Selector==1
            imds = imageDatastore(trainingData3.Original_Outodoor_Image_Path,'ReadFcn',@Yasser_customreader_4thSize);
else
            imds = imageDatastore(trainingData3.Original_Outodoor_Image_Path,'ReadFcn',@Yasser_customreader_2ndSize);
end
blds = boxLabelDatastore(trainingData3(startI:imageRange,2:end));
ds = combine(imds, blds);
%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
% dat11 = preview(ds)

figure,
for i = 2:17
    img = readimage(imds,i);
    figure,imshow(img);
    hold on;
    detectedImg = insertShape(img, 'Rectangle', ds.UnderlyingDatastores{1, 2}.LabelData{i, 1},'Color','green','LineWidth',5);
    imshow(detectedImg)
    pause(1.0);
    if mod(i,8)==0
        close all
    end
end
hold off;
pause(3);
close all;
%///////////////////////////////////////////////////////////////////////////
% return

% load('620_model_ssd_.mat')
% load('620_model_v2.mat');
% load('Synthetic_Natural_Model_v2.mat')
% lgraph=Synthetic_Natural_Model_v2;


%///////////////////////////////////////////////////////////////////////////


temp_YasserEpochs=-11;
for i=1:NoOfTimesExperiments  %//  4x5=20 epochs
   
%         options = trainingOptions('sgdm', ...
%             'MiniBatchSize', 2, ...
%             'ExecutionEnvironment','cpu', ...
%             'InitialLearnRate', 1e-3, ...
%             'MaxEpochs',YasserEpochs, ...
%             'CheckpointPath', tempdir);
        %  return;
   


        %%
        if i==1
            Used_Model_U=Used_Model; 
        else
            load(YsrModel_Name);
            Used_Model_U=frcnn;
        end
        %////////////////////////////////////////////////////////////////////
        %////////////////////////////////////////////////////////////////////
        %//////////////////////////////////////////////////////////////////
        if DetectorModelSelector_SSD==1      %% SSD Trainer
% % %                     disp('Estimating Anchorboxes values from data..... (wait a little)');
% % %                     trainingData3_estimate = boxLabelDatastore(trainingData3(:,2:end));
% % %                     numAnchors = 12;
% % %                     [anchorBoxes,meanIoU] = estimateAnchorBoxes(trainingData3_estimate,numAnchors);
% % %                     anchorBoxes
%320X320
%                             12,12;10,41;19,38;13,29;
%                             19,23;29,49;18,14;38,25;
%620X629
%     28,59;16,31;24,21;40,81; 
%     10,37;77,44;26,41;38,32;
%     60,94;19,82;16,53;9,19;
                   options = trainingOptions('sgdm',...
                      'InitialLearnRate',1e-4,...
                      'MiniBatchSize',2,...
                      'Verbose',true,...
                      'MaxEpochs',YasserEpochs,...
                      'Shuffle','every-epoch',...
                      'VerboseFrequency',10,...
                      'GradientThreshold',0.7,...
                      'CheckpointPath',[pwd '\ssd_Natural_images_Training']);
                  if ReTrainOption==0
                                    inputSize = [imgRows imgCols imgDimension];
                                    numClasses = width(trainingData3)-1;
                                    % lgraph_1 = ssdLayers(inputSize, numClasses, 'resnet50');
                                    lgraph_1 = ssdLayers(inputSize, numClasses, Used_Model);
%                                     load('620_model_v2.mat')
                                    load('SSD_Natural_Updated_620_Model_4.mat');
                                    tic;
                                    [frcnn,info] = trainSSDObjectDetector(ds,lgraph_3,options);
                                    Y_endTime=toc;
                                    Y_TrainTime=Y_endTime;
                  else
                                    disp('Loading SSD Pretrained Model......');
                                    pause(1);
                                    temp_YasserEpochs=YasserEpochs;
                                    %% SSD 
%                                     load('Yasser_Natural_DataSet_v2_Urdu_SSD_620x620_Trained_On_157_Tested_On_25-images_n_Model-Name_resnet50_Ep1_TrainTime_394.0657seconds_.mat')                 
                                    load('Yasser_Chandio2020_Urdu_SSD_Trained_On_900_Tested_On_100-images_n_Model-Name_YsrModel_Ep240_Tr_ap-0.14908_Ts_ap-0.082421_TrainTime-44973.0582-Seconds_.mat')
                                    temp_YasserEpochs=temp_YasserEpochs+YasserEpochs;
                                    YasserEpochs=temp_YasserEpochs;
                                    tic;
%                                     [frcnn,info] = trainSSDObjectDetector(ds,frcnn,options);
                                    [frcnn,info] = trainSSDObjectDetector(ds,detector,options);
                                    Y_endTime=toc;
                                    Y_TrainTime=Y_endTime;
                  end
                    
        else        %% FasterRCNN Trainer
            
                    options = trainingOptions('sgdm',...
                              'InitialLearnRate',1e-3,...
                              'MiniBatchSize',1,...
                              'Verbose',true,...
                              'MaxEpochs',YasserEpochs,...
                              'Shuffle','every-epoch',...
                              'VerboseFrequency',10,...
                              'GradientThreshold',2,...
                              'CheckpointPath',[pwd '\FasterRCNN_Natural_images_Training']);
                    if ReTrainOption==0
                                    disp('Estimating Anchorboxes values from data..... (wait a little)');
                                    return;
                                    trainingData3_estimate = boxLabelDatastore(trainingData3(:,2:end));
                                    numAnchors = 15;
                                    [anchorBoxes,meanIoU] = estimateAnchorBoxes(trainingData3_estimate,numAnchors);
                                    anchorBoxes
                                    featureExtractionNetwork = Used_Model;
                                    featureLayer = 'activation_40_relu';

                   %                 featureExtractionNetwork = Used_Model;
                   %                 featureLayer = 'res5b_relu';
                                    
                                    numClasses = width(trainingData3)-1;
                                    inputSize = [imgRows imgCols imgDimension];
                                    lgraph_1 = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);
                                    tic;
                                    [frcnn,info] = trainFasterRCNNObjectDetector(ds, lgraph_1 , options, ...
                                        'NegativeOverlapRange', [0 0.1], ...
                                        'PositiveOverlapRange', [0.66 1], ...
                                    	'TrainingMethod', 'four-step'); %, ...
                                    %     'NumRegionsToSample',512, ...
                                    %     'BoxPyramidScale',5, ...
                                    %     'SmallestImageDimension', 224);
                                    Y_endTime=toc;
                                    Y_TrainTime=Y_endTime;
                    else
                                    temp_YasserEpochs=YasserEpochs;
                                    disp('Loading Pretrained FasterRCNN Model......');
                                    %load('Yasser_Natural_DataSet_v2_Urdu_FasterRCNN_620x620_Trained_On_157_Tested_On_25-images_n_Model-Name_resnet50_Ep20_Tr_ap-0.027263_Ts_ap-0.014706_5744.0975_.mat')
                                    %load('Yasser_Natural_DataSet_v2_Urdu_FasterRCNN_620x620_Trained_On_157_Tested_On_25-images_n_Model-Name_resnet50_Ep40_Tr_ap-0.027452_Ts_ap-0.0035014_8413.5355_.mat');
                                    %load('Yasser_Natural_DataSet_v2_Urdu_FasterRCNN_620x620_Trained_On_157_Tested_On_25-images_n_Model-Name_resnet50_Ep120_TrainTime_27489.1928seconds_.mat');
%                                   load('Yasser_Natural_DataSet_v2_Urdu_FasterRCNN_620x620_Trained_On_157_Tested_On_25-images_n_Model-Name_resnet50_Ep150_TrainTime_10212.5965seconds_.mat')
%                                   load('Yasser_Natural_DataSet_v2_Urdu_FasterRCNN_620x620_Trained_On_157_Tested_On_13-images_n_Model-Name_resnet50_Ep180_Tr_ap-0.041982_Ts_ap-0.014401_8652.1755_.mat')
%                                     load('volume2b_Yasser_Nat_DataSet_v2_Urdu_FasterRCNN_620x620_Trained_On_157_Tested_On_26-images_n_Model-Name_resnet50_Ep190_Tr_ap-0.038857_Ts_ap-0.0058764_2353.9673_.mat')
                                    load('volume2b_Yasser_Nat_DataSet_v2_Urdu_FasterRCNN_620x620_Trained_On_355_Tested_On_26-images_n_Model-Name_resnet50_Ep30_Tr_ap-0.16283_Ts_ap-0.037729_37507.8302_.mat');
%                                     load('faster_rcnn_stage_2_checkpoint__1172__2020_09_01__16_09_10.mat')
%                                   load('faster_rcnn_stage_2_checkpoint__879__2020_09_01__15_59_22.mat')
                                    temp_YasserEpochs=temp_YasserEpochs+YasserEpochs;
                                    YasserEpochs=temp_YasserEpochs;
                                    tic;
%                                       [frcnn,info] = trainFasterRCNNObjectDetector(ds, detector , options, ...
%                                         'NegativeOverlapRange', [0 0.1], ...
%                                         'PositiveOverlapRange', [0.66 1], ...
%                                         'TrainingMethod', 'four-step'); %, ...
                                    [frcnn,info] = trainFasterRCNNObjectDetector(ds, frcnn , options, ...
                                        'NegativeOverlapRange', [0 0.1], ...
                                        'PositiveOverlapRange', [0.66 1], ...
                                        'TrainingMethod', 'four-step'); %, ...
                                    Y_endTime=toc;
                                    Y_TrainTime=Y_endTime;
                    end
        end     
        %////////////////////////////////////////////////////////////////////
        %////////////////////////////////////////////////////////////////////
        %////////////////////////////////////////////////////////////////////

        old_Epochs=YasserEpochs*i;
        YsrModel_Name=[Yassers_Model num2str(size(trainingData3,1)) '_Tested_On_' num2str(size(testData,1)) '-images_n_Model-Name_' Used_Model '_Ep' num2str(old_Epochs) '_TrainTime_' num2str(Y_TrainTime) 'seconds_.mat'];
        save(YsrModel_Name,'frcnn','info','Y_TrainTime','YasserEpochs');

        figure,
        plot(info.TrainingLoss);
        grid on;
        xlabel('Number of Iterations');
        ylabel('Training Loss for Each Iteration');

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
                            % % % % Y_TrainTime=Y_endTime;
		%/////////////////////////////////////////////////////////////////////////
                %/////////////////////////////////////////////////////////////////////////////////////
                results=[];
                numImages = size(trainingData3,1);
                results= struct('Boxes',[],'Scores',[]);
                GroundTruth=table((trainingData3.Merged_Text_n_MixText_Rects_CoOrdinates));
                BlackZerosImg=0;
                hold on,
                for i = 1:numImages/1
                %                 I = (imread(stopSigns2(i).imageFileName));
                                I=imread(trainingData3.Original_Outodoor_Image_Path{i});
                                if ReduceSize4th_Selector==1
                                    BlackZerosImg=uint8(zeros(320,320,3));
                                    I = imresize(I,1/4);
                                    I= imresize(I,[233 310]);
                                    BlackZerosImg(1:233,1:310,:)=I;
                                else
                                    BlackZerosImg=uint8(zeros(620,620,3));
                                    IsizeR=size(I,1);
                                    IsizeC=size(I,2);
                                    if IsizeR > 466 || IsizeC > 620
                                             I = imresize(I,1/2);
                                             BlackZerosImg(1:465,1:620,:)=I;
                                    else
                                             BlackZerosImg(1:465,1:620,:)=imresize(I,[465 620]);
                                    end
                                end
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
%                                 [bboxes,scores] = detect(frcnn,I,'ExecutionEnvironment','gpu');
%                                 detectedImg = insertShape(I, 'Rectangle', bboxes,'Color','red');
                                [bboxes,scores] = detect(frcnn,BlackZerosImg,'ExecutionEnvironment','gpu','MiniBatchSize',1,'NumStrongestRegions',50);


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
                [ap_Train,Train_recall,Train_precision] = evaluateDetectionPrecision(results,GroundTruth);
                figure
                plot(Train_recall,Train_precision)
                grid on
                title(sprintf('Train-Set Average Precision = %.4f',ap_Train));
                TrResults={ap_Train,Train_recall,Train_precision};
                TrGroundTruth=GroundTruth;
                %/////////////////////////////////////////////////////////////////////////////////////
                %/////////////////////////////////////////////////////////////////////////////////////


                %//////////////////////////////////////////////////////////////////
                %//////////////////////////// Testing Accuracy//////////////////////////////////////
                %/////////////////////////////////////////////////////////////////////////////////////
                %/////////////////////////////////////////////////////////////////////////////////////
                figure,
                results=[];
                numImages = size(testData,1);
                results= struct('Boxes',[],'Scores',[]);
                GroundTruth=table((testData.Merged_Text_n_MixText_Rects_CoOrdinates));
                BlackZerosImg=0;
                hold on,
                for i = 1:numImages/1
                %                 I = (imread(stopSigns2(i).imageFileName));
%                                 I = imresize(imread(testData.Original_Outodoor_Image_Path{i}),1/4);
                            %     RatioPreservedImage=YsrNetCopiedCode_RatioPreserve(YourImage,EqualDimenstion)
                            %     Following function 'YsrNetCopiedCode_RatioPreserve' is only necessary for
                            %     InceptionV3. Others Alexnet+Googlenet+Squeeznet automatically adjusts for
                            %     the image input size.

                %///////////////////////////////////////////////////////////////////////////            
                % % % % %                 I=YsrNetCopiedCode_RatioPreserve(I,299);
                % % % % %                 GroundTruthCoords=cell2mat(GroundTruth.Var1(i));
                % % % % %                 GroundTruth.Var1{i,1}(1)=GroundTruth.Var1{i,1}(1)-10;   
                %///////////////////////////////////////////////////////////////////////////
                               I =imread(testData.Original_Outodoor_Image_Path{i});
                               if ReduceSize4th_Selector==1
                                    BlackZerosImg=uint8(zeros(320,320,3));
                                    I = imresize(I,1/4);
                                    BlackZerosImg(1:233,1:310,:)=I;
                               else
                                    BlackZerosImg=uint8(zeros(620,620,3));
                                    IsizeR=size(I,1);
                                    IsizeC=size(I,2);
                                    if IsizeR > 466 || IsizeC > 620
                                             I = imresize(I,1/2);
                                             BlackZerosImg(1:465,1:620,:)=I;
                                    else
                                             BlackZerosImg(1:465,1:620,:)=imresize(I,[465 620]);
                                    end
                                end
                                
                                [bboxes,scores] = detect(frcnn,BlackZerosImg,'ExecutionEnvironment','gpu','MiniBatchSize',1,'NumStrongestRegions',50);
                                detectedImg = insertShape(BlackZerosImg, 'Rectangle', bboxes,'Color','red');
%                                 [bboxes,scores] = detect(frcnn,I,'ExecutionEnvironment','gpu');
%                                 detectedImg = insertShape(I, 'Rectangle', bboxes,'Color','red');


                % % % %                 GroundTruthCoords(2)=GroundTruthCoords(2)-10;
                % % % %                 GroundTruthCoords(1)=GroundTruthCoords(1)-11;   % changing column value
                % % % %                 detectedImg = insertShape(detectedImg, 'Rectangle',GroundTruthCoords ,'Color','green');
                                imshow(detectedImg)
                                drawnow 
                                pause(0.1);
                                results(i).Boxes = bboxes;
                                results(i).Scores = scores;
                                disp(['Ts-' num2str(i)]);
% % %                                 spath=[fullfile(pwd,'SSD_Test_Detection_Results') '\SSD_Detections' num2str(i)];
% % % % %                                 savefig([fullfile(pwd,'SSD_Test_Detection_Results') '\SSD_Detections' num2str(i)])
% % %                                 f = gcf;
% % %                                 % Requires R2020a or later
% % %                                 exportgraphics(f,[spath '.png'],'Resolution',600);
                end
                results = struct2table(results);

                % GroundTruth=table((s3.RotatedCoordinates_plus_Angle));
                % [ap,recall,precision] = evaluateDetectionPrecision(results(1:20,:),GroundTruth(1:20,:));
                [ap_Test,Test_recall,Test_precision] = evaluateDetectionPrecision(results,GroundTruth);
                figure
                plot(Test_recall,Test_precision)
                grid on
                title(sprintf('Test-Set Average Precision = %.4f',ap_Test))
                xlabel('Recall');
                ylabel('Precision');
                %/////////////////////////////////////////////////////////////////////////////////////
                %/////////////////////////////////////////////////////////////////////////////////////
                TsResults={ap_Test,Test_recall,Test_precision};
                TsGroundTruth=GroundTruth;
                YsrModel_Name=[Yassers_Model num2str(size(trainingData3,1)) '_Tested_On_' num2str(size(testData,1)) '-images_n_Model-Name_' Used_Model '_Ep' num2str(old_Epochs) '_Tr_ap-' num2str(ap_Train) '_Ts_ap-' num2str(ap_Test) '_' num2str(Y_TrainTime) '_.mat'];
                save(YsrModel_Name,'frcnn','info','TrResults','YasserEpochs','TsResults','TrGroundTruth','TsGroundTruth','ap_Train','ap_Test','Y_TrainTime','YasserEpochs');
                pause(1);
                gpuDevice(1);
                pause(4);
end

disp('Returning before Ending ......');
return
%///////////////////////////////////////////////////

%%
% % % %% 
% % % 
% % % % [m,n] = size(A) ;
% % % m=51200;
% % % P = 0.90 ;
% % % idx = randperm(m)  ;
% % % YTrainingData = YSeq(:,idx(1:round(P*m))) ; 
% % % YTestingData = YSeq(:,idx(round(P*m)+1:end)) ;
% % % 
% % % YTrainingDataLabels = YOffsetCoorsTargets(idx(1:round(P*m)),1) ; 
% % % YTestingDataLabels = YOffsetCoorsTargets(idx(round(P*m)+1:end),1) ;

% % % % Sample data (54000 x 10)
% % % data = rand(54000,10);
% % % % Cross varidation (train: 70%, test: 30%)
% % % cv = cvpartition(size(data,1),'HoldOut',0.3);
% % % idx = cv.test;
% % % % Separate to training and test data
% % % dataTrain = data(~idx,:);
% % % dataTest  = data(idx,:);

% % % % % PD = 0.80 ;  % percentage 80%
% % % % % % Let P be your N-by-M input dataset
% % % % % % Solution-1 (need Statistics & ML Toolbox)
% % % % % cv = cvpartition(size(P,1),'HoldOut',PD);
% % % % % Ptrain = P(cv.training,:);
% % % % % Ptest = P(cv.test,:);