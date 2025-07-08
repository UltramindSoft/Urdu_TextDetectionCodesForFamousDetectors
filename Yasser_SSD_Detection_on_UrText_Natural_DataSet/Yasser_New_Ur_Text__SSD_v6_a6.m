clear all
close all
clc

gpuDevice(1);
Yassers_Model=['Yasser_Nat_Ur_Text_Urdu_SSD_450x338_Trained_On_'];

Ysr_Model_Epochs=30;
%%
% Transfer Learning Model

% Used_Model='resnet50';
% Ysr_FL='activation_40_relu';

Used_Model='vgg16';
Ysr_FL='relu5_3';

%%
% Training Coordinates File
load('Volume_#2b_Final_Info__Urdu-Text-images-573_Mixed-Text-336_Signboards-400_V2_.mat')

% data = load('Volume_#2b_Final_Info__Urdu-Text-images-573_Mixed-Text-336_Signboards-400_V2_.mat',  'Yasser_OutDoor_Urdu_Text_Recognition_structure');
% Yasser_Urdu_Text = data.Yasser_OutDoor_Urdu_Text_Recognition_structure;
% stopSigns2=struct2table(Yasser_Urdu_Text);

stopSigns2=struct2table(Yasser_OutDoor_Urdu_Text_Detection_structure);
% stopSigns2b = stopSigns2(1:250,[1,5]);    % trains on First & Last column
% stopSigns2b = stopSigns2(1:250,[1:4]);    % trains on First,to 4th column
stopSigns2b = stopSigns2(1:504,[1,5]);    % trains on First & Last column
% stopSigns2b = stopSigrns2(1:5,[1,2]);    % trains on First & Last column
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
    stopSigns2b.Merged_Text_n_MixText_Rects_CoOrdinates{kYasser}=temp;
end
s3=stopSigns2b;


%%//////////////////////////////////////////////////////////////////////////
% Set random seed to ensure example training reproducibility.
% rng(0);
rng('default');
% rng(7);
% Randomly split data into a training and test set.
shuffledIndices = randperm(height(s3));
idx = floor(0.10 * length(shuffledIndices) );
trainingData = s3(shuffledIndices(1:idx),:);
testData = s3(shuffledIndices(idx+1:end),:);
% Ysr_Final_Text_CoOrs=(Yasser_Urdu_Text);

% imgRows=900;
% imgCols=680;

imgRows=338;
imgCols=450;

inputImageSize=[imgRows imgCols 3];
NewFolderName=['volume_2b_images__' num2str(imgRows) 'X' num2str(imgCols)];
NewFolder=mkdir(NewFolderName);
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                 Training Data Cleansed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainingData_Cleansed=trainingData;
ColExceeded=0
RowExceeded=0
BoundaryFlag=0
for LoopSize=1:size(trainingData_Cleansed,1)/1
                imP=trainingData_Cleansed.Original_Outodoor_Image_Path{LoopSize};
                [filepath,name,ext] = fileparts(imP);
                 NewPath_n_File=[fullfile(pwd,NewFolderName) '\' name ext];
                  imG=imread(imP);
                  scale = (inputImageSize(1:2)./size(imG,[1 2]));
                  New_imG=imresize(imG,[imgRows imgCols]);
                  imwrite(New_imG,NewPath_n_File);
                  trainingData_Cleansed.Original_Outodoor_Image_Path{LoopSize}=NewPath_n_File;
                  Old_CoOrs=uint16(trainingData_Cleansed.Merged_Text_n_MixText_Rects_CoOrdinates{LoopSize});
                  trainingData_Cleansed.Merged_Text_n_MixText_Rects_CoOrdinates{LoopSize} = bboxresize(double(Old_CoOrs),scale)+1;
                   CoOrs=trainingData_Cleansed.Merged_Text_n_MixText_Rects_CoOrdinates{LoopSize};
%                   imshow(New_imG);
%                    hold on
%                    rectangle('Position',CoOrs,'EdgeColor','r','LineWidth',2);
%                    pause(0.25);
%                    hold off
LoopSize
end
ColExceeded
RowExceeded
pause(0.5);
%Remove empty rows, acoording to 2nd empty column
idx=all(cellfun(@isempty,trainingData_Cleansed{:,2}),2);
trainingData_Cleansed(idx,:)=[];


%%

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                 Testing Data Cleansed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
testData_Cleansed=testData;
ColExceeded=0
RowExceeded=0
BoundaryFlag=0
for LoopSize=1:size(testData_Cleansed,1)/1
                  imP=testData_Cleansed.Original_Outodoor_Image_Path{LoopSize}; 
                  imG=imread(imP);
                  [filepath,name,ext] = fileparts(imP);
                  NewPath_n_File=[fullfile(pwd,NewFolderName) '\' name ext];
                  scale = (inputImageSize(1:2)./size(imG,[1 2]));
                  New_imG=imresize(imG,[imgRows imgCols]);
                  imwrite(New_imG,NewPath_n_File);
                  testData_Cleansed.Original_Outodoor_Image_Path{LoopSize}=NewPath_n_File;
                  Old_CoOrs=uint16(testData_Cleansed.Merged_Text_n_MixText_Rects_CoOrdinates{LoopSize});
                  testData_Cleansed.Merged_Text_n_MixText_Rects_CoOrdinates{LoopSize} = bboxresize(double(Old_CoOrs),scale)+1;
%                 imG=imread(imP);
%                 imshow(imG);
%                 hold on
                   CoOrs=testData_Cleansed.Merged_Text_n_MixText_Rects_CoOrdinates{LoopSize};
                    
%                    hold on
%                    rectangle('Position',CoOrs,'EdgeColor','r','LineWidth',2);
%                    pause(0.25);
%                    hold off
LoopSize

end
ColExceeded
RowExceeded

%Remove empty rows, acoording to 2nd empty column
idx=all(cellfun(@isempty,testData_Cleansed{:,2}),2);
testData_Cleansed(idx,:)=[];
%%
%%

%%

imdsTrain=imageDatastore(trainingData_Cleansed.Original_Outodoor_Image_Path);
blds_Train=boxLabelDatastore(trainingData_Cleansed(:,2));
ds_Train=combine(imdsTrain, blds_Train);

imdsTest=imageDatastore(testData_Cleansed.Original_Outodoor_Image_Path);
blds_Test=boxLabelDatastore(testData_Cleansed(:,2));
ds_Test=combine(imdsTest, blds_Test);

% imdsTrain=imageDatastore(trainingData_Cleansed.Original_Outodoor_Image_Path,'ReadFcn',@Yasser_customreader_X);
% blds_Train=boxLabelDatastore(trainingData_Cleansed(:,2));
% ds_Train=combine(imdsTrain, blds_Train);
% 
% imdsTest=imageDatastore(testData_Cleansed.Original_Outodoor_Image_Path,'ReadFcn',@Yasser_customreader_X);
% blds_Test=boxLabelDatastore(testData_Cleansed(:,2));
% ds_Test=combine(imdsTest, blds_Test);

% imdsTrain=imageDatastore(trainingData.Original_Outodoor_Image_Path);
% blds_Train=boxLabelDatastore(trainingData(:,2));
% ds_Train=combine(imdsTrain, blds_Train);
% 
% imdsTest=imageDatastore(testData.Original_Outodoor_Image_Path);
% blds_Test=boxLabelDatastore(testData(:,2));
% ds_Test=combine(imdsTest, blds_Test);

%%

%%
% %%   Routine to verify correctness of annotation.
% while hasdata(ds_Test)
%     T = read(ds_Test);
%     RGB = insertObjectAnnotation(T{1},'rectangle',T{2},'YYY--',...
%     'TextBoxOpacity',0.9,'FontSize',18);
%     imshow(RGB)
%     title('Annotated Mine Dataset');
%     pause(0.5);
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % % disp('Returning Earlier ...');
% % % return;

% imds_Train_90 = transform(ds_Train,@(data)preprocessData(data,inputImageSize));

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 1, ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 4, ...
    'ExecutionEnvironment','gpu',...
    'CheckpointPath', [pwd, '\SSD_Natural_images_Training']);

anchorBoxes = [
    28,59;16,31;24,21;40,81; 
    10,37;77,44;26,41;38,32;
    60,94;19,82;16,53;9,19;
];

featureExtractionNetwork = Used_Model;
featureLayer = Ysr_FL;
numClasses = width(trainingData_Cleansed)-1;

        ssd_detector = ssdLayers(inputImageSize, numClasses, featureExtractionNetwork)
        tic;
         [detector,info] = trainSSDObjectDetector(ds_Train,ssd_detector,options);
        Y_endTime=toc;
        Y_TrainTime=Y_endTime;

                                   
save('Yasser_450x338_pixels_SSD_OutDoor_Resnet50_v4.mat', 'detector','info','Y_TrainTime');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%/////////////////////////////////////////////////////////////////////////
                %/////////////////////////////////////////////////////////////////////////////////////
                results=[];
                numImages = size(trainingData_Cleansed,1);
                results= struct('Boxes',[],'Scores',[]);
                GroundTruth=table((trainingData_Cleansed.Merged_Text_n_MixText_Rects_CoOrdinates));
                BlackZerosImg=0;
                hold on,
                for i = 1:numImages/1
                                I=imread(trainingData_Cleansed.Original_Outodoor_Image_Path{i});
                                [bboxes,scores] = detect(detector,I,'ExecutionEnvironment','gpu','MiniBatchSize',1,'NumStrongestRegions',30);
                                results(i).Boxes = bboxes;
                                results(i).Scores = scores;
                                disp(['Tr-' num2str(i)]);
                end
                results = struct2table(results);
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
%                 GroundTruth=table((testData_Cleansed.Merged_Text_n_MixText_Rects_CoOrdinates));
                GroundTruth=table((testData_Cleansed.Merged_Text_n_MixText_Rects_CoOrdinates(1:4)));
                BlackZerosImg=0;
                hold on,
                for i = 1:numImages/100
                               I =imread(testData_Cleansed.Original_Outodoor_Image_Path{i});
                              
                                [bboxes,scores] = detect(detector,I,'ExecutionEnvironment','gpu','MiniBatchSize',1,'NumStrongestRegions',30);
                                detectedImg = insertShape(I, 'Rectangle', bboxes,'Color','red');
                                imshow(detectedImg)
                                drawnow 
                                pause(0.1);
                                results(i).Boxes = bboxes;
                                results(i).Scores = scores;
                                disp(['Ts-' num2str(i)]);
                                
                                spath=[fullfile(pwd,'SSD_Test_Detection_Results') '\FasterRCNN_Detections' num2str(i)];
%                                 savefig([fullfile(pwd,'FasterRCNN_Test_Detection_Results') '\FasterRCNN_Detections' num2str(i)])
                                f = gcf;
                                % Requires R2020a or later
                                exportgraphics(f,[spath '.png'],'Resolution',150);
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
                YsrModel_Name=[Yassers_Model num2str(size(trainingData,1)) '_Tested_On_' num2str(size(testData,1)) '-images_n_Model-Name_' Used_Model '_Ep' num2str(Ysr_Model_Epochs) '_Tr_ap-' num2str(ap_Train) '_Ts_ap-' num2str(ap_Test) '_' num2str(Y_TrainTime) '_.mat'];
                save(YsrModel_Name,'detector','info','TrResults','Ysr_Model_Epochs','TsResults','TrGroundTruth','TsGroundTruth','ap_Train','ap_Test','Y_TrainTime');
                
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%