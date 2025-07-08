clc
close all
clear all

xt1=imread('peppers.png');
figure,imshow(xt1);

YFontSize=44;
feature('locale');
feature('DefaultCharacterSet', 'UTF-8');


BackGroundImagesDir=dir(fullfile(pwd,'BackGround_Images','*.jpg'));
LengthOfBackImages=size(BackGroundImagesDir,1);
PickRandomBackImages=randi(LengthOfBackImages,[1 LengthOfBackImages]);


% baseName='Mat_Files_UPTi';
% YasserPauseTime=0.0125;
YasserPauseTime=0.025;
% YFontName='Jameel Noori Nastaleeq';
FontNameList={ 'Adobe Naskh'  'Jameel Noori Nastaleeq'  'Dubai' };
FontSizeList=[54, 44];
YassersSecondName='Arafat ...';
RotatedDestination_Folder='Ysr_Rotation_CLE_DataSet_v1_Multi_Font';

if ~exist(RotatedDestination_Folder,'dir')
    mkdir(fullfile(pwd,RotatedDestination_Folder));
    disp('Creating Folder ....');
else
    disp('Folder Already exist .....');
end
% return;

% % % % % YassersSecondName=native2unicode("?? ???? ??",'UTF8');
% % % % % 
% % % % % load([pwd '\' baseName '\Yasser_images_n_titles_upti.mat']);
% % % % % YassersSecondName=Yasser_upti_data(1).Y_SentenceLabel;

load('CLE_Normalized_DataSet_Yasser_Generated_7k_testSet_v1.mat');
% YassersSecondName=Yasser_CLE_DataSet(2341).OrignialLigature
% size(YassersSecondName,2)
WidthOfRectangle=size(YassersSecondName,2)*1;
HeightOfRectangle=YFontSize;
TrainingDataForRegression= struct;
TrainingDataYCounter=1;

TrainingDataForRegression4DImagesRotation=zeros(240,320,3);
TrainingDataForRegression4DTargetsRotation=[];
TrainingDataForRegression4DTargetsQuadrilateral=[];

YColorVector=[ 'r' 'g' 'b' 'k' 'm' 'y' 'c' 'w'];

rng('default');
Yasser_BoundingLimiter=100;
for yKFont_Size=1:length(FontSizeList)  %  2-Font Sizes
             for yKFont=1:length(FontNameList)   %  3-font styles    3
                             YFontName=char(FontNameList(yKFont));
                                        for yKcolor=1:4    % 4-Font Colors
                                            yKcolor
                                                  for chooseBack=1:4    % 4-Backgrounds
                                                    BackImagePath=fullfile(BackGroundImagesDir((chooseBack)).folder,BackGroundImagesDir(PickRandomBackImages(chooseBack)).name);
                                                    BackGround=imread(BackImagePath);
                                                    BackGround=imresize(BackGround,[240 320]);
                                                    for LoopOverLigatures=1:100
                                                            YassersSecondName=Yasser_CLE_DataSet(LoopOverLigatures).OrignialLigature;
                                                            WidthOfRectangle=size(char(YassersSecondName),2)*YFontSize;
                                                            clf
                                                            imshow(BackGround);
                                                            drawnow 
                                                            pause(YasserPauseTime);


                                                            BoundrySafeHeightFactor=HeightOfRectangle*1.2;
                                                            b_MaXrow=size(BackGround,1)-((HeightOfRectangle)+Yasser_BoundingLimiter);
                                                            a_MiNrow=(HeightOfRectangle+BoundrySafeHeightFactor);  

                                                            BoundrySafeWidthFactor=WidthOfRectangle*1.2;
                                                            b_MaXcolumn=size(BackGround,2)-((WidthOfRectangle)+Yasser_BoundingLimiter);
                                                            a_MiNcolumn=(WidthOfRectangle+BoundrySafeWidthFactor);

                                                            %LABEL YasserBlock

                                                            InitialColumnCoordinates=(b_MaXcolumn-a_MiNcolumn)*rand(1,1)+a_MiNcolumn;
                                                            InitialRowCoordinates= ((b_MaXrow-a_MiNrow)*rand(1,1)+a_MiNrow);

                                                            RecRatio2=WidthOfRectangle/2;
                                                            IC=InitialColumnCoordinates-RecRatio2;
                                                            IR=InitialRowCoordinates-RecRatio2;


                                        %                     RecCoors=[ IR IC  ceil((char(YassersSecondName)*YFontSize)/1.5) HeightOfRectangle];


                                                            %%////////////////////////////////////////////////////////////////////////////////////////
                                                            % DesiredAngle=40
                                                            %%  When Changing Angles, we need to adjust atleast following 8-lines for rectangle setting (Yasser)
                                                            YasserAngleRange=45;
                                                            yMaxAngle=-YasserAngleRange;
                                                            yMinAngle=YasserAngleRange;
                                                            yAngleRange=[yMaxAngle-yMinAngle];
                                                            DesiredAngle=yAngleRange*rand(1,1)+yMinAngle;
                                                            DesiredAngle=round(DesiredAngle);

                                                            %%////////////////////////////////////////////////////
                                                            %//////////////////////////////////////////////////////
                                        %                     DesiredAngle=0;
                                                              if InitialColumnCoordinates >= 220
                                                                 InitialColumnCoordinates=InitialColumnCoordinates-200;
                                                              end

                                                              if InitialColumnCoordinates <= 45
                                                                InitialColumnCoordinates=InitialColumnCoordinates+Yasser_BoundingLimiter/2;
                                                              end

                                                               if InitialRowCoordinates >= 180
                                                                InitialRowCoordinates=InitialRowCoordinates-160;
                                                               end

                                                              if InitialRowCoordinates <=45
                                                                InitialRowCoordinates=InitialRowCoordinates+Yasser_BoundingLimiter/2;
                                                              end

                                                              h2=text(InitialColumnCoordinates,InitialRowCoordinates,YassersSecondName,'Color',YColorVector(yKcolor),'Interpreter','none','FontSize',FontSizeList(yKFont_Size),'FontName',YFontName);
                                                                                                    pause(YasserPauseTime);
                                                              ext = get(h2,'extent');
                                                              pause(YasserPauseTime);
                                        %                     rectangle('Position', ext , 'EdgeColor','c');   
                                        %                     pause(YasserPauseTime);

                                                              CorrectedExt=round(floor([ext(1) ext(2)-YFontSize*2 ext(3)+YFontSize/3  ext(4)]));

                                        %                     if ext(1) > 220
                                        %                         ext(1)=ext(1)-100
                                        %                     end
                                        % % % %                     rectangle('Position', CorrectedExt , 'EdgeColor','b','LineWidth',2);   
                                        % % % %                     pause(YasserPauseTime);
                                        % % % %                     CorrectedExt

                                                            set(h2,'Rotation',-DesiredAngle);
                                                            pause(YasserPauseTime);

                                                           MoveLeftFactor=15;
                                                           MoveUpFactor=7;
                                                           GreenRecCoors=round(floor([CorrectedExt(1)-MoveLeftFactor  CorrectedExt(2)- MoveUpFactor CorrectedExt(3)+ MoveLeftFactor*2 CorrectedExt(4)+MoveUpFactor*2]));

                                                           if DesiredAngle >=0
                                                              GreenRecCoorsForRotation=round(floor([GreenRecCoors(1)+DesiredAngle/2    GreenRecCoors(2)+DesiredAngle/2   GreenRecCoors(3)  GreenRecCoors(4)]));
                                                           else
                                                               GreenRecCoorsForRotation=round(floor([GreenRecCoors(1)+DesiredAngle/2    GreenRecCoors(2)-DesiredAngle/2   GreenRecCoors(3)  GreenRecCoors(4)]));
                                                           end

                                                           RotatedNewCoordinates=round(floor(YYY_Rotated_Rectangles_new_v2_blank(GreenRecCoorsForRotation,DesiredAngle)));
                                                           QuadriLateralFeatures=[reshape(RotatedNewCoordinates(1:2,1:4),[1 8]) -DesiredAngle a_MiNrow HeightOfRectangle ] ;


                                                            allOneString = sprintf('%.0f,' , GreenRecCoors);
                                                            allOneString = allOneString(1:end-1);

                                        % % %                     TrainingDataForRegression4DTargetsHorizontal=GreenRecCoors;
                                        % % %                     TrainingDataForRegression4DTargetsHorizontal=[TrainingDataForRegression4DTargetsHorizontal; [reshape(RotatedNewCoordinates(1:2,1:4),[1 8])  a_MiNrow HeightOfRectangle ]];
                                                            % saveas(gcf,'Yassers_Rotated_Rectangle.png')
                                                            t = datetime('now','TimeZone','Asia/Kolkata','Format','d-MMM-y-HH=mm=ss');
                                        %                     NewFileNameTry=[ '_' num2str(LoopOverLigatures) '_' num2str(chooseBack) '_Angle=' num2str(-DesiredAngle) '_time_' char(t) '_Coors_' num2str(GreenRecCoors(1)) '_' num2str(GreenRecCoors(3)) '_' num2str(GreenRecCoors(4)) '_' num2str(GreenRecCoors(1)) '___.jpg'];
                                                            NewYFileName=[ num2str(TrainingDataYCounter) '#_' num2str(yKcolor) '_' num2str(LoopOverLigatures) '_' num2str(chooseBack) '_Angle=' num2str(-DesiredAngle) '_time_' char(t) '_Coors_' num2str(GreenRecCoors(1)) '_' num2str(GreenRecCoors(3)) '_' num2str(GreenRecCoors(4)) '_' num2str(GreenRecCoors(1)) '__' char(YassersSecondName) '_.jpg'];

                                                            cData=[];
                        %                                     a = getframe(gcf); %getting a screen shot for the currently opened plot window
                                                            a = getframe;
                                                            pause(0.05);
                                                            ImgWritingPath=fullfile(RotatedDestination_Folder,NewYFileName);
                                                            TrainingDataForRegression(TrainingDataYCounter).imageFileName=ImgWritingPath;
                                                % %             TrainingDataForRegression(TrainingDataYCounter).RotatedCoordinates_plus_Angle= GreenRecCoors;
                                                            TrainingDataForRegression(TrainingDataYCounter).RotatedCoordinates_plus_Angle= allOneString;
                                                            TrainingDataForRegression(TrainingDataYCounter).Ysr_Angle= -DesiredAngle;
                                                            TrainingDataForRegression(TrainingDataYCounter).text_in_box_wordForm= Yasser_CLE_DataSet(LoopOverLigatures).OrignialLigature;
                                                            TrainingDataForRegression(TrainingDataYCounter).text_in_box_charForm= Yasser_CLE_DataSet(LoopOverLigatures).OrignialUnicodeLigaturesCharacterWithSpacing;
                                                            TrainingDataForRegression(TrainingDataYCounter).Quadrilateral_Features_plus_Angle=QuadriLateralFeatures;
 
                        %                                          saveas(gcf,ImgWritingPath);
                                                            catch

                                                                 NewYFileNameTry=[ num2str(TrainingDataYCounter) '#_' num2str(yKcolor) '_' num2str(LoopOverLigatures) '_' num2str(chooseBack) '_Angle=' num2str(-DesiredAngle) '_time_' char(t) '_Coors_' num2str(GreenRecCoors(1)) '_' num2str(GreenRecCoors(3)) '_' num2str(GreenRecCoors(4)) '_' num2str(GreenRecCoors(1)) '__XX__.jpg'];
                                                                 ImgWritingPath=fullfile(RotatedDestination_Folder,NewYFileNameTry);
                                                                 imwrite(cData_beebo,ImgWritingPath );
                                                                 TrainingDataForRegression(TrainingDataYCounter).imageFileName=ImgWritingPath;
                                        %                          keyboard
                                        %                          cData=a.cdata;
                                                            end

                                        %                     imwrite(Y_imageFileToBeWritten,fullfile(RotatedDestination_Folder,NewYFileName));
                                        %                     rectangle('Position', RecCoors , 'EdgeColor','r'); 
                                                            rectangle('Position', GreenRecCoors , 'EdgeColor','g');   
                                                            pause(YasserPauseTime);

                                                            plot(RotatedNewCoordinates(1,:),RotatedNewCoordinates(2,:) , 'color','r','LineWidth',1); 
                                                            pause(YasserPauseTime);
                                                            % 
                                        % %                     RotatedNewCoordinates=YYY_Rotated_Rectangles_new_v2_blank(RecCoors,DesiredAngle);
                                        % %                     plot(RotatedNewCoordinates(1,:),RotatedNewCoordinates(2,:) , 'color','y','LineWidth',3); 


                                                            %%////////////////////////////////////////////////////////////////////////////////////////
                                        %%    
                                        % % % % % % % %                     imageFor4D=imread(ImgWritingPath);  
                                        %                     figure,imshow(imageFor4D);
                                        %                     pause(2);
                                        %                     close figure 2
                                        % % % % % % % %                     TrainingDataForRegression4DImagesRotation(:,:,:,TrainingDataYCounter)=imageFor4D;
                                        % % % % % % % %                     TrainingDataForRegression4DTargetsRotation=[TrainingDataForRegression4DTargetsRotation;-DesiredAngle];
                                        % % % % % % % %                     TrainingDataForRegression4DTargetsQuadrilateral=[TrainingDataForRegression4DTargetsQuadrilateral;QuadriLateralFeatures];
                                        % % % % % % % % 
                                        % % % % % % % %     

                                        %%
                                                            TrainingDataYCounter
                                                            TrainingDataYCounter=TrainingDataYCounter+1;
                                                      end
                                                  end

                                        end
             end
end
                TotalNumberOfLigatures=LoopOverLigatures;
                TrainingDataYCounter=TrainingDataYCounter-1;
                save (['Training_Data_Regression_CLE_Multi_Font_v1_' RotatedDestination_Folder '_Ysr_NoOf_Images_' num2str(TrainingDataYCounter) '_' num2str(yKFont_Size) '_' num2str(yKFont) '_' num2str(yKcolor) '_' num2str(chooseBack) '_' num2str(LoopOverLigatures) '_.mat'],'TrainingDataForRegression','TotalNumberOfLigatures');

                %%
                % % % % % % % % save (['4D_Training_Data_Regression_' RotatedDestination_Folder '_i' num2str(TrainingDataYCounter) '_.mat'],'TrainingDataForRegression4DImagesRotation','TrainingDataForRegression4DTargetsRotation','TrainingDataForRegression4DTargetsQuadrilateral','-v7.3');
