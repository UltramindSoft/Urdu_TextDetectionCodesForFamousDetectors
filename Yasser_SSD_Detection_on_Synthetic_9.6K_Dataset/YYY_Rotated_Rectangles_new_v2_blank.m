function [RotatedNewCoordinates]=YYY_Rotated_Rectangles_new_v2_blank(OldRectanglePoints,GivenAngle)
% Usage Info ...
% % RotatedNewCoordinates=YYY_Rotated_Rectangles_new_v2_blank('peppers.png',[150 150 100 50],-10)
% 
% ix=imread(imageName);
% imshow(ix);

%%
%/////////////////////////////////////////////////////////////////////
Y_Rec_Coords= OldRectanglePoints;

hold on


xScaleHeightScaling=1;
yScaleWidthScaling=1;

Y_TransformationMatrix=[cos(GivenAngle),-sin(GivenAngle) sin(GivenAngle);cos(GivenAngle) 0 0; 0 0 1]';
% NewT=maketform('affine',[cos(GivenAngle),-sin(GivenAngle) sin(GivenAngle);cos(GivenAngle) 0 0; 0 0 1]');
NewT=maketform('affine',Y_TransformationMatrix);
% NewT

% New Top-Left Coordinates
NewCoordinates1=tformfwd(Y_Rec_Coords(:,1:2),NewT);
TLCx=Y_Rec_Coords(1);
TLCy=Y_Rec_Coords(2);

WidthCoords=Y_Rec_Coords(1)+Y_Rec_Coords(3);     % column + width
HeightCoords=Y_Rec_Coords(2)+Y_Rec_Coords(4);    % row + Height

% New Top-right Coordinates
TRCx=WidthCoords;
TRCy=TLCy;

% New bottom-left Coordinates
BLCx=TLCx;
BLCy=TLCy+Y_Rec_Coords(4);

% New bottom-right Coordinates
BRCx=TRCx;
BRCy=BLCy;

NewUnTransformedCoordinates=[TLCx TLCy BLCx BLCy BRCx BRCy TRCx TRCy];

Y_QuadrilaterX_Column_Points=[TLCx BLCx BRCx TRCx TLCx];
Y_QuadrilaterY_Rows_Points=[TLCy BLCy BRCy TRCy TLCy];

% % % % % plot(Y_QuadrilaterX_Column_Points, Y_QuadrilaterY_Rows_Points, 'color','c','LineWidth',7); 



%%

% figure,
% A = [-2,-2,6,6,-2; -2,2,2,-2,-2; 1 1 1 1 1];
A =[Y_QuadrilaterX_Column_Points;Y_QuadrilaterY_Rows_Points; 1 1 1 1 1];

% Define Translation Matrix
trans = @(x,y,z) repmat([x; y; z],[1 5]);

% Define Rotation Matrix
se2 = @(x, y, theta) [
    cosd(theta), -sind(theta), x;
    sind(theta), cosd(theta), y;
    0,        0,           1];

% Calculate Rotated Rect
RotatedNewCoordinates = se2(0,0,GivenAngle) * (A - trans(TLCx,TLCy,0) ) + trans(TLCx,TLCy,0);
hold on
% Plot Rectangles
% figure; 
% % % % % % % % plot(A(1,:),A(2,:),'b','LineWidth',4)
% % % % % % % % hold on;
% % % % % % % % plot(RotatedNewCoordinates(1,:),RotatedNewCoordinates(2,:), 'color','r','LineWidth',2)
% % % % % % % % hold off;
% axis equal

% RotatedNewCoordinates
