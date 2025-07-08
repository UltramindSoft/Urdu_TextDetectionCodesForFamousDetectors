function data = Yasser_customreader_X(filename)
filename
I=imread(filename);
% I=imresize(I,[420,338]);
data=imresize(I,[450,338]);
% data=uint8(zeros(320,320,3));
% data(1:233,1:310,:)=I;
end