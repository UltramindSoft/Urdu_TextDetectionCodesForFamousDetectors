function data = Yasser_customreader(filename)
I=imread(filename);
data=uint8(zeros(320,320,3));
data(1:240,1:320,:)=I;
end