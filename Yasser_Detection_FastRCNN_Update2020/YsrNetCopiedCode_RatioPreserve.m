function RatioPreservedImage=YsrNetCopiedCode_RatioPreserve(YourImage,EqualDimenstion)
            %figure out the pad value to pad to white
            if isinteger(YourImage)
               pad = 1;   %white for floating point is 1.0  Or 0 for black
            else
               pad = intmax(class(YourImage));
            end
            %figure out which dimension is longer and rescale that to be the 256
            %and pad the shorter one to 256
%             EqualDimenstion=256
            newImage=[];
            [r, c, ~] = size(YourImage);
            if r > c
              newImage = imresize(YourImage, EqualDimenstion / r);
              newImage(:, end+1 : EqualDimenstion, :) = pad;
            elseif c > r
              newImage = imresize(YourImage, EqualDimenstion / c);
              newImage(end+1 : EqualDimenstion, :, :) = pad;
            else
              newImage = imresize(YourImage, [EqualDimenstion, EqualDimenstion]);
            end
            RatioPreservedImage=newImage;
end
