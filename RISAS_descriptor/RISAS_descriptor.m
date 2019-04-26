function [keypoint, descriptor] = RISAS_descriptor(RgbFile, DepthFile, kps, camK)
% Input:
% RgbFile - the RGB image path
% DepthFile - the depth image path
% kps - the coordinates of keypoints (2*N)
% camK - the instrinsic matrix of the camera 
%
% Output:
% keypoint - the coordinates of filtered keypoints (2*N)
% descriptor - feature vector of per keypoint

center = [camK(1,3), camK(2,3)];
focal = camK(1,1);

% we recommand that the following parameters are not changed.
npies = 8;          % the number of spatial blocks
nbins = 8;          % the number of grayscale blocks
dnbins = 2;         % the number of normal blocks
patchsize = 41;     % the size of patch
R = 20;             % the scaling factor of patch radius
win = 5;            % the window size
thre_dotmap = 0.9;  % id of dotproduct block = 3 if the dotproduct > thre_dotmap
threshold = 0.01;   % a threshold to weight the index
Dscale = 0.001;     % the scale of the RGB-D camera(mostly it is 1000 or 5000 which depent on the data)
topleft = [1 1];    % the position of the top-left corner of depth in the original depth image. Assumes depth is uncropped if not provided

% read data
RgbImage = imread(RgbFile);
DepthImage = imread(DepthFile);
[height, width, ~] = size(RgbImage);
gaussian  =  fspecial('gaussian', 5, 1);    % the setting of filter

% calculate the normal
[normals, pcloud, kinect_map]  =  estimate_normal(DepthImage, topleft, center, focal, win, threshold);
normals(:, :, 1) =  filter2(gaussian, normals(:, :, 1));  % smoothing the normal
normals(:, :, 2) =  filter2(gaussian, normals(:, :, 2));
normals(:, :, 3) =  filter2(gaussian, normals(:, :, 3));

% the image preprocess
GrayImage = rgb2gray(RgbImage);
GrayImage = double(GrayImage);
GrayImage = filter2(gaussian, GrayImage);
DepthImage = double(DepthImage);
DepthImage = filter2(gaussian, DepthImage);

[kps, point_num] = scale_kps(kps, DepthImage, Dscale, R);
j = 1;
[mask, angs] = lookup_table();  % serching the table to get the mask efficient
for i = 1:point_num            % calculate the descriptors looping
    point = kps(i);
    radius = round(point.r);   % calculate the initial patch radius based on empirical formula
    [patch_pcloud, interestpoint] = get_PC_patch(pcloud, kinect_map, point.x, point.y, radius); %% get the patch of pointcloud but the patch is not normalized
    flag = (size(patch_pcloud, 1) >= 2*radius^2)&(interestpoint ~= [0, 0, 0]); % the point number of the patch is enough & the depth of the interest point is not missed
    if flag
        radius = estimate_scale(patch_pcloud, interestpoint, radius, R, focal);
        point.r = radius;  % update the scale radius of the patch
        patch_rgb = get_normalize_gray_patch(GrayImage, patchsize, R, point.x, point.y, radius, point.scale);  % get the patch of depth image and normalize the patch
        patch_normal = get_normalize_normal_patch(normals, patchsize, R, point.x, point.y, radius, point.scale);  % get the patch of normals and normalize the patch
        [patch_pcloud, interestpoint] = get_PC_patch(pcloud, kinect_map, point.x, point.y, radius);  % get the patch of pointcloud using the new scale radius

        d_3D = fit_normal(patch_pcloud, interestpoint, false);  % plane fitting based on the patch get the normal of the plane
        [u, v, orentition] = map_2Dcoor(d_3D, interestpoint, point.x, point.y, height, width, focal);  % mapping the normal to the CCD to get the dominant orientation
        point.orientation = orentition;
        result_num = ceil(orentition*8/(2*pi));
        if result_num == 0
            result_num = 1;
        end
        pieindex = pieindex_table(result_num);

        [descriptor, indictor] = get_feature(patchsize, patch_rgb, mask, pieindex, npies, nbins, dnbins, patch_normal, thre_dotmap);  % calculate the descriptor
        point.descriptor = descriptor;
        point.orientation = point.orientation;

        if(indictor)
            feature(j) = point;   % mark the interest point which depth is not missed
            j = j+1;
        end
    end
end

num = size(feature, 2);
for i = 1:num
    
   keypoint(1, i) = feature(i).x;
   keypoint(2, i) = feature(i).y;
   keypoint(3, i) = feature(i).r;
   descriptor(:, i) = feature(i).descriptor(:);
end

end


function [normal, pcloud, kinect_map] = estimate_normal(DepthImage, topleft, center, focal, win, threshold)
% Estimate the normal
[pcloud, kinect_map] = map_depth_to_cloud(DepthImage, topleft, center, focal);
normal = get_pc_normal(pcloud, win, threshold);
gaussian = fspecial('gaussian', 5, 1);
normal(:, :, 1) = filter2(gaussian, normal(:, :, 1));  % smoothing the normal
normal(:, :, 2) = filter2(gaussian, normal(:, :, 2));
normal(:, :, 3) = filter2(gaussian, normal(:, :, 3));
end


function [pcloud, kinect_map] = map_depth_to_cloud(depth, topleft, center, focal)
% Convert depth image into 3D point cloud
MM_PER_M = 1000;
depth = double(depth);
depth(depth == 0) = 0;

[imh, imw] = size(depth);
kinect_map = depth&1;

% convert depth image to 3d point clouds
pcloud = zeros(imh, imw, 3);
xgrid = ones(imh, 1)*(1:imw) + (topleft(1)-1) - center(1);
ygrid = (1:imh)'*ones(1, imw) + (topleft(2)-1) - center(2);
s = xgrid.*depth/focal/MM_PER_M;
pcloud(:, :, 1) = xgrid.*depth/focal/MM_PER_M;
pcloud(:, :, 2) = ygrid.*depth/focal/MM_PER_M;
pcloud(:, :, 3) = depth/MM_PER_M;
end


function normal = get_pc_normal(pcloud, win, threshold)
% Compute surface normals by PCA
[imh, imw, ddim] = size(pcloud);
normal = zeros(size(pcloud));
for i = 1:imh
    for j = 1:imw
        minh = max(i - win, 1);
        maxh = min(i + win, size(pcloud, 1));
        minw = max(j - win, 1);
        maxw = min(j + win, size(pcloud, 2));
        index = abs(pcloud(minh:maxh, minw:maxw, 3) - pcloud(i, j, 3)) < pcloud(i, j, 3)*threshold;
        if sum(index(:)) > 3 && pcloud(i, j, 3) > 0 % the minimum number of points required
            wpc = reshape(pcloud(minh:maxh, minw:maxw, :), (maxh-minh+1)*(maxw-minw+1), 3);
            subwpc = wpc(index(:), :);
            subwpc = subwpc - ones(size(subwpc, 1), 1)*(sum(subwpc)/size(subwpc, 1));
            [coeff, ~] = eig(subwpc'*subwpc);
            normal(i, j, :) = coeff(:, 1)';
        end
    end
end

dd = sum(pcloud.*normal, 3);
normal = normal.*repmat(sign(dd), [1 1 3]);
end


function [new_kps, point_num] = scale_kps(kps, DepthImage, Dscale, R)
% Add the scale information to the keypoints
DepthImage = double(DepthImage);
point_num = size(kps, 2);
for i = 1:point_num
    new_kps(i).x = kps(1, i);
    new_kps(i).y = kps(2, i);
    u = round(new_kps(i).x);
    v = round(new_kps(i).y);
    distance = DepthImage(v, u)*Dscale;  % the distance
    new_kps(i).scale = max(0.2, (3.8-0.4*max(2, distance))/3);   % empirical equation from the article BRAND
    new_kps(i).r = R*new_kps(i).scale;    % the radius of the patch to be described
end
end


function [patch] = reshape_lxy(patch, ps, R, scale)
% Resize the patch to a common size ps*ps based on bilinear
[height, width] = size(patch);
radius = (height-1)/2;
kernel = ceil(scale);
gaussian = fspecial('gaussian', 5, kernel);
if radius > R
    patch = filter2(gaussian, patch);
end
patch = imresize(patch, [ps, ps], 'bilinear');
end


function [patch_pcloud, interestpoint] = get_PC_patch(pcloud, kinect_map, x, y, radius)
% get and normalize the patch of point cloud
patch_pcloud = [];
[height, width, ~] = size(pcloud);

num = 1;
for i = -radius:radius
    for j = -radius:radius
        v = int32(y)+i;
        u = int32(x)+j;
        if u > 0 && u < width && v > 0 && v < height&&kinect_map(v, u)~= 0
            patch_pcloud(num, 1) = pcloud(v, u, 1);
            patch_pcloud(num, 2) = pcloud(v, u, 2);
            patch_pcloud(num, 3) = pcloud(v, u, 3);
            num=num+1;
        end;
    end
end
interestpoint(1, 1) = pcloud(int32(y), int32(x), 1);
interestpoint(1, 2) = pcloud(int32(y), int32(x), 2);
interestpoint(1, 3) = pcloud(int32(y), int32(x), 3);
end


function [patch] = get_normalize_gray_patch(GrayImage, ps, R, x, y, raduis, scale)
% Resize the RGB patch to a common size ps*ps
patchsize = 2*raduis+1;
tmp_patch = zeros(patchsize);
[height, width] = size(GrayImage);

for i = -raduis:raduis
    for j = -raduis:raduis
        if int32(x)+j > 0 && int32(x)+j < width && int32(y)+i > 0 && int32(y)+i < height
            tmp_patch(raduis+1+i, raduis+1+j) = GrayImage(int32(y)+i, int32(x)+j);
        end;
    end
end

patch = zeros(ps, ps);
patch = reshape_lxy(tmp_patch, ps, R, scale);
end


function [patch_normal] = get_normalize_normal_patch(normal, ps, R, x, y, radius, scale)
% Resize the normal patch to a common size ps*ps
patchsize = 2*radius+1;
tmp_patch_normal = zeros(patchsize, patchsize, 3);
[height, width, ~] = size(normal);
for i = -radius:radius
    for j = -radius:radius
        if int32(x)+j > 0 && int32(x)+j < width && int32(y)+i > 0 && int32(y)+i < height
            tmp_patch_normal(radius+1+i, radius+1+j, 1) = normal(int32(y)+i, int32(x)+j, 1);
            tmp_patch_normal(radius+1+i, radius+1+j, 2) = normal(int32(y)+i, int32(x)+j, 2);
            tmp_patch_normal(radius+1+i, radius+1+j, 3) = normal(int32(y)+i, int32(x)+j, 3);
        end;
    end
end
patch_normal = zeros(ps, ps, 3);
patch_normal(:, :, 1) = reshape_lxy(tmp_patch_normal(:, :, 1), ps, R, scale);
patch_normal(:, :, 2) = reshape_lxy(tmp_patch_normal(:, :, 2), ps, R, scale);
patch_normal(:, :, 3) = reshape_lxy(tmp_patch_normal(:, :, 3), ps, R, scale);
end


function [radius] = estimate_scale(pointcloud, interestpoint, exradius, R, focal)
% background eliminate
average = mean(pointcloud);
dz = pointcloud(:, 3)-average(1, 3);
[row] = find(abs(dz) > 0.1);  % get rid of the points which are far away from the interesting point
pointcloud(row, :) = [];

% if the pointcloud is vary sparse, we keep on the empirical equation result
if size(pointcloud) < R  
    radius = exradius;
    return
end
normal = fit_normal(pointcloud, interestpoint, false);

% fit a plane P:Ax+By+Cz+D=0
A = normal(1, 1);
B = normal(2, 1);
C = normal(3, 1);
D = -(interestpoint*normal);
%  [x, y] = meshgrid((average(1, 1)-0.2):0.01:(average(1, 1)+0.2), (average(1, 2)-0.2):0.01:(average(1, 2)+0.2));
%  z = -(A*x+B*y+D)/C;
%  mesh(x, y, z);
%  xlabel('{\itx}/m')
%  ylabel('{\ity}/m')
%  zlabel('{\itz}/m')
pointcloud = [pointcloud, ones(size(pointcloud, 1), 1)];
P = pointcloud*[A, B, C, D]';
Q = A^2+B^2+C^2;
distance = abs(P)/sqrt(Q);
b = max(distance);  % select the max distance to the plane P  as b

% project all the points to the plane P
K = -P/Q;
point(:, 1) = pointcloud(:, 1)+K*A;
point(:, 2) = pointcloud(:, 2)+K*B;
point(:, 3) = pointcloud(:, 3)+K*C;

% build a 2D coordinate system in the plane P
p1 = [0, 0, -D/C];
p1p0 = p1-interestpoint;
p1p0 = p1p0/norm(p1p0);  % x axis
p2p0 = cross(p1p0, normal);  % y axis
% plot3([interestpoint(1, 1), interestpoint(1, 1)+p1p0(1, 1)], [interestpoint(1, 2), interestpoint(1, 2)+p1p0(1, 2)], [interestpoint(1, 3), interestpoint(1, 3)+p1p0(1, 3)], 'r-', 'LineWidth', 3);
% hold on;
% plot3([interestpoint(1, 1), interestpoint(1, 1)+p2p0(1, 1)], [interestpoint(1, 2), interestpoint(1, 2)+p2p0(1, 2)], [interestpoint(1, 3), interestpoint(1, 3)+p2p0(1, 3)], 'r-', 'LineWidth', 3);
point(:, 1) = point(:, 1)-interestpoint(1, 1);
point(:, 2) = point(:, 2)-interestpoint(1, 2);
point(:, 3) = point(:, 3)-interestpoint(1, 3);
px = point*p1p0';
py = point*p2p0';
% figure(22)
% plot(px, py, '*');
% hold on;
ellipse_t = fit_ellipse(px, py);

if isempty(ellipse_t)|| isempty(ellipse_t.a)   % if ellipse fitting failed
    a = 0;
    c = 0;
else
    a = ellipse_t.a;
    c = ellipse_t.b;
end

tmp = max(max(a, b), c);  % select the max value as the radius of the pointcloud
radius = ceil(tmp*focal/interestpoint(1, 3));  % project the radius  to the image plane
if radius > 50  % if the result is too awful, we keep on the empirical equation result
    radius = exradius;
end
end


function [u, v, orentition] = map_2Dcoor(normal, interestpoint, x, y, height, width, focal)
% Map the 3D dominant orientation d_3D to the CCD to get the dominant orientation
normal_world = normal'+interestpoint;
u = normal_world(1, 1)*focal/normal_world(1, 3)+width/2;
v = normal_world(1, 2)*focal/normal_world(1, 3)+height/2;
du = u-x;
dv = y-v;
if du > 0
    orentition = mod(atan(dv/du), 2*pi);
else
    orentition = atan(dv/du)+pi;
end
end


function [desD, indictor] = get_feature(patchsize, patch, mask, pieindex, npies, nbins, dnbins, normal, thre_dotmap)
% Compute the feature vector
radius = int32((patchsize-1)/2);
center_x = patchsize-radius;
center_y = patchsize-radius;

number_piex = sum(sum(mask));    % calculate the total number of the pixels
mask_normal = normal(:, :, 1)|normal(:, :, 2)|normal(:, :, 3);  % get the mask map corresponding to the normal
if mask_normal(center_x, center_y) == 0  % if the keypoint's normal is lost
    desD = 0;
    indictor = 0;
    return;
end;
number_normal = sum(sum(mask_normal));  % calculate the total number of the pixels whose normal is not lost

if number_normal < number_piex*0.7
    desD = 0;
    indictor = 0;
    return;
end

indictor = 1;
center_normalx = normal(center_x, center_y, 1);
center_normaly = normal(center_x, center_y, 2);
center_normalz = normal(center_x, center_y, 3);

dotproduct = zeros(size(patch));

for i = -radius : radius
    for j = -radius :radius
        nx = normal(center_x+i, center_y+j, 1);
        ny = normal(center_x+i, center_y+j, 2);
        nz = normal(center_x+i, center_y+j, 3);
        dotproduct(center_x+i, center_y+j) = abs(nx*center_normalx+ny*center_normaly+nz*center_normalz);  % calculate the dotproduct
        if isnan(dotproduct(center_x+i, center_y+j))
            dotproduct(center_x+i, center_y+j) = 0;
        end
    end;
end;

uniq_patch = unique(patch(:));   % calculate the gray levels of the patch to be described
uniq_dotproduct = unique(dotproduct(dotproduct < thre_dotmap));   % calculate the dotproduct levels of the patch to be described

number_nbins = ceil(size(uniq_patch, 1)/nbins);   % the number of gray levels in each grayscale block
upper_bound_gray = ones(1, nbins);
for i = 1:1:nbins
    index = i*number_nbins;
    if index > size(uniq_patch, 1)
        index = size(uniq_patch, 1);
    end
    upper_bound_gray(i) = uniq_patch(index);  % get the upper bound of each grayscale block
end

number_dnbins = ceil(size(uniq_dotproduct, 1)/dnbins);  % the number of dotproduct levels in each dotproduct block
upper_bound_dotproduct = ones(1, dnbins);
if number_dnbins ~= 0
    for i = 1:1:dnbins
        index = i*number_dnbins;
        if index > size(uniq_dotproduct, 1)
            index = size(uniq_dotproduct, 1);
        end
        upper_bound_dotproduct(i) = uniq_dotproduct(index); % get the upper bound of each dotproduct block
    end
else
    for i = 1:1:dnbins
        upper_bound_dotproduct(i) = 0;
    end
end

histogram = zeros(nbins, npies, dnbins+1);   % 3D statistical histogram
for i = -radius : radius
    for j = -radius : radius
        if mask(center_x+i, center_y+j) == 0
            continue;
        end;
        for index = 1:1:nbins
            if patch(center_x+i, center_y+j) <= upper_bound_gray(index)
                nbin_id = index;  % the id of grayscale block
                break;
            end
        end;

        if dotproduct(center_x+i, center_y+j) >= thre_dotmap  % if the dotproduct is greater than thre_dotmap, the id of dotproduct block is 3
            dnbin_id = 3;
        else
            for index = 1:1:dnbins
                if dotproduct(center_x+i, center_y+j) <= upper_bound_dotproduct(index)
                    dnbin_id = index;  % the id of dotproduct block
                    break;
                end
            end;
        end;

        npie_id = pieindex(center_x+i, center_y+j);   % the id of the spatial block

        histogram(nbin_id, npie_id, dnbin_id) = histogram(nbin_id, npie_id, dnbin_id) + 1;
    end;
end;

desD = histogram(:);
end

