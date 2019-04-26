
function [keypoints, pcloud, mainNormal, dotMap, DepthMap, R_rgb, R_depth, R] = RISAS_detector(rgbFile, depthFile, camK)
% Get the key points from the RISAS feature
%
% Input:
% rgbImage - the RGB image path 
% DepthImage - the depth image path
% camK - the instrinsic matrix of the camera 
%
% Output:
% keypoints - key points (2xN)

center = [camK(1,3), camK(2,3)];
focal = camK(1,1);

% we recommand that the following parameters are not changed.
alpha = 0.04;         % weight the two kinds of evaulation
gama = 0.001;        
topleft = [1 1];      % the position of the top-left corner of depth in the original depth image. Assumes depth is uncropped if not provided
win = 5;              % window size
threshold = 0.01;     
si = 1;               % the variance of the Gaussian window
sd = 0.7*si;
k=8;
belt = 10^(k);        % to balance the gray value and the dotmap value

% compute gray component
rgbImage = imread(rgbFile);
DepthImage = imread(depthFile);

I = double(rgb2gray(rgbImage));
I = vl_imsmooth(I, sd);     % gaussian blur

[Ix, Iy] = gradient(I) ;
H11 = vl_imsmooth(Ix.*Ix, si) ;
H12 = vl_imsmooth(Ix.*Iy, si) ;
H22 = vl_imsmooth(Iy.*Iy, si) ;

dt = H11.*H22 - H12.^2;    % dt is the determinant
tr = H11+H22;              % tr is the trace
R_rgb = dt-alpha*(tr.^2);

% compute depth component 
[pcloud, ~] = map_depth_to_cloud(DepthImage, topleft, center, focal);
normals = get_pc_normal(pcloud, win, threshold);
mainNormal = get_main_normal(normals);

xCh = normals(:, :, 1);
yCh = normals(:, :, 2);
zCh = normals(:, :, 3);

% % draw the main normal vector 
% surf(xCh,yCh,zCh,'linestyle','none')
% shading interp
% hold on;
% plot3([mainNormal(1),0],[mainNormal(2),0],[mainNormal(3),0],'r-','LineWidth',4);
% hold on;

dotMap = mainNormal(1)*xCh+mainNormal(2)*yCh+mainNormal(3)*zCh;
dotMap = abs(dotMap);

[~, ps] = mapminmax(dotMap);
ps.ymin = 0;
[dotMapN, ~] = mapminmax(dotMap, ps);
DepthMap = dotMapN;
DepthMap = vl_imsmooth(DepthMap, 5);

[DIx, DIy] = gradient(DepthMap);
DH11 = vl_imsmooth(DIx.*DIx, si);
DH12 = vl_imsmooth(DIx.*DIy, si);
DH22 = vl_imsmooth(DIy.*DIy, si);

Ddt = DH11.*DH22 - DH12.^2;
Dtr = DH11+DH22;
R_depth = Ddt-alpha*(Dtr.^2);

R = R_rgb + belt*R_depth;

Rmax = max(max(R));
idx = vl_localmax(R);
[v, u] = ind2sub(size(I), idx);
num = size(v, 2);
index = 1;
for i = 1:num
    if R(v(i), u(i)) > gama*Rmax
        keypoints(1, index) = u(i);
        keypoints(2, index) = v(i);
        index = index+1;
    end
end

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
pcloud(:, :, 1) = xgrid.*depth/focal/MM_PER_M;
pcloud(:, :, 2) = ygrid.*depth/focal/MM_PER_M;
pcloud(:, :, 3) = depth/MM_PER_M;
end


function normal = get_pc_normal(pcloud, win, threshold)
% Compute surface normals by PCA

[imh, imw, ~] = size(pcloud);
normal = zeros(size(pcloud));
for i = 1:imh
    for j = 1:imw
        minh = max(i - win, 1);
        maxh = min(i + win, size(pcloud, 1));
        minw = max(j - win, 1);
        maxw = min(j + win, size(pcloud, 2));
        index = abs(pcloud(minh:maxh, minw:maxw, 3) - pcloud(i, j, 3)) < pcloud(i, j, 3)*threshold;
        if sum(index(:)) > 3 && pcloud(i, j, 3) > 0   % the minimum number of points required
            wpc = reshape(pcloud(minh:maxh, minw:maxw, :),  (maxh-minh+1)*(maxw-minw+1), 3);
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


function [mainNormal] = get_main_normal(normals)
% Compute the main normal

nbin = 4;
histgrom = zeros(nbin, nbin, nbin);

[height, width, ~] = size(normals);
for i = 1:height
    for j = 1:width
        normal(1) = normals(i, j, 1);
        normal(2) = normals(i, j, 2);
        normal(3) = normals(i, j, 3);
        if normal == [0, 0, 0]
            continue;
        else
            angle = abs(acos(normal));
            ids = ceil(angle*4/pi);
            ids(find(ids>4)) = 4;
            ids(find(ids<1)) = 1;
            histgrom(ids(1), ids(2), ids(3)) = histgrom(ids(1), ids(2), ids(3))+1;
        end
    end
end

[~, indmax] = max(histgrom(:));
[i, j, w] = ind2sub(size(histgrom), indmax);
angels = [i, j, w]*pi/4;
mainNormal = cos(angels);
end
