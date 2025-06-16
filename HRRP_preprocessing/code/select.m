clc
clear
close all;
%dir = 'D:\博士研究课题\hrrp\datasets\rawHRRP0725\';
dir = 'D:\博士研究课题\hrrp\datasets\rawHRRP0725_div_ang\concat\';
%filename = 'BY737_0322_7.mat';
filename = 'BY737_7790.mat';
data = load([dir, filename]);
res = data.HRRP_Fragment;
[row,col] = size(res);
index = [];
for i = 1:row
    if res{i,1} >= 70 && res{i,1} <= 85 && res{i,2} >= 13.5 && res{i,2} <= 15.5
        index = [index, i];
    end
end
HRRP_Fragment = res(index,:);
[newrow,newcol] = size(HRRP_Fragment);
%save(['D:\博士研究课题\hrrp\datasets\rawHRRP0725\new\',num2str(newrow),'_', filename], 'newcell')
%save(['D:\博士研究课题\hrrp\datasets\rawHRRP0725\new4FSLB5N3\',num2str(newrow),'_', filename], 'newcell')
save(['D:\博士研究课题\hrrp\datasets\rawHRRP0725_div_ang\concat\new_B4N4_GFSL\','train_testseen_', strtok(filename,'_'), '_', num2str(newrow),'.mat'], 'HRRP_Fragment')
%save(['D:\博士研究课题\hrrp\datasets\rawHRRP0725_div_ang\concat\new_B4N4_GFSL\','val_', strtok(filename,'_'), '_', num2str(newrow),'.mat'], 'HRRP_Fragment')
