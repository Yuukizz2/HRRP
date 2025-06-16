clc
clear
close all;
dirs = 'D:\博士研究课题\hrrp\datasets\rawHRRP0725\new\';
%filename = 'Z8_654.mat';
files = dir([dirs,'*.mat']);
for i = 1:length(files)
    data = load(files(i).name);
    res0 = data.newcell;
    res = cell2mat(res0(:, 5));
    [r,c] = size(res);
    fig = figure;
    for j = 1:20
        plot(res(j,:))
        xlim([0 512])
        xlabel('Range Cell')
        ylabel('Normalized Amplitude')
        title([res0{j,6}, ', 方位角=', num2str(round(res0{j,1}, 1)), '°','俯仰角=', num2str(round(res0{j,2}, 1)),'°'], 'FontWeight', 'bold')
        frame = getframe(fig);
        im{j} = frame2im(frame);
    end
    close;
    filename = [dirs,'GIF\',files(i).name(1:end-4),'.gif'];
    for idx = 1:20
        [A,map] = rgb2ind(im{idx},256);
        if idx == 1
            imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',0.5);
        else
            imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',0.5);
        end
    end

end