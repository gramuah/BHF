function eval_results(bindataDir)
% bindataDir --> bindata dir with the results
addpath([cd '/data']);
BHFdetectionDir = [bindataDir, '/detections']; % output Dir

disp('Reading detections');
myFramework(bindataDir, BHFdetectionDir);
%non maximum-supression of the detections
overlap = 0.3;
disp('Performing Non-maximum supression');
run_NMS(overlap, BHFdetectionDir);

%% Show Prec/recall curve
minoverlap=0.5;
groundTruthFile=['/test_all_gt.txt'];
disp('Obtaining results: prec/recall curves, etc.');
evaldetpose(BHFdetectionDir, groundTruthFile, minoverlap);

end

function myFramework(bindataDir,BHFDataDir)


mkdir([BHFDataDir]);

[imNames] = textread(['test_images.txt'],'%s');

for i=1:length(imNames)
    mkdir([ BHFDataDir, '/', imNames{i}]);
    fidC = fopen(sprintf('%s/%s/candidates.txt', BHFDataDir, imNames{i}),'w');
    fidBB = fopen(sprintf('%s/%s/boundingboxes.txt', BHFDataDir, imNames{i}),'w');
   
    datb = importdata(sprintf('%s/bboxes/bboxes_testimg_%s.txt', bindataDir, num2str(i-1)), ' ', 1);
    datp = importdata(sprintf('%s/bboxes/pose_testimg_%s.txt', bindataDir, num2str(i-1)), ' ', 1);
    
    fprintf(fidC, '%d\n', str2num(cell2mat(datb.textdata)));
    fprintf(fidBB, '%d\n', str2num(cell2mat(datb.textdata)));
    for j=1:str2num(cell2mat(datb.textdata))
        fprintf(fidC, '%.4f %.2f %.2f %.2f %d %d %d %d\n', datb.data(j,5), ...
            datb.data(j,1)+(datb.data(j,3)/2), ...
            datb.data(j,2)+(datb.data(j,4)/2), ...
            1, j-1, datp.data(j,1),datp.data(j,2),datp.data(j,3));
        fprintf(fidBB, '%.2f %.2f %.2f %.2f\n', datb.data(j,1),datb.data(j,2),...
            datb.data(j,1)+datb.data(j,3),datb.data(j,2)+datb.data(j,4));
    end
end
end

function run_NMS(overlap, BHFdetectionDir)

detections=dir([BHFdetectionDir, '/*.jpg']);
Ndetections=length(detections);

for n=1:Ndetections
    %% Load data
    filenameBBox= fullfile (sprintf('%s/%s/boundingboxes.txt',BHFdetectionDir, detections(n).name));
    filenameCand= fullfile (sprintf('%s/%s/candidates.txt',BHFdetectionDir, detections(n).name));
    
    
    dataBBox = textread(filenameBBox);
    dataCand = textread(filenameCand);

    mm=1;

    boxes=[];
    centers=[];
    score=[];
    pose=[];
    scale=[];
    for m=2:(size(dataBBox,1))
     
        boxes(mm,:) = [dataBBox(m,1) dataBBox(m,2) dataBBox(m,3) dataBBox(m,4)];
        centers(mm,:) = [dataCand(m,2) dataCand(m,3)];
        score(mm) = dataCand(m,1); 
        pose(mm) = dataCand(m,6);
        zenith(mm) = dataCand(m,7);
        scale(mm) = dataCand(m,4);
        trainIDimg(mm) = dataCand(m,8);
        mm=mm+1;
    end
    
    bbs=[boxes score'];
    pick=[];
    %% Non-maximum suppression
    pick = nms(bbs, overlap);
    [value ind]=sort(score(pick),'descend');
    % Save the results
    fid=fopen(fullfile (sprintf('%s/%s/Results.txt',BHFdetectionDir, detections(n).name)),'w');
    for m =1:length(pick)
        bb = boxes(pick(ind(m)),:);
        if bb(1)<0
             bb(1)=0;
        end
        if bb(2)<0
             bb(2)=0;
        end
        fprintf(fid,'%1.4f %3.2f %3.2f %3.2f %3.2f %3.2f %3.2f %d %d %d\n', score(pick(ind(m))), ...
            centers(pick(ind(m)),1), centers(pick(ind(m)),2), bb(1), bb(2), bb(3), ...
                bb(4), pose(pick(ind(m))),zenith(pick(ind(m))), trainIDimg(pick(ind(m))));


    end
    
    fclose ('all');
   
end
 
end

function evaldetpose(DetectionsDir, groundTruthFile, minoverlap)

% load gt
[gtids, bb1, bb2, bb3, bb4, gtpose, gtzenith]=textread(groundTruthFile,'%s %f %f %f %f %d %d');
gtbbox=[bb1, bb2, bb3, bb4]';
for ii=1:length(gtids)
    gt(ii).det=false;
end
npos=length(gtids);

% load results

detections=dir([DetectionsDir, '/*jpg']);
Ndetections=length(detections);
h=1;

for i=1:Ndetections
   
    if exist([DetectionsDir, '/',detections(i).name, '/Results.txt'],'file')
        fprintf('%\n', detections(i).name);
        [s c1 c2 bb1 bb2 bb3 bb4 po ze sc]=textread([DetectionsDir, '/',detections(i).name, '/Results.txt']);
        [Nc]=size(s,1);
		
        for a=1:Nc
            [p, ids{h}(1,:), ext]=fileparts(detections(i).name);
            confidence(h)= s(a);
            bb(h,:)=[bb1(a) bb2(a) bb3(a) bb4(a)];
            centers(h,:) = [c1(a) c2(a)];
            pose(h)=po(a);
            zenith(h)=ze(a);
            
            h=h+1;
        end
    end
end

BB=bb';
% sort detections by decreasing confidence
[sc,si]=sort(-confidence);
ids=strcat(ids(si), '.jpg');
BB=BB(:,si);
centers = centers(si,:);
pose = pose(si);
zenith = zenith(si);

% assign detections to ground truth objects
nd=length(confidence);
tp=zeros(nd,1);
fp=zeros(nd,1);

tp_pose=zeros(nd,1);
fp_pose=zeros(nd,1);

tic;
dd = 1;
c=1;
for d=1:nd
    % display progress
    if toc>1
        fprintf('%s: pr: compute: %d/%d\n',cls,d,nd);
        drawnow;
        tic;
    end
    
    % find ground truth image
    i=strmatch(ids{d},gtids,'exact');
  
    if isempty(i)
        fprintf('unrecognized image "%s"',ids{d});
    elseif length(i)>1
        fprintf('multiple image "%s"',ids{d});
    end

    % assign detection to ground truth object if any
    bb=BB(:,d);
    ovmax=-inf;
    bbgt=gtbbox(:,i);
    bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
    iw=bi(3)-bi(1)+1;
    ih=bi(4)-bi(2)+1;
    if iw>0 & ih>0                
       % compute overlap as area of intersection / area of union
        ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
        (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
        iw*ih;
        ov=iw*ih/ua;
        if ov>ovmax
          ovmax=ov;
        end
    end
    % assign detection as true positive/don't care/false positive
    if ovmax>=minoverlap
        if ~gt(i).det(1)
          tp(d)=1;            % true positive
          gt(i).det(1)=true;
          error(dd)=min(mod(pose(d)-gtpose(i),360), mod(gtpose(i)-pose(d),360));
          error_zenith(dd)=min(mod(zenith(d)-gtzenith(i),360), mod(gtzenith(i)-zenith(d),360));
          dd=dd+1;
       
       else
          fp(d)=1;            % false positive (multiple detection)
       end
        
    else
   
    
      fp(d)=1;                    % false positive
      
    end

end
x=0:10:170;
aux=[length(find(error<10)), length(find(error<20))-length(find(error<10)), length(find(error<30))-length(find(error<20)), length(find(error<40))-length(find(error<30)), ...
    length(find(error<50))-length(find(error<40)), length(find(error<60))-length(find(error<50)), length(find(error<70))-length(find(error<60)), length(find(error<80))-length(find(error<70)), ...
    length(find(error<90))-length(find(error<80)), length(find(error<100))-length(find(error<90)), length(find(error<110))-length(find(error<100)), ...
    length(find(error<120))-length(find(error<110)), length(find(error<130))-length(find(error<120)), length(find(error<140))-length(find(error<130)), ...
    length(find(error<150))-length(find(error<140)), length(find(error<160))-length(find(error<150)), length(find(error<170))-length(find(error<160)), ...
    length(find(error>=170))];
error_aux=sort(error);
%%%% Thresholds
MedianAE=error_aux(round((length(error)+1)/2));
MeanAE=sum(error)/length(error);
StdAE=std(error);
figure()
bar(x, aux);
set(gca,'xTick',[0:10:170])
text(90,200,sprintf('MeanAE=%.1f\nMedianAE=%.1f\nStdAE=%.1f', MeanAE, MedianAE,StdAE),'FontSize',18)
title('azimuth')
axis([-10 180 0 350]);

aux1=[length(find(error_zenith<10)), length(find(error_zenith<20))-length(find(error_zenith<10)), length(find(error_zenith<30))-length(find(error_zenith<20)), ...
    length(find(error_zenith<40))-length(find(error_zenith<30)), ...
    length(find(error_zenith<50))-length(find(error_zenith<40)), length(find(error_zenith<60))-length(find(error_zenith<50)), ...
    length(find(error_zenith<70))-length(find(error_zenith<60)), length(find(error_zenith<80))-length(find(error_zenith<70)), ...
    length(find(error_zenith<90))-length(find(error_zenith<80)), length(find(error_zenith<100))-length(find(error_zenith<90)), ...
    length(find(error_zenith<110))-length(find(error_zenith<100)), ...
    length(find(error_zenith<120))-length(find(error_zenith<110)), length(find(error_zenith<130))-length(find(error_zenith<120)), ...
    length(find(error_zenith<140))-length(find(error_zenith<130)), ...
    length(find(error_zenith<150))-length(find(error_zenith<140)), length(find(error_zenith<160))-length(find(error_zenith<150)), ...
    length(find(error_zenith<170))-length(find(error_zenith<160)), length(find(error_zenith>=170))];
error_aux=sort(error_zenith);
MedianAE=error_aux(round((length(error_zenith)+1)/2));
MeanAE=sum(error_zenith)/length(error_zenith);
StdAE=std(error_zenith);
figure()
bar(x, aux1);
set(gca,'xTick',[0:10:170])
text(90,200,sprintf('MeanAE=%.2f\nMedianAE=%.2f\nStdAE=%.2f', MeanAE, MedianAE,StdAE),'FontSize',18)
title('zenith')
axis([-10 180 0 550]);


ind_fp=find(fp);
ind_tp=find(tp);
% compute precision/recall in detection
fp=cumsum(fp);
tp=cumsum(tp);
recD=tp/npos;
precD=tp./(fp+tp);




% compute average orientation similarity (AOS) for the azimuth
tmp(ind_fp)=0;
for h=1:length(error)
    tmp(ind_tp(h))=(1 + cos((error(h)*pi)/180))/2;
end
acc=cumsum(tmp);
accD=acc./(fp+tp)';
AOS=0;
for t=0:0.1:1
    p=max(accD(recD>=t));
    if isempty(p)
        p=0;
    end
    AOS=AOS+p/11;
end

% compute AOS for the zenith
tmp_z(ind_fp)=0;
for h=1:length(error_zenith)
    tmp_z(ind_tp(h))=(1 + cos((error_zenith(h)*pi)/180))/2;
end
acc=cumsum(tmp_z);
accZ=acc./(fp+tp)';
AOS_z=0;
for t=0:0.1:1
    p=max(accZ(recD>=t));
    if isempty(p)
        p=0;
    end
    AOS_z=AOS_z+p/11;
end



% compute AP for detection
apD=0;
for t=0:0.1:1
    p=max(precD(recD>=t));
    if isempty(p)
        p=0;
    end
    apD=apD+p/11;
end

result.prec = precD;
result.rec = recD;
result.ap = apD;

figure()
% plot precision/recall curves
plot(recD,accD,'-r','LineWidth',2);
hold on;
plot(recD,accZ,'-.g','LineWidth',2);
plot(recD,precD,'--b','LineWidth',2);
grid;
legend('Azimuth precision', 'Zenith precision','Detection precision','Location','southwest')
xlabel 'recall'
ylabel 'precision'
title(sprintf('AP(det) = %.4f ; AOS(azimuth) = %.4f ; AOS(zenith) = %.4f', apD,AOS, AOS_z));
axis([0 1 0 1]);

end


