clearvars;
PATH='./';
load(strcat(PATH,'mpii_human_pose_v1_u12_1.mat'));
clear PATH;

annolist=RELEASE.annolist;
img_train=RELEASE.img_train;
version=RELEASE.version;
single_person=RELEASE.single_person;
act=RELEASE.act;
video_list=RELEASE.video_list;
clear RELEASE;

outfile = fopen('RELEASE.txt', 'w');
fprintf(outfile, 'version: ');
fwrite(outfile, version);
fprintf(outfile, '\n');
fprintf(outfile, 'act: act.csv\n');
fprintf(outfile, 'annolist: annolist.csv\n');
fprintf(outfile, 'annorect: annorect.csv\n');
fprintf(outfile, 'img_train: img_train.csv\n');
fprintf(outfile, 'single_person: single_person.csv\n');
fprintf(outfile, 'video_list: video_list.csv\n');
fclose(outfile);

% Write fields of RELEASE to file (in ascending complexity)

% 1. img_train
writematrix(img_train', 'img_train.csv');
add_header('img_train.csv', 'train\n');

% 2. video_list
writecell(video_list', 'video_list.csv');
add_header('video_list.csv', 'video_id\n');

% 3. single_person
writecell(single_person,'single_person.csv');
add_header('single_person.csv', 'p0,p1,p2,p3,p4,p5,p6,p7,p8\n');

% 4. act
outfile=fopen('act.csv', 'w');
fprintf(outfile, "cat_name; act_name; act_id\n");
for i=1:length(act)
    % CSV file separator is ";" since there are commas in field values
    fprintf(outfile, "%s;%s;%d\n", ...
        act(i).cat_name, act(i).act_name, act(i).act_id);
end
fclose(outfile);

% 5. annolist
% This structure contains (potentially) multiple annotations of varying
% size inside of each entry. Let's set up another table to keep track of
% the annorect objects in each. We'll use annorect_id so we aren't
% dependent on a 0-based or 1-based index system (but in principle, we're
% matching the 1-based index)
annolist_id=1;
annolist_out=fopen('annolist.csv','w');
annorect_out=fopen('annorect.csv','w');
fprintf(annolist_out, "annolist_id, filename, frame_sec, vididx\n");
fprintf(annorect_out, "annolist_id, scale, objpos_x, objpos_y, x1, y1, x2, y2, " + ...
    "pt00_x, pt00_y, pt00_v, " + ...
    "pt01_x, pt01_y, pt01_v, " + ...
    "pt02_x, pt02_y, pt02_v, " + ...
    "pt03_x, pt03_y, pt03_v, " + ...
    "pt04_x, pt04_y, pt04_v, " + ...
    "pt05_x, pt05_y, pt05_v, " + ...
    "pt06_x, pt06_y, pt06_v, " + ...
    "pt07_x, pt07_y, pt07_v, " + ...
    "pt08_x, pt08_y, pt08_v, " + ...
    "pt09_x, pt09_y, pt09_v, " + ...
    "pt10_x, pt10_y, pt10_v, " + ...
    "pt11_x, pt11_y, pt11_v, " + ...
    "pt12_x, pt12_y, pt12_v, " + ...
    "pt13_x, pt13_y, pt13_v, " + ...
    "pt14_x, pt14_y, pt14_v, " + ...
    "pt15_x, pt15_y, pt15_v\n");
for i=1:length(annolist)
    % First, deal with the top-level stuff: filename, frame_sec and vididx
    fprintf(annolist_out, "%d, ", annolist_id);
    fprintf(annolist_out, "%s, ", annolist(i).image.name);
    if isempty(annolist(i).frame_sec)
        fprintf(annolist_out, ", ");
    else
        fprintf(annolist_out, "%d, ", annolist(i).frame_sec);
    end
    if isempty(annolist(i).vididx)
        fprintf(annolist_out, "\n");
    else
        fprintf(annolist_out, "%d\n", annolist(i).vididx);
    end


    % Then, deal with the annotations (may be multiple, so need to loop)
    for j=1:length(annolist(i).annorect)
        fprintf(annorect_out, "%d, ", annolist_id);

        if ismember('scale', fieldnames(annolist(i).annorect))
            fprintf(annorect_out, "%f", annolist(i).annorect(j).scale);
        end
        fprintf(annorect_out, ", ");

        % Next two: obj position
        if ismember('objpos', fieldnames(annolist(i).annorect))
            if false == isempty(annolist(i).annorect(j).objpos)
                fprintf(annorect_out, "%d, ", annolist(i).annorect(j).objpos.x);
                fprintf(annorect_out, "%d, ", annolist(i).annorect(j).objpos.y);
            else
                fprintf(annorect_out, ", , ");
            end
        else
            fprintf(annorect_out, ", , ");
        end

        % Next four: head bounding box
        if ismember('x1', fieldnames(annolist(i).annorect))
            fprintf(annorect_out, "%d", annolist(i).annorect(j).x1);
        end
        fprintf(annorect_out, ", ");

        if ismember('y1', fieldnames(annolist(i).annorect))
            fprintf(annorect_out, "%d", annolist(i).annorect(j).y1);
        end
        fprintf(annorect_out, ", ");

        if ismember('x2', fieldnames(annolist(i).annorect))
            fprintf(annorect_out, "%d", annolist(i).annorect(j).x2);
        end
        fprintf(annorect_out, ", ");

        if ismember('y2', fieldnames(annolist(i).annorect))
            fprintf(annorect_out, "%d", annolist(i).annorect(j).y2);
        end
        fprintf(annorect_out, ", ");

        % The rest are the joints. Not all are present, but we'll flatten
        % the array and always encode 16 joints. Format will be a triplet:
        % x, y, v, where v is the visibility (empty for joints 8, 9, pelvis
        % and thorax)
        for k=0:15
            if ismember('annopoints', fieldnames(annolist(i).annorect)) && (false==isempty(annolist(i).annorect(j).annopoints))
                            %annolist(i).annorect(j).annopoints
                % search for each joint index
                found=false;
                for l=1:length(annolist(i).annorect(j).annopoints.point)
                    if annolist(i).annorect(j).annopoints.point(l).id == k
                        found=true;
                        break
                    end
                end
                if found
                    if k == 8 || k == 9

                        % Argh! Sometimes the data is encoded in binary and
                        % sometimes in ASCII!
                        if false == ismember('is_visible', fieldnames(annolist(i).annorect(j).annopoints.point))
                            % Very specific corner-case here... sample 24731
                            is_visible = [];
                            % Also, script pukes if you try to get rid of
                            % the warning in the following two conditions
                            % by replacing with the short-circuit operator
                        elseif annolist(i).annorect(j).annopoints.point(l).is_visible == 0 | annolist(i).annorect(j).annopoints.point(l).is_visible == '0'
                            is_visible = 0;
                        elseif annolist(i).annorect(j).annopoints.point(l).is_visible == 1 | annolist(i).annorect(j).annopoints.point(l).is_visible == '1'
                            is_visible = 1;
                        elseif isempty(annolist(i).annorect(j).annopoints.point(l).is_visible)
                            is_visible = annolist(i).annorect(j).annopoints.point(l).is_visible;
                        else
                            fprintf("WHATWHAT1");
                        end

                        fprintf(annorect_out, "%f, %f, %d", ...
                            annolist(i).annorect(j).annopoints.point(l).x, ...
                            annolist(i).annorect(j).annopoints.point(l).y, ...
                            is_visible);
                    else
                        % Argh! Sometimes the data is encoded in binary and
                        % sometimes in ASCII!
                        if annolist(i).annorect(j).annopoints.point(l).is_visible == 0 || annolist(i).annorect(j).annopoints.point(l).is_visible == '0'
                            is_visible = 0;
                        elseif annolist(i).annorect(j).annopoints.point(l).is_visible == 1 || annolist(i).annorect(j).annopoints.point(l).is_visible == '1'
                            is_visible = 1;
                        elseif isempty(annolist(i).annorect(j).annopoints.point(l).is_visible)
                            is_visible = annolist(i).annorect(j).annopoints.point(l).is_visible;
                        else
                            fprintf("WHATWHAT2");
                        end

                        fprintf(annorect_out, "%d, %d, %d", ...
                            annolist(i).annorect(j).annopoints.point(l).x, ...
                            annolist(i).annorect(j).annopoints.point(l).y, ...
                            is_visible);
                    end
                else
                    fprintf(annorect_out, ", , ");
                end
            else
                fprintf(annorect_out, ", , ");
            end
            if k ~= 15
                fprintf(annorect_out, ", ");
            end

        end % for

        fprintf(annorect_out, "\n");
    end
    annolist_id = annolist_id + 1;
end
fclose(outfile);

