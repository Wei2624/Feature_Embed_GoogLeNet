function check_images(image_dir, image_list)
                            
for i_img = 1:length(image_list)
    
    if mod(i_img, 1000)==0
        fprintf('%d/%d\n', i_img, length(image_list));
    end
    
    filename = fullfile(image_dir, image_list{i_img});
    
    try
        orig_img = imread(filename);
    catch
        disp(filename);
    end
end