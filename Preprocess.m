function [X, Y, Xpreprocess, Y_all] = Preprocess()
    %%% 数据导入与预处理
    SamplePath = fullfile(pwd,'Spectra Data');
    files = dir(SamplePath);     % 获取Spectra Data文件夹下的所有文件信息
    num_file = size(files,1);    % 获取文件数量

    j =0;
    Str_mes = {}; % 存储Mes开头的文件名
    statistics_all = [];  % 存储所有样本的统计数据

    % 筛选Mes开头的文件名并去除前缀"Mes"
    for i = 3:num_file
       if strncmp(files(i).name,'Mes',3)
          Str_mes{j+1} =  files(i).name(5:end); 
          j = j+1;
       end
    end
    num_mes_file  = (j);   

    % 选择特定的数据集
    selected_str = {'Data_6-8-7-P1-P3.csv','Data_9-1-20-P1-P3.csv','Data_10-1-41-P1-P3.csv','Data_10-42-70-P1-P3.csv','Data_11-1-35-P1.csv','Data_11-1-35-P2-P3.csv','Data_11-36-50-P1-P3.csv','Data_11-90-123-P1-P3.csv'};
    num_dataset = length(selected_str);  

    mean_sample_all_spectra_all = []; % 存储所有样本的平均光谱数据

    % 光谱数据预处理参数
    for i_num_dataset = 1:num_dataset
        % 对于每个选定的数据集，找到对应的Mes文件
        selected_spectra = selected_str{i_num_dataset};
        selected_str_mes = '';
        for i=1:num_mes_file
            str_mes_i = Str_mes{i};
            if strcmp(selected_spectra(6:end-9),str_mes_i(1:end-9))    
                selected_str_mes = str_mes_i; 
            end
        end
        
        if isempty(selected_str_mes)
            msgbox('未找到匹配标签的光谱数据文件','错误','error') 
            break;
        end

        % 标签文件名
        str_mes =  [SamplePath '\Mes_' selected_str_mes]; 

        % 导入数据
        Data_struct = importdata(str_mes); 
        Data = Data_struct.data; 
        statistics = Data(:,2:4); 
        % statistics = roundn(statistics,-4); % 替换为通用写法
        statistics = round(statistics * 10000) / 10000;
        % statistics_all = [statistics_all;statistics];  
        
        % 导入Data数据并进行预处理
        str = [SamplePath '\' selected_str{i_num_dataset}]; 
        data = csvread(str,13,1);  
        wavelength = data(1,:);    
        data_col_1 = sum(data,2);  
        data_interval = find(data_col_1 == 0);  
        num_sample = length(data_interval);     

        %%% 新增：解决Xprocess超出界限
        % 确保 statistics 行数与 num_sample 一致
        if size(statistics, 1) > num_sample
             statistics = statistics(1:num_sample, :);
        end
        statistics_all = [statistics_all;statistics];  
        %%% 新增：解决Xprocess超出界限   

        % 波长范围
        wavelength_start = 563;
        wavelength_end = 1110;

        [~, location_wavelength_start] = min(abs(wavelength - wavelength_start));
        [~, location_wavelength_end] = min(abs(wavelength - wavelength_end));
        
        mean_sample_all_spectra = []; 
      
        % 光谱强度范围  
        intensity_start = 0;
        intensity_end = 60000;
         
        % 处理csv文件中的光谱数据
        for i=1:num_sample
            if i==num_sample
                ith_sample_spectra = data(data_interval(i)+1:end-1,location_wavelength_start:location_wavelength_end); 
                [Maxvalue,~] = max(ith_sample_spectra,[],2); 
                DeletPointPosition = find(Maxvalue > intensity_end | Maxvalue< intensity_start);
                ith_sample_spectra(DeletPointPosition,:) = [];
                
                Stvalue = ith_sample_spectra(:,10); 
                DeletPointPosition = find(Stvalue > 30000);
                ith_sample_spectra(DeletPointPosition,:) = [];
                
                mean_sample_all_spectra(i,:) = mean(ith_sample_spectra,1);
            else
                ith_sample_spectra =  data(data_interval(i)+1:data_interval(i+1)-2,location_wavelength_start:location_wavelength_end);    
                [Maxvalue,~] = max(ith_sample_spectra,[],2);  
                DeletPointPosition = find(Maxvalue > intensity_end | Maxvalue< intensity_start);
                ith_sample_spectra(DeletPointPosition,:) = [];
                
                Stvalue = ith_sample_spectra(:,10); 
                DeletPointPosition = find(Stvalue > 30000);
                ith_sample_spectra(DeletPointPosition,:) = [];
                
                mean_sample_all_spectra(i,:) = mean(ith_sample_spectra,1);
            end
        end
        mean_sample_all_spectra_all= [mean_sample_all_spectra_all; mean_sample_all_spectra];
    end

    %%% 光谱预处理
    Xpreprocess = mean_sample_all_spectra_all;

    %%% 移动平均滤波
    segment = 29;   
    Xpreprocess = average_moving(Xpreprocess,segment);

    %%% 一阶导数
    [~, ~,~,~,Xpreprocess,~]=FirstDerivative(Xpreprocess,Xpreprocess);

    % 获取标签
    Y_all = statistics_all(:,3); 

    % 筛选：只保留标签为1(正常)和7(水脱)
    valid_idx = find(Y_all == 1 | Y_all == 7);
    Y_all = Y_all(valid_idx);
    Xpreprocess = Xpreprocess(valid_idx, :);

    %%% 数据平衡处理
    num_normal = find(Y_all==1);  
    num_moldy = find(Y_all==7);   
    X_normal = Xpreprocess(num_normal,:);  
    Y_normal = Y_all(num_normal,:);            
    X_moldy = Xpreprocess(num_moldy,:);    
    Y_moldy = Y_all(num_moldy,:);              

    n_normal = length(num_normal);
    n_moldy = length(num_moldy);

    if n_normal > n_moldy 
        % 复制水脱样本（过采样）
        rep_count = ceil(n_normal / n_moldy);
        X_moldy_over = repmat(X_moldy, rep_count, 1);
        Y_moldy_over = repmat(Y_moldy, rep_count, 1);
        X_moldy_over = X_moldy_over(1:n_normal, :);
        Y_moldy_over = Y_moldy_over(1:n_normal, :);
        
        X = [X_normal; X_moldy_over];
        Y = [Y_normal; Y_moldy_over];

    elseif n_normal < n_moldy 
        % 复制正常样本（过采样）
        rep_count = ceil(n_moldy / n_normal);
        X_normal_over = repmat(X_normal, rep_count, 1);
        Y_normal_over = repmat(Y_normal, rep_count, 1);
        X_normal_over = X_normal_over(1:n_moldy, :);
        Y_normal_over = Y_normal_over(1:n_moldy, :);
        
        X = [X_normal_over; X_moldy];
        Y = [Y_normal_over; Y_moldy];

    else 
        X = [X_normal ;X_moldy];
        Y = [Y_normal;Y_moldy];
    end

    disp(['X的大小: ', num2str(size(X))]);
    disp(['Y的大小: ', num2str(size(Y))]);
end
