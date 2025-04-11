CREATE TABLE fundus (
    id INTEGER PRIMARY KEY,
    age INTEGER,
    gender CHAR, -- m for male, f for female
    left_path NCHAR(260), -- MAX_PATH
    right_path NCHAR(260), -- MAX_PATH
    normal_fundus BOOLEAN DEFAULT 0, -- 正常眼底
    diabetic BOOLEAN DEFAULT 0, -- 糖尿病
    glaucoma BOOLEAN DEFAULT 0, -- 青光眼
    cataract BOOLEAN DEFAULT 0, -- 白内障
    age_related_macular_degeneration BOOLEAN DEFAULT 0, -- 老年性黄斑变性
    hypertensive_retinopathy BOOLEAN DEFAULT 0, -- 高血压
    myopia BOOLEAN DEFAULT 0, -- 近视
    other_diseases BOOLEAN DEFAULT 0, -- 其他疾病
    low_quality_image BOOLEAN DEFAULT 0, -- 低质量照片
    is_test BOOLEAN DEFAULT 0 -- 是否为测试数据
);